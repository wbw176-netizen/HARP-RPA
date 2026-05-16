import math

import torch
from torch import nn

from models.components.pairformer_lite import TriangleMultiplication, Transition, make_pair_mask
from models.components.rope import RotaryPositionEmbedding


def _default_relation_ids(pair, relation_ids):
    if relation_ids is not None:
        return relation_ids
    return torch.zeros(pair.shape[:3], device=pair.device, dtype=torch.long)


class HeteroPairBiasedSelfAttention(nn.Module):
    def __init__(self, embed_dim, pair_dim, num_heads, num_relations=8, attention_dropout=0.0, use_rot_emb=True, bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimensionality must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rot_emb = use_rot_emb

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.pair_bias_norm = nn.LayerNorm(pair_dim)
        self.pair_bias = nn.Linear(pair_dim, num_heads, bias=False)
        self.relation_bias = nn.Embedding(num_relations, num_heads)
        self.attention_dropout = nn.Dropout(attention_dropout)

        nn.init.zeros_(self.relation_bias.weight)

        if use_rot_emb:
            self.rotary_emb = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x, pair, relation_ids=None, key_padding_mask=None, attn_mask=None, need_attn_weights=False):
        batch_size, seq_len, _ = x.shape
        relation_ids = _default_relation_ids(pair, relation_ids)

        q = self.to_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rot_emb:
            q, k = self.rotary_emb(q, k)

        pair_bias = self.pair_bias(self.pair_bias_norm(pair)).permute(0, 3, 1, 2)
        relation_bias = self.relation_bias(relation_ids).permute(0, 3, 1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = logits + pair_bias + relation_bias

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None, :, :]
            logits = logits.masked_fill(attn_mask.bool(), torch.finfo(logits.dtype).min)

        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :].bool(), torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        attn = torch.nan_to_num(attn)
        attn = self.attention_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)

        if key_padding_mask is not None:
            out = out * (~key_padding_mask).unsqueeze(-1).to(out.dtype)

        return out, attn if need_attn_weights else None


class RelationConditionedSingleToPairUpdate(nn.Module):
    def __init__(self, embed_dim, pair_dim, c_hidden=32, num_relations=8, bias=True):
        super().__init__()
        self.left_proj = nn.Linear(embed_dim, c_hidden, bias=bias)
        self.right_proj = nn.Linear(embed_dim, c_hidden, bias=bias)
        self.out_proj = nn.Linear(c_hidden, pair_dim, bias=bias)
        self.gate = nn.Linear(pair_dim, pair_dim, bias=bias)
        self.relation_gate = nn.Embedding(num_relations, pair_dim)
        self.relation_bias = nn.Embedding(num_relations, pair_dim)

        nn.init.zeros_(self.relation_gate.weight)
        nn.init.zeros_(self.relation_bias.weight)

    def forward(self, x, pair, relation_ids=None, pair_mask=None):
        relation_ids = _default_relation_ids(pair, relation_ids)
        left = self.left_proj(x)
        right = self.right_proj(x)
        update = torch.einsum("bih,bjh->bijh", left, right) / math.sqrt(left.shape[-1])
        update = self.out_proj(update)
        gate = torch.sigmoid(self.gate(pair) + self.relation_gate(relation_ids))
        update = update * gate + self.relation_bias(relation_ids)

        if pair_mask is not None:
            update = update * pair_mask.unsqueeze(-1).to(update.dtype)

        return update


class RelationConditionedTriangle(nn.Module):
    def __init__(self, pair_dim, c_hidden, mode="outgoing", num_relations=8, bias=True):
        super().__init__()
        self.base = TriangleMultiplication(pair_dim, c_hidden, mode=mode, bias=bias)
        self.relation_gate = nn.Embedding(num_relations, pair_dim)
        self.relation_bias = nn.Embedding(num_relations, pair_dim)

        nn.init.zeros_(self.relation_gate.weight)
        nn.init.zeros_(self.relation_bias.weight)

    def forward(self, pair, relation_ids=None, pair_mask=None):
        relation_ids = _default_relation_ids(pair, relation_ids)
        update = self.base(pair, pair_mask=pair_mask)
        update = update * (1.0 + torch.tanh(self.relation_gate(relation_ids)))
        update = update + self.relation_bias(relation_ids)

        if pair_mask is not None:
            update = update * pair_mask.unsqueeze(-1).to(update.dtype)

        return update


class RelationConditionedPairTransition(nn.Module):
    def __init__(self, pair_dim, transition_factor=4, dropout=0.0, num_relations=8):
        super().__init__()
        hidden_dim = int(2 / 3 * transition_factor * pair_dim)
        self.in_proj = nn.Linear(pair_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, pair_dim)
        self.dropout = nn.Dropout(dropout)
        self.relation_scale = nn.Embedding(num_relations, hidden_dim)
        self.relation_bias = nn.Embedding(num_relations, hidden_dim)

        nn.init.zeros_(self.relation_scale.weight)
        nn.init.zeros_(self.relation_bias.weight)

    def forward(self, pair, relation_ids=None, pair_mask=None):
        relation_ids = _default_relation_ids(pair, relation_ids)
        hidden = self.in_proj(pair)
        hidden = hidden * (1.0 + torch.tanh(self.relation_scale(relation_ids)))
        hidden = hidden + self.relation_bias(relation_ids)
        hidden = torch.nn.functional.silu(hidden)
        hidden = self.dropout(hidden)
        update = self.out_proj(hidden)

        if pair_mask is not None:
            update = update * pair_mask.unsqueeze(-1).to(update.dtype)

        return update


class HeteroPairformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        pair_dim,
        num_heads,
        num_relations=8,
        use_rot_emb=True,
        attn_qkv_bias=False,
        transition_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        transition_factor=4,
        triangle_hidden_dim=None,
        single_to_pair_hidden_dim=None,
    ):
        super().__init__()
        triangle_hidden_dim = triangle_hidden_dim or max(8, pair_dim // 4)
        single_to_pair_hidden_dim = single_to_pair_hidden_dim or max(8, embed_dim // 16)

        self.single_attn_norm = nn.LayerNorm(embed_dim)
        self.single_transition_norm = nn.LayerNorm(embed_dim)
        self.pair_transition_norm = nn.LayerNorm(pair_dim)
        self.single_to_pair_norm = nn.LayerNorm(embed_dim)

        self.single_to_pair = RelationConditionedSingleToPairUpdate(
            embed_dim, pair_dim, single_to_pair_hidden_dim, num_relations=num_relations
        )
        self.triangle_outgoing = RelationConditionedTriangle(
            pair_dim, triangle_hidden_dim, mode="outgoing", num_relations=num_relations
        )
        self.triangle_incoming = RelationConditionedTriangle(
            pair_dim, triangle_hidden_dim, mode="incoming", num_relations=num_relations
        )
        self.single_attention = HeteroPairBiasedSelfAttention(
            embed_dim,
            pair_dim,
            num_heads,
            num_relations=num_relations,
            attention_dropout=attention_dropout,
            use_rot_emb=use_rot_emb,
            bias=attn_qkv_bias,
        )
        self.single_transition = Transition(embed_dim, transition_factor, transition_dropout)
        self.pair_transition = RelationConditionedPairTransition(
            pair_dim, transition_factor=transition_factor, dropout=transition_dropout, num_relations=num_relations
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x, pair, relation_ids=None, key_padding_mask=None, attn_mask=None, need_attn_weights=False):
        pair_mask = make_pair_mask(pair, key_padding_mask)

        pair = pair + self.residual_dropout(
            self.single_to_pair(self.single_to_pair_norm(x), pair, relation_ids=relation_ids, pair_mask=pair_mask)
        )
        pair = pair + self.residual_dropout(
            self.triangle_outgoing(pair, relation_ids=relation_ids, pair_mask=pair_mask)
        )
        pair = pair + self.residual_dropout(
            self.triangle_incoming(pair, relation_ids=relation_ids, pair_mask=pair_mask)
        )
        pair = pair + self.residual_dropout(
            self.pair_transition(self.pair_transition_norm(pair), relation_ids=relation_ids, pair_mask=pair_mask)
        )

        if pair_mask is not None:
            pair = pair * pair_mask.unsqueeze(-1).to(pair.dtype)

        attn_out, attn = self.single_attention(
            self.single_attn_norm(x),
            pair,
            relation_ids=relation_ids,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_attn_weights=need_attn_weights,
        )
        x = x + self.residual_dropout(attn_out)
        x = x + self.residual_dropout(self.single_transition(self.single_transition_norm(x)))

        if key_padding_mask is not None:
            x = x * (~key_padding_mask).unsqueeze(-1).to(x.dtype)

        return x, pair, attn


class HeteroPairformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        pair_dim,
        num_blocks,
        num_heads,
        num_relations=8,
        use_rot_emb=True,
        attn_qkv_bias=False,
        transition_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        transition_factor=4,
        use_flash_attn=False,
        triangle_hidden_dim=None,
        single_to_pair_hidden_dim=None,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        self.blocks = nn.ModuleList(
            [
                HeteroPairformerBlock(
                    embed_dim=embed_dim,
                    pair_dim=pair_dim,
                    num_heads=num_heads,
                    num_relations=num_relations,
                    use_rot_emb=use_rot_emb,
                    attn_qkv_bias=attn_qkv_bias,
                    transition_dropout=transition_dropout,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    transition_factor=transition_factor,
                    triangle_hidden_dim=triangle_hidden_dim,
                    single_to_pair_hidden_dim=single_to_pair_hidden_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.pair_final_layer_norm = nn.LayerNorm(pair_dim)

    def forward(self, x, struct_embed, key_padding_mask=None, need_attn_weights=False, attn_mask=None, relation_ids=None):
        attn_weights = [] if need_attn_weights else None

        for block in self.blocks:
            x, struct_embed, attn = block(
                x,
                struct_embed,
                relation_ids=relation_ids,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_attn_weights=need_attn_weights,
            )
            if need_attn_weights:
                attn_weights.append(attn)

        x = self.final_layer_norm(x)
        struct_embed = self.pair_final_layer_norm(struct_embed)
        return x, struct_embed, attn_weights
