import torch
import torch.nn as nn

from models.encoders.attn import GAEncoder
from models.encoders.pair import ResiduePairEncoder


class ParallelGraphAttentionFusion(nn.Module):
    def __init__(
        self,
        embed_dim,
        pair_dim,
        max_num_atoms=4,
        max_aa_types=30,
        num_layers=1,
        protein=True,
        rna=True,
        fusion="gated",
        dropout=0.0,
        ga_block_opt=None,
    ):
        super().__init__()
        if not protein and not rna:
            raise ValueError("At least one graph branch must be enabled.")

        self.max_aa_types = max_aa_types
        self.protein = protein
        self.rna = rna
        self.fusion = fusion

        self.pair_encoder = ResiduePairEncoder(
            feat_dim=pair_dim,
            max_num_atoms=max_num_atoms,
            max_aa_types=max_aa_types,
        )
        self.protein_node_embed = nn.Embedding(max_aa_types, embed_dim)
        self.rna_node_embed = nn.Embedding(max_aa_types, embed_dim)

        ga_block_opt = ga_block_opt or {}
        if protein:
            self.protein_gat = GAEncoder(embed_dim, pair_dim, num_layers, ga_block_opt=ga_block_opt)
        if rna:
            self.rna_gat = GAEncoder(embed_dim, pair_dim, num_layers, ga_block_opt=ga_block_opt)

        if fusion == "gated":
            self.gate = nn.Linear(embed_dim * 2, embed_dim)
            self.graph_proj = nn.Linear(embed_dim, embed_dim)
        elif fusion == "concat":
            self.concat_proj = nn.Linear(embed_dim * 2, embed_dim)
        elif fusion != "add":
            raise ValueError(f"Unknown graph fusion type: {fusion}")

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(embed_dim)

    def _residue_mask(self, mask_atoms):
        if mask_atoms.size(-1) > 1:
            return mask_atoms[:, :, 1].bool()
        return mask_atoms[:, :, 0].bool()

    def _modality_masks(self, aa, mask_atoms, identifier):
        residue_mask = self._residue_mask(mask_atoms)
        if identifier is None:
            protein_mask = residue_mask & (aa >= 0) & (aa < 21)
            rna_mask = residue_mask & (aa >= 21) & (aa < 26)
        else:
            protein_mask = residue_mask & (identifier == 0)
            rna_mask = residue_mask & (identifier == 1)
        return residue_mask, protein_mask, rna_mask

    def forward(self, lm_embedding, aa, res_nb, chain_nb, pos_atoms, mask_atoms, identifier=None):
        residue_mask, protein_mask, rna_mask = self._modality_masks(aa, mask_atoms, identifier)
        safe_aa = aa.clamp(min=0, max=self.max_aa_types - 1).long()
        pair_feat = self.pair_encoder(
            aa=safe_aa,
            res_nb=res_nb,
            chain_nb=chain_nb,
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
            identifier=identifier,
        )

        graph_embedding = torch.zeros_like(lm_embedding)
        if self.protein:
            protein_node = self.protein_node_embed(safe_aa).to(lm_embedding.dtype)
            protein_out = self.protein_gat(pos_atoms, protein_node, pair_feat, protein_mask)
            graph_embedding = graph_embedding + protein_out * protein_mask.unsqueeze(-1).to(protein_out.dtype)
        if self.rna:
            rna_node = self.rna_node_embed(safe_aa).to(lm_embedding.dtype)
            rna_out = self.rna_gat(pos_atoms, rna_node, pair_feat, rna_mask)
            graph_embedding = graph_embedding + rna_out * rna_mask.unsqueeze(-1).to(rna_out.dtype)

        valid = residue_mask.unsqueeze(-1).to(lm_embedding.dtype)
        graph_embedding = graph_embedding.to(lm_embedding.dtype)
        if self.fusion == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([lm_embedding, graph_embedding], dim=-1)))
            update = gate * self.graph_proj(graph_embedding)
            fused = lm_embedding + self.dropout(update) * valid
        elif self.fusion == "concat":
            update = self.concat_proj(torch.cat([lm_embedding, graph_embedding], dim=-1))
            fused = lm_embedding + self.dropout(update) * valid
        else:
            fused = lm_embedding + self.dropout(graph_embedding) * valid

        return self.out_norm(fused)
