import contextlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm

from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from models.encoders.pair import HeteroResiduePairEncoder, ResiduePairEncoder
from models.encoders.parallel_gat import ParallelGraphAttentionFusion
from models.register import ModelRegister
from models.components.hetero_pairformer import HeteroPairformer
from models.lora_tune import (
    LoRAESM,
    LoRARiNALMo,
    LoRAESMC,
    ESMConfig,
    RiNALMoConfig,
    ESMCConfig,
)
from data.complex import SUPER_PROT_IDX, SUPER_RNA_IDX, SUPER_CPLX_IDX, SUPER_CHAIN_IDX
from peft import LoraConfig, get_peft_model

R = ModelRegister()


def load_esm(esm_type):
    if esm_type == '650M':
        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        feat_size = model.embed_dim
        return model, feat_size, 'esm2'

    elif esm_type == '3B':
        model, _ = esm.pretrained.esm2_t36_3B_UR50D()
        feat_size = model.embed_dim
        return model, feat_size, 'esm2'

    elif esm_type == '15B':
        model, _ = esm.pretrained.esm2_t48_15B_UR50D()
        feat_size = model.embed_dim
        return model, feat_size, 'esm2'

    elif esm_type == '150M':
        model, _ = esm.pretrained.esm2_t30_150M_UR50D()
        feat_size = model.embed_dim
        return model, feat_size, 'esm2'

    elif esm_type == '35M':
        model, _ = esm.pretrained.esm2_t12_35M_UR50D()
        feat_size = model.embed_dim
        return model, feat_size, 'esm2'

    elif esm_type == '8M':
        model, _ = esm.pretrained.esm2_t6_8M_UR50D()
        feat_size = model.embed_dim
        return model, feat_size, 'esm2'

    elif esm_type.startswith('esmc'):
        try:
            from esm.models.esmc import ESMC
            from esm.tokenization import EsmSequenceTokenizer
        except ImportError as exc:
            raise ImportError(
                "ESMC requires the EvolutionaryScale `esm` package. "
                "Install it in the HARP-RPA environment; the legacy `fair-esm` "
                "package only supports ESM-2 models."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight_file = os.environ.get("ESMC_600M_WEIGHTS") or os.environ.get("ESMC_WEIGHTS")

        if weight_file:
            if not os.path.exists(weight_file):
                raise FileNotFoundError(
                    f"ESMC weight file not found: {weight_file}. "
                    "Set ESMC_600M_WEIGHTS to the local esmc_600m_2024_12_v0.pth path."
                )
            tokenizer = EsmSequenceTokenizer()
            model = ESMC(d_model=1152, n_heads=18, n_layers=36, tokenizer=tokenizer)
            try:
                state_dict = torch.load(weight_file, map_location=device, weights_only=True)
            except TypeError:
                state_dict = torch.load(weight_file, map_location=device)
            model.load_state_dict(state_dict.get("model", state_dict))
        else:
            pretrained_names = [esm_type.replace("-", "_")]
            if esm_type in {"esmc-600m", "esmc_600m"}:
                pretrained_names = ["esmc_600m", "esmc_600m_2024_12"]

            last_error = None
            model = None
            for pretrained_name in pretrained_names:
                try:
                    model = ESMC.from_pretrained(pretrained_name)
                    break
                except Exception as exc:
                    last_error = exc
            if model is None:
                raise RuntimeError(
                    "Failed to load ESMC from the default pretrained registry. "
                    "Set ESMC_600M_WEIGHTS to a local ESMC-600M checkpoint path."
                ) from last_error

        model = model.to(device).eval()
        return model, 1152, 'esmc'

    else:
        raise NotImplementedError(f"Unknown esm_type: {esm_type}")


def load_esm_simple(esm_type):
    model, feat_size, _ = load_esm(esm_type)
    return model, feat_size


def load_rinalmo(rinalmo_weights, rinalmo_type):
    if rinalmo_type == '650M':
        size = 'giga'
    elif rinalmo_type == '150M':
        size = 'mega'
    elif rinalmo_type == '35M':
        size = 'micro'
    elif rinalmo_type == '8M':
        size = 'nano'
    else:
        raise NotImplementedError(f"Unknown rinalmo_type: {rinalmo_type}")

    config = model_config(size)
    model = RiNALMo(config)
    model.load_state_dict(torch.load(rinalmo_weights))
    feat_size = config.globals.embed_dim
    return model, feat_size


def _select_valid_patch_embeddings(selected, patch_idx, patch_mask):
    """Ignore padded indices, not index value zero (a valid first residue)."""
    if patch_mask is None:
        raise ValueError('patch_idx requires the residue padding mask from collate')
    if patch_idx.dim() != 1 or patch_mask.shape != patch_idx.shape:
        raise ValueError('patch_idx and patch_mask must be matching one-dimensional tensors')
    valid = patch_mask.to(device=selected.device, dtype=torch.bool)
    # Collate right-pads all structural fields. Reject holes instead of silently
    # compacting embeddings out of alignment with coordinates and distances.
    if ((~valid[:-1]) & valid[1:]).any().item():
        raise ValueError('patch_mask must describe right padding only')
    indices = patch_idx.to(device=selected.device)[valid]
    return torch.index_select(selected, 0, indices)


def cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx, patch_mask=None):
    new_complexes = []
    masks = []

    for i in range(len(prot_embedding)):
        item_prot_embed = prot_embedding[i]
        item_prot_mask = prot_mask[i]
        item_na_embed = na_embedding[i]
        item_na_mask = na_mask[i]

        item_embed = torch.cat([item_prot_embed, item_na_embed], dim=0)
        indices = torch.nonzero(torch.cat([item_prot_mask, item_na_mask])).flatten()
        selected = torch.index_select(item_embed, 0, indices)

        if patch_idx is not None:
            selected = _select_valid_patch_embeddings(
                selected, patch_idx[i], None if patch_mask is None else patch_mask[i])

        p1d = (0, 0, 0, max_len - len(selected))
        selected_pad = F.pad(selected, p1d, 'constant', 0)

        mask = torch.zeros((selected_pad.shape[0]), device=selected.device)
        mask[:len(selected)] = 1
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad)

    result = torch.stack(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    return result, masks


def segment_cat_pad(prot_embedding, prot_chains, prot_mask, na_embedding, na_chains, na_mask, max_len, patch_idx=None, patch_mask=None):
    cum_prot = torch.cat([torch.tensor([0]), torch.cumsum(torch.Tensor(prot_chains), dim=0)]).int()
    cum_na = torch.cat([torch.tensor([0]), torch.cumsum(torch.Tensor(na_chains), dim=0)]).int()

    new_complexes = []
    masks = []

    for i, (s_prot, e_prot, s_na, e_na) in enumerate(zip(cum_prot[:-1], cum_prot[1:], cum_na[:-1], cum_na[1:])):
        item_prot_embed = prot_embedding[s_prot:e_prot].reshape((-1, prot_embedding.shape[-1]))
        item_prot_mask = prot_mask[s_prot:e_prot].reshape(-1)

        item_na_embed = na_embedding[s_na:e_na].reshape((-1, na_embedding.shape[-1]))
        item_na_mask = na_mask[s_na:e_na].reshape(-1)

        item_embed = torch.cat([item_prot_embed, item_na_embed], dim=0)
        indices = torch.nonzero(torch.cat([item_prot_mask, item_na_mask])).flatten()
        selected = torch.index_select(item_embed, 0, indices)

        if patch_idx is not None:
            selected = _select_valid_patch_embeddings(
                selected, patch_idx[i], None if patch_mask is None else patch_mask[i])

        p1d = (0, 0, 0, max_len - len(selected))
        selected_pad = F.pad(selected, p1d, 'constant', 0)

        mask = torch.zeros((selected_pad.shape[0]), device=selected.device)
        mask[:len(selected)] = 1
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad.unsqueeze(0))

    result = torch.cat(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    return result, masks


@R.register('harp_rpa')
@R.register('copra')
class ESM2RiNALMo(nn.Module):
    def __init__(
        self,
        rinalmo_weights='./weights/rinalmo_giga_pretrained.pt',
        esm_type='650M',
        rinalmo_type='650M',
        pooling='mean',
        output_dim=1,
        pair_dim=320,
        fix_lms=True,
        lora_tune=True,
        lora_rank=16,
        lora_alpha=32,
        representation_layer=33,
        use_fusion=None,
        break_point=20,
        fusion_layer=18,
        rinalmo_lora_tune=False,
        rinalmo_lora_rank=8,
        rinalmo_lora_alpha=16,
        rinalmo_full_ft_last_k=2,
        rinalmo_train_ffn=False,
        rinalmo_freeze=True,
        **kwargs
    ):
        super(ESM2RiNALMo, self).__init__()

        self.esm, esm_feat_size, self.esm_version = load_esm(esm_type)
        self.rinalmo, rinalmo_feat_size = load_rinalmo(rinalmo_weights, rinalmo_type)

        self.esm_type = esm_type
        self.representation_layer = representation_layer
        self.break_point = break_point
        self.fusion_layer = fusion_layer
        self.use_fusion = (fusion_layer is not None) if use_fusion is None else bool(use_fusion)
        if self.use_fusion and self.esm_version == 'esmc' and not lora_tune:
            raise ValueError("ESMC fusion requires lora_tune=True because raw ESMC does not return shallow embeddings.")
        self.lora_tune = lora_tune

        self.rinalmo_lora_tune = rinalmo_lora_tune
        self.rinalmo_lora_rank = rinalmo_lora_rank
        self.rinalmo_lora_alpha = rinalmo_lora_alpha
        self.rinalmo_full_ft_last_k = rinalmo_full_ft_last_k
        self.rinalmo_train_ffn = rinalmo_train_ffn
        self.rinalmo_freeze = rinalmo_freeze

        pair_max_num_atoms = kwargs.get('pair_max_num_atoms', 4)
        self.pair_encoder_type = kwargs.get('pair_encoder_type', 'residue_pair')
        if self.pair_encoder_type == 'residue_pair':
            self.pair_encoder = ResiduePairEncoder(pair_dim, max_num_atoms=pair_max_num_atoms)
        elif self.pair_encoder_type == 'hetero_pair':
            self.pair_encoder = HeteroResiduePairEncoder(pair_dim, max_num_atoms=pair_max_num_atoms)
        else:
            raise ValueError(f"Unknown pair_encoder_type: {self.pair_encoder_type}")

        self.coformer_type = kwargs.get('coformer_type', 'hetero_pairformer')
        if self.coformer_type != 'hetero_pairformer':
            raise ValueError(
                "Only coformer_type='hetero_pairformer' is supported in this codebase. "
                "Use the archived baseline project for legacy baselines."
            )
        self.c_former = HeteroPairformer(**kwargs['coformer'])

        self.proj = 0
        if self.use_fusion:
            self.proj = 1
            self.project_feat = nn.Linear(esm_feat_size * 2, rinalmo_feat_size)
        elif esm_feat_size != rinalmo_feat_size:
            self.proj = 1
            self.project_feat = nn.Linear(esm_feat_size, rinalmo_feat_size)

        self.complex_dim = kwargs['coformer']['embed_dim']
        self.feat_size = rinalmo_feat_size
        self.proj_cplx = nn.Linear(self.feat_size, self.complex_dim)
        self.z_proj = nn.Linear(pair_dim, self.complex_dim) if pair_dim != self.complex_dim else nn.Identity()

        parallel_gat_cfg = kwargs.get('parallel_gat', None)
        self.parallel_gat = None
        if parallel_gat_cfg is not None and parallel_gat_cfg.get('enabled', False):
            graph_pair_dim = parallel_gat_cfg.get('pair_dim', pair_dim)
            ga_block_opt = parallel_gat_cfg.get('ga_block', {})
            self.parallel_gat = ParallelGraphAttentionFusion(
                embed_dim=self.complex_dim,
                pair_dim=graph_pair_dim,
                max_num_atoms=pair_max_num_atoms,
                num_layers=parallel_gat_cfg.get('num_layers', 1),
                protein=parallel_gat_cfg.get('protein', True),
                rna=parallel_gat_cfg.get('rna', True),
                fusion=parallel_gat_cfg.get('fusion', 'gated'),
                dropout=parallel_gat_cfg.get('dropout', 0.0),
                ga_block_opt=dict(ga_block_opt),
            )
            print(
                "[Parallel-GAT] enabled "
                f"(protein={parallel_gat_cfg.get('protein', True)}, "
                f"rna={parallel_gat_cfg.get('rna', True)}, "
                f"layers={parallel_gat_cfg.get('num_layers', 1)}, "
                f"fusion={parallel_gat_cfg.get('fusion', 'gated')})"
            )

        # ---------- ESMC ----------
        if self.esm_version == 'esmc':
            if lora_tune:
                esm_config = ESMCConfig(
                    break_point=break_point,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                )
                self.esm = LoRAESMC(self.esm, esm_config)

                print(
                    f"[DDFT] ESMC: layers 0-{break_point-1} attention use LoRA "
                    f"(layernorm_qkv + out_proj), layers {break_point}-{self.esm.n_layers-1} full fine-tune"
                )
                if self.use_fusion:
                    print(f"[DDFT] ESMC: fuse layer {fusion_layer} shallow hidden with final embeddings")
                self._print_esmc_trainable()

            elif fix_lms:
                for p in self.esm.parameters():
                    p.requires_grad_(False)
                print("ESMC frozen because fix_lms=True")

            if self.rinalmo_lora_tune:
                rinalmo_config = RiNALMoConfig(
                    ddft_tune=True,
                    lora_rank=self.rinalmo_lora_rank,
                    lora_alpha=self.rinalmo_lora_alpha,
                    full_ft_last_k=self.rinalmo_full_ft_last_k,
                    train_ffn=self.rinalmo_train_ffn,
                )
                self.rinalmo = LoRARiNALMo(self.rinalmo, rinalmo_config)
                print(
                    f"[DDFT] RiNALMo: shallow attention uses LoRA; last {self.rinalmo_full_ft_last_k} layers "
                    + ("full fine-tune" if self.rinalmo_train_ffn else "attention + LN fine-tune")
                )
                self._print_rinalmo_trainable()
            elif self.rinalmo_freeze:
                for p in self.rinalmo.parameters():
                    p.requires_grad_(False)
                print("[DDFT] RiNALMo frozen")
            else:
                print("[DDFT] RiNALMo trainable without LoRA")

        # ---------- ESM2 ----------
        elif self.esm_version == 'esm2':
            if lora_tune:
                rinalmo_linear_modules = ['Wqkv']
                esm_linear_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']

                rinalmo_lora_config = LoraConfig(
                    r=lora_rank,
                    bias="none",
                    target_modules=rinalmo_linear_modules,
                    lora_alpha=lora_alpha,
                )
                esm_lora_config = LoraConfig(
                    r=lora_rank,
                    bias="none",
                    target_modules=esm_linear_modules,
                    lora_alpha=lora_alpha,
                )

                rinalmo_config = RiNALMoConfig()
                esm_config = ESMConfig()

                self.rinalmo = LoRARiNALMo(self.rinalmo, rinalmo_config)
                self.esm = LoRAESM(self.esm, esm_config)

                self.rinalmo = get_peft_model(self.rinalmo, rinalmo_lora_config)
                self.esm = get_peft_model(self.esm, esm_lora_config)

                self.rinalmo.print_trainable_parameters()
                self.esm.print_trainable_parameters()

            elif fix_lms:
                for p in self.rinalmo.parameters():
                    p.requires_grad_(False)
                for p in self.esm.parameters():
                    p.requires_grad_(False)

        # ---------- RiNALMo ----------
        if self.esm_version == 'esm2' and not lora_tune:
            for p in self.rinalmo.parameters():
                p.requires_grad_(False)
        print("[DEBUG] rinalmo_lora_tune =", self.rinalmo_lora_tune)
        print("[DEBUG] rinalmo_freeze =", self.rinalmo_freeze)
        
        rinalmo_trainable = 0
        for n, p in self.named_parameters():
            if p.requires_grad and n.startswith("rinalmo"):
                rinalmo_trainable += p.numel()
                print("[RiNALMo Trainable]", n, p.numel())
        
        print(f"[DEBUG] RiNALMo trainable params = {rinalmo_trainable/1e6:.2f}M")
        self.pooling = 'token_interface' if pooling == 'mixed' else pooling
        self.token_poolings = {'token', 'token_interface'}
        self.interface_poolings = {'interface', 'token_interface'}
        self.use_interface_pooling = self.pooling in self.interface_poolings
        self.interface_cross_relation_id = kwargs.get('interface_cross_relation_id', 4)
        self.interface_dist_tau = kwargs.get('interface_dist_tau', 8.0)
        print("Pooling Strategy:", self.pooling)

        if self.pooling in self.token_poolings:
            self.prot_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.rna_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.complex_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            nn.init.normal_(self.prot_embedding)
            nn.init.normal_(self.rna_embedding)
            nn.init.normal_(self.complex_embedding)

        if self.pooling == 'token_interface':
            self.interface_readout_proj = nn.Sequential(
                nn.Linear(self.complex_dim * 2, self.complex_dim), nn.ReLU(),
                nn.Linear(self.complex_dim, self.complex_dim)
            )


        self.pred_head = nn.Sequential(
            nn.Linear(self.complex_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim)
        )
        self.ddg_pair_gate = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, 1),
        )
        self.ddg_head = nn.Sequential(
            nn.LayerNorm(self.complex_dim * 3),
            nn.Linear(self.complex_dim * 3, self.complex_dim), nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(self.complex_dim, max(self.complex_dim // 2, output_dim)), nn.ReLU(),
            nn.Linear(max(self.complex_dim // 2, output_dim), output_dim),
        )
        nn.init.zeros_(self.ddg_head[-1].weight)
        nn.init.zeros_(self.ddg_head[-1].bias)

        self.pretrain_contact_head = nn.Sequential(
            nn.Linear(pair_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, 1)
        )

    def _print_esmc_trainable(self):
        total = sum(p.numel() for p in self.esm.parameters() if p.requires_grad)
        lora = sum(
            p.numel()
            for n, p in self.esm.named_parameters()
            if p.requires_grad and ("lora_A" in n or "lora_B" in n)
        )
        full = total - lora
        print(f"[DDFT] ESMC 闁告瑯鍨甸鍕磼閸愩劌妫橀柡?-> 闁诡剚妲掗? {total/1e6:.2f}M | LoRA: {lora/1e6:.2f}M | Full FT: {full/1e6:.2f}M")

    def _print_rinalmo_trainable(self):
        total = sum(p.numel() for p in self.rinalmo.parameters() if p.requires_grad)
        lora = sum(
            p.numel()
            for n, p in self.rinalmo.named_parameters()
            if p.requires_grad and ("lora_A" in n or "lora_B" in n)
        )
        full = total - lora
        print(f"[DDFT] RiNALMo 闁告瑯鍨甸鍕磼閸愩劌妫橀柡?-> 闁诡剚妲掗? {total/1e6:.2f}M | LoRA: {lora/1e6:.2f}M | Full FT: {full/1e6:.2f}M")

    def _pool_global_embedding(self, output, z, key_padding_mask):
        complex_embedding = output + self.z_proj(z).sum(-2) * 0.001

        if self.pooling in self.token_poolings:
            return complex_embedding[:, 0, :]

        complex_embedding = (complex_embedding * (~key_padding_mask).unsqueeze(-1)).sum(dim=1)
        if self.pooling == 'mean':
            seq_mask_sum = (~key_padding_mask).sum(dim=1, keepdim=True)
            complex_embedding = complex_embedding / (seq_mask_sum + 1e-10)
        return complex_embedding

    def _align_pair_distance(self, atom_min_dist, pair_len, core_start, device, dtype):
        if atom_min_dist is None:
            return None
        dist = atom_min_dist.to(device=device, dtype=dtype)
        if dist.dim() != 3:
            return None
        if dist.shape[1] == pair_len and dist.shape[2] == pair_len:
            return dist

        aligned = torch.full(
            (dist.shape[0], pair_len, pair_len),
            fill_value=torch.inf,
            device=device,
            dtype=dtype,
        )
        copy_len = min(dist.shape[1], dist.shape[2], pair_len - core_start)
        if copy_len > 0:
            aligned[:, core_start:core_start + copy_len, core_start:core_start + copy_len] = dist[:, :copy_len, :copy_len]
        return aligned

    def _pool_interface_embedding(self, z, key_padding_mask, relation_ids=None, atom_min_dist=None):
        valid_pair_mask = self._valid_pair_mask(key_padding_mask)
        core_start = self._core_start()
        if core_start > 0:
            valid_pair_mask = valid_pair_mask.clone()
            valid_pair_mask[:, :core_start, :] = False
            valid_pair_mask[:, :, :core_start] = False

        diag = torch.eye(z.shape[1], device=z.device, dtype=torch.bool).unsqueeze(0)
        valid_pair_mask = valid_pair_mask & (~diag)

        if relation_ids is not None:
            candidate_mask = valid_pair_mask & (relation_ids == self.interface_cross_relation_id)
        else:
            candidate_mask = valid_pair_mask

        contact_logits = self.pretrain_contact_head(z).squeeze(-1)
        dist = self._align_pair_distance(
            atom_min_dist,
            pair_len=z.shape[1],
            core_start=core_start,
            device=z.device,
            dtype=contact_logits.dtype,
        )
        if dist is not None:
            finite_dist = torch.isfinite(dist)
            distance_bias = torch.zeros_like(contact_logits)
            distance_bias[finite_dist] = -dist[finite_dist].clamp(min=0.0) / max(float(self.interface_dist_tau), 1e-6)
            contact_logits = contact_logits + distance_bias
            candidate_mask = candidate_mask & finite_dist

        pair_values = self.z_proj(z)
        pooled = []
        for b in range(z.shape[0]):
            mask = candidate_mask[b]
            if not mask.any().item():
                mask = valid_pair_mask[b]
            if not mask.any().item():
                pooled.append(torch.zeros(self.complex_dim, device=z.device, dtype=pair_values.dtype))
                continue
            weights = torch.softmax(contact_logits[b][mask].float(), dim=0).to(pair_values.dtype)
            pooled.append((weights.unsqueeze(-1) * pair_values[b][mask]).sum(dim=0))
        return torch.stack(pooled, dim=0)

    def _pool_complex_embedding(self, output, z, key_padding_mask, relation_ids=None, atom_min_dist=None):
        if self.pooling == 'interface':
            return self._pool_interface_embedding(
                z,
                key_padding_mask,
                relation_ids=relation_ids,
                atom_min_dist=atom_min_dist,
            )

        global_embedding = self._pool_global_embedding(output, z, key_padding_mask)
        if self.pooling != 'token_interface':
            return global_embedding

        interface_embedding = self._pool_interface_embedding(
            z,
            key_padding_mask,
            relation_ids=relation_ids,
            atom_min_dist=atom_min_dist,
        )
        return self.interface_readout_proj(torch.cat([global_embedding, interface_embedding], dim=-1))

    def _align_mutation_mask(self, mut_identifier, pair_len, key_padding_mask):
        mut_mask = torch.zeros(
            (key_padding_mask.shape[0], pair_len),
            device=key_padding_mask.device,
            dtype=torch.bool,
        )
        if mut_identifier is None:
            return mut_mask

        core_start = self._core_start()
        copy_len = min(mut_identifier.shape[1], pair_len - core_start)
        if copy_len > 0:
            mut_mask[:, core_start:core_start + copy_len] = mut_identifier[:, :copy_len].to(
                device=key_padding_mask.device,
                dtype=torch.bool,
            )
        return mut_mask & (~key_padding_mask.bool())

    def _pool_mutation_token_delta(self, output_wild, output_mut, key_padding_mask, mut_identifier):
        mut_mask = self._align_mutation_mask(mut_identifier, output_wild.shape[1], key_padding_mask)
        delta = output_wild - output_mut
        denom = mut_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        return (delta * mut_mask.unsqueeze(-1).to(delta.dtype)).sum(dim=1) / denom.to(delta.dtype)

    def _pool_mutation_interface_delta(
            self,
            z_wild,
            z_mut,
            key_padding_mask,
            relation_ids,
            mut_identifier,
        ):
        pair_delta = z_wild - z_mut
        pair_values = self.z_proj(pair_delta)
        valid_pair_mask = self._valid_pair_mask(key_padding_mask)

        core_start = self._core_start()
        if core_start > 0:
            valid_pair_mask = valid_pair_mask.clone()
            valid_pair_mask[:, :core_start, :] = False
            valid_pair_mask[:, :, :core_start] = False

        diag = torch.eye(z_wild.shape[1], device=z_wild.device, dtype=torch.bool).unsqueeze(0)
        valid_pair_mask = valid_pair_mask & (~diag)

        mut_mask = self._align_mutation_mask(mut_identifier, z_wild.shape[1], key_padding_mask)
        mut_pair_mask = mut_mask[:, :, None] | mut_mask[:, None, :]
        candidate_mask = valid_pair_mask & mut_pair_mask
        if relation_ids is not None:
            candidate_mask = candidate_mask & relation_ids.eq(self.interface_cross_relation_id)

        gate_logits = self.ddg_pair_gate(pair_delta).squeeze(-1)
        pooled = []
        for b in range(pair_delta.shape[0]):
            mask = candidate_mask[b]
            if not mask.any().item():
                mask = valid_pair_mask[b] & mut_pair_mask[b]
            if not mask.any().item():
                pooled.append(torch.zeros(self.complex_dim, device=z_wild.device, dtype=pair_values.dtype))
                continue
            weights = torch.softmax(gate_logits[b][mask].float(), dim=0).to(pair_values.dtype)
            pooled.append((weights.unsqueeze(-1) * pair_values[b][mask]).sum(dim=0))
        return torch.stack(pooled, dim=0)

    def _predict_antisymmetric_ddg(self, delta_features):
        pred_pos = self.ddg_head(delta_features).squeeze(1)
        pred_neg = self.ddg_head(-delta_features).squeeze(1)
        return 0.5 * (pred_pos - pred_neg)

    def _run_hetero_mutation_ddg(
            self,
            input,
            strategy,
            out_embedding,
            z,
            key_padding_mask,
            relation_ids,
        ):
        mut_input = dict(input)
        mut_input['prot'] = input['prot_mut']
        mut_input['restype'] = input['mut_restype']
        out_mut, z_mut, _, relation_ids_mut = self._forward(mut_input, strategy)

        output_wild, z_wild, _ = self._run_c_former(
            out_embedding,
            z,
            key_padding_mask=key_padding_mask,
            need_attn_weights=False,
            relation_ids=relation_ids,
        )
        output_mut, z_mut, _ = self._run_c_former(
            out_mut,
            z_mut,
            key_padding_mask=key_padding_mask,
            need_attn_weights=False,
            relation_ids=relation_ids_mut,
        )

        wild_global = self._pool_global_embedding(output_wild, z_wild, key_padding_mask)
        mut_global = self._pool_global_embedding(output_mut, z_mut, key_padding_mask)
        global_delta = wild_global - mut_global
        token_delta = self._pool_mutation_token_delta(
            output_wild,
            output_mut,
            key_padding_mask,
            input.get('mut_identifier'),
        )
        pair_delta = self._pool_mutation_interface_delta(
            z_wild,
            z_mut,
            key_padding_mask,
            relation_ids if relation_ids is not None else relation_ids_mut,
            input.get('mut_identifier'),
        )

        delta_features = torch.cat([global_delta, token_delta, pair_delta], dim=-1)
        output_forward = self._predict_antisymmetric_ddg(delta_features)
        return output_forward, -output_forward

    def extract_mutation_interface_delta(self, input, strategy='separate'):
        out_embedding, z, key_padding_mask, relation_ids = self._forward(input, strategy)

        mut_input = dict(input)
        mut_input['prot'] = input['prot_mut']
        mut_input['restype'] = input['mut_restype']
        out_mut, z_mut, _, relation_ids_mut = self._forward(mut_input, strategy)

        output_wild, z_wild, _ = self._run_c_former(
            out_embedding,
            z,
            key_padding_mask=key_padding_mask,
            need_attn_weights=False,
            relation_ids=relation_ids,
        )
        output_mut, z_mut, _ = self._run_c_former(
            out_mut,
            z_mut,
            key_padding_mask=key_padding_mask,
            need_attn_weights=False,
            relation_ids=relation_ids_mut,
        )

        pair_delta = z_wild - z_mut
        readout_delta = self.z_proj(pair_delta)
        gate_logits = self.ddg_pair_gate(pair_delta).squeeze(-1)

        valid_pair_mask = self._valid_pair_mask(key_padding_mask)
        core_start = self._core_start()
        if core_start > 0:
            valid_pair_mask = valid_pair_mask.clone()
            valid_pair_mask[:, :core_start, :] = False
            valid_pair_mask[:, :, :core_start] = False

        diag = torch.eye(z_wild.shape[1], device=z_wild.device, dtype=torch.bool).unsqueeze(0)
        valid_pair_mask = valid_pair_mask & (~diag)
        relation_ids_use = relation_ids if relation_ids is not None else relation_ids_mut
        if relation_ids_use is not None:
            cross_pair_mask = valid_pair_mask & relation_ids_use.eq(self.interface_cross_relation_id)
        else:
            cross_pair_mask = valid_pair_mask

        mut_mask = self._align_mutation_mask(input.get('mut_identifier'), z_wild.shape[1], key_padding_mask)
        mut_pair_mask = mut_mask[:, :, None] | mut_mask[:, None, :]
        mutation_interface_mask = cross_pair_mask & mut_pair_mask

        gate_probs = torch.zeros_like(gate_logits, dtype=torch.float32)
        for b in range(gate_logits.shape[0]):
            mask = mutation_interface_mask[b]
            if mask.any().item():
                gate_probs[b][mask] = torch.softmax(gate_logits[b][mask].float(), dim=0)

        wild_global = self._pool_global_embedding(output_wild, z_wild, key_padding_mask)
        mut_global = self._pool_global_embedding(output_mut, z_mut, key_padding_mask)
        global_delta = wild_global - mut_global
        token_delta = self._pool_mutation_token_delta(
            output_wild,
            output_mut,
            key_padding_mask,
            input.get('mut_identifier'),
        )
        pooled_pair_delta = self._pool_mutation_interface_delta(
            z_wild,
            z_mut,
            key_padding_mask,
            relation_ids_use,
            input.get('mut_identifier'),
        )
        delta_features = torch.cat([global_delta, token_delta, pooled_pair_delta], dim=-1)
        ddg_pred = self._predict_antisymmetric_ddg(delta_features)

        return {
            'ddg_pred': ddg_pred,
            'pair_delta_norm': torch.linalg.norm(pair_delta.float(), dim=-1),
            'readout_delta_norm': torch.linalg.norm(readout_delta.float(), dim=-1),
            'gate_logits': gate_logits,
            'gate_probs': gate_probs,
            'valid_pair_mask': valid_pair_mask,
            'cross_pair_mask': cross_pair_mask,
            'mutation_mask': mut_mask,
            'mutation_interface_mask': mutation_interface_mask,
            'relation_ids': relation_ids_use,
            'key_padding_mask': key_padding_mask,
            'core_start': core_start,
        }

    def _run_c_former(self, out_embedding, z, key_padding_mask=None, need_attn_weights=False, attn_mask=None, relation_ids=None):
        return self.c_former(
            out_embedding,
            z,
            key_padding_mask=key_padding_mask,
            need_attn_weights=need_attn_weights,
            attn_mask=attn_mask,
            relation_ids=relation_ids,
        )

    def _core_start(self):
        return 3 if self.pooling in self.token_poolings else 0

    def _valid_pair_mask(self, key_padding_mask):
        valid = ~key_padding_mask.bool()
        return valid[:, :, None] & valid[:, None, :]

    def _forward(self, input, strategy='separate'):
        prot_input = input['prot']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        na_input = input['na']
        na_chains = input['na_chains']
        na_mask = input['na_mask']

        autocast_ctx = torch.cuda.amp.autocast if torch.cuda.is_available() else contextlib.nullcontext

        with autocast_ctx():
            if self.esm_version == 'esmc':
                import esm.data
                from esm.sdk.api import ESMProtein

                esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

                all_embeddings = []
                all_shallow_embeddings = [] if self.use_fusion else None
                l_padded = prot_input.shape[1]

                for i in range(prot_input.shape[0]):
                    valid_mask = (
                        (prot_input[i] != esm_alphabet.padding_idx) &
                        (prot_input[i] != esm_alphabet.cls_idx) &
                        (prot_input[i] != esm_alphabet.eos_idx)
                    )

                    valid_tokens = prot_input[i][valid_mask]
                    seq_str = ''.join([esm_alphabet.all_toks[tok.item()] for tok in valid_tokens])

                    protein = ESMProtein(sequence=seq_str)
                    protein_tensor = self.esm.encode(protein)
                    tokens = protein_tensor.sequence

                    ctx = contextlib.nullcontext if self.lora_tune else torch.no_grad
                    with ctx():
                        tokens = tokens.unsqueeze(0).to(self.esm.device)
                        if self.lora_tune:
                            output = self.esm(
                                tokens,
                                fusion_layer=self.fusion_layer if self.use_fusion else None,
                            )
                        else:
                            raw_output = self.esm(sequence_tokens=tokens)
                            output = {"embeddings": raw_output.embeddings}
                            if self.use_fusion:
                                output["shallow_embeddings"] = raw_output.hidden_states[self.fusion_layer]

                    emb = output["embeddings"][0, 1:-1, :]
                    aligned_emb = torch.zeros(
                        (l_padded, emb.shape[-1]),
                        device=emb.device,
                        dtype=emb.dtype,
                    )
                    aligned_emb[valid_mask] = emb
                    all_embeddings.append(aligned_emb)

                    if self.use_fusion:
                        shallow_emb = output["shallow_embeddings"][0, 1:-1, :]
                        aligned_shallow = torch.zeros(
                            (l_padded, shallow_emb.shape[-1]),
                            device=shallow_emb.device,
                            dtype=shallow_emb.dtype,
                        )
                        aligned_shallow[valid_mask] = shallow_emb
                        all_shallow_embeddings.append(aligned_shallow)

                prot_embedding = torch.stack(all_embeddings, dim=0)

                if self.use_fusion:
                    shallow_embedding = torch.stack(all_shallow_embeddings, dim=0)
                    prot_embedding = torch.cat([shallow_embedding, prot_embedding], dim=-1)

            else:
                prot_embedding = self.esm(
                    prot_input,
                    repr_layers=[self.representation_layer],
                    return_contacts=False
                )['representations'][self.representation_layer]

            na_embedding = self.rinalmo(na_input)['representation']

            if self.proj:
                prot_embedding = self.project_feat(prot_embedding)

        prot_embedding = prot_embedding.float()
        na_embedding = na_embedding.float()
        max_len = input['pos_atoms'].shape[1]
        patch_idx = input.get('patch_idx', None)
        patch_mask = input.get('mask', None)

        if strategy == 'separate':
            out_embedding, masks = segment_cat_pad(
                prot_embedding, prot_chains, prot_mask,
                na_embedding, na_chains, na_mask,
                max_len, patch_idx, patch_mask=patch_mask
            )
            assert out_embedding.shape[0] == input['size']
        else:
            out_embedding, masks = cat_pad(
                prot_embedding, prot_mask,
                na_embedding, na_mask,
                max_len, patch_idx, patch_mask=patch_mask
            )
            assert out_embedding.shape[0] == input['size']

        out_embedding = self.proj_cplx(out_embedding)
        key_padding_mask = ~masks

        aa = input['restype']
        res_nb = input['res_nb']
        chain_nb = input['chain_nb']
        pos_atoms = input['pos_atoms']
        mask_atoms = input['mask_atoms']
        identifier = input.get('identifier', None)

        if self.parallel_gat is not None:
            out_embedding = self.parallel_gat(
                out_embedding,
                aa=aa,
                res_nb=res_nb,
                chain_nb=chain_nb,
                pos_atoms=pos_atoms,
                mask_atoms=mask_atoms,
                identifier=identifier,
            )

        if self.pooling in self.token_poolings:
            mask_special = torch.zeros(
                (len(out_embedding), 1),
                device=out_embedding.device,
                dtype=key_padding_mask.dtype
            )
            cplx_embed = self.complex_embedding.repeat(len(out_embedding), 1, 1)
            prot_embed = self.prot_embedding.repeat(len(out_embedding), 1, 1)
            rna_embed = self.rna_embedding.repeat(len(out_embedding), 1, 1)

            out_embedding = torch.cat([cplx_embed, prot_embed, rna_embed, out_embedding], dim=1)
            key_padding_mask = torch.cat([mask_special, mask_special, mask_special, key_padding_mask], dim=1)

            cplx_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_CPLX_IDX
            prot_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_PROT_IDX
            rna_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_RNA_IDX
            aa = torch.cat([cplx_type, prot_type, rna_type, aa], dim=1)

            res_nb_cplx = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 0
            res_nb_prot = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 1
            res_nb_rna = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 2
            res_nb = torch.cat([res_nb_cplx, res_nb_prot, res_nb_rna, res_nb], dim=1)

            super_chain_id = torch.ones_like(mask_special, device=out_embedding.device, dtype=chain_nb.dtype) * SUPER_CHAIN_IDX
            chain_nb = torch.cat([super_chain_id, super_chain_id, super_chain_id, chain_nb], dim=1)

            if identifier is not None:
                cplx_identifier = torch.ones_like(mask_special, device=out_embedding.device, dtype=identifier.dtype) * 2
                prot_identifier = torch.zeros_like(mask_special, device=out_embedding.device, dtype=identifier.dtype)
                rna_identifier = torch.ones_like(mask_special, device=out_embedding.device, dtype=identifier.dtype)
                identifier = torch.cat([cplx_identifier, prot_identifier, rna_identifier, identifier], dim=1)

            center_cplx = torch.zeros(
                (len(out_embedding), 1, pos_atoms.shape[2], 3),
                device=out_embedding.device,
                dtype=pos_atoms.dtype
            )

            center_prot = (
                (
                    pos_atoms * (1 - input['identifier'])[:, :, None, None] * mask_atoms.unsqueeze(-1)
                ).reshape([len(out_embedding), -1, 3]).sum(dim=1)
                /
                (
                    ((1 - input['identifier'][:, :, None]) * mask_atoms + 1e-10)
                    .reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1)
                )
            )[:, None, None, :].repeat(1, 1, pos_atoms.shape[2], 1)

            center_rna = (
                (
                    pos_atoms * input['identifier'][:, :, None, None] * mask_atoms.unsqueeze(-1)
                ).reshape([len(out_embedding), -1, 3]).sum(dim=1)
                /
                (
                    (input['identifier'][:, :, None] * mask_atoms + 1e-10)
                    .reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1)
                )
            )[:, None, None, :].repeat(1, 1, pos_atoms.shape[2], 1)

            pos_atoms = torch.cat([center_cplx, center_prot, center_rna, pos_atoms], dim=1)

            mask_atom = torch.zeros(
                (len(out_embedding), 1, pos_atoms.shape[2]),
                device=out_embedding.device,
                dtype=mask_atoms.dtype
            )
            mask_atom[:, :, 0] = 1
            mask_atoms = torch.cat([mask_atom, mask_atom, mask_atom, mask_atoms], dim=1)

        z = self.pair_encoder(
            aa=aa,
            res_nb=res_nb,
            chain_nb=chain_nb,
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
            identifier=identifier,
        )
        relation_ids = getattr(self.pair_encoder, 'last_relation_ids', None)

        return out_embedding, z, key_padding_mask, relation_ids

    def forward(
            self,
            input,
            strategy='separate',
            stage='finetune',
        ):
        out_embedding, z, key_padding_mask, relation_ids = self._forward(input, strategy)

        if stage == 'finetune':
            output, z, attn = self._run_c_former(
                out_embedding,
                z,
                key_padding_mask=key_padding_mask,
                need_attn_weights=False,
                relation_ids=relation_ids,
            )
            complex_embedding = self._pool_complex_embedding(
                output,
                z,
                key_padding_mask,
                relation_ids=relation_ids,
                atom_min_dist=input.get('atom_min_dist'),
            )
            output = self.pred_head(complex_embedding).squeeze(1)
            return output

        elif stage == 'mutation':
            return self._run_hetero_mutation_ddg(
                input,
                strategy,
                out_embedding,
                z,
                key_padding_mask,
                relation_ids,
            )

        else:
            raise NotImplementedError
