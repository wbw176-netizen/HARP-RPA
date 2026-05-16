from transformers import PreTrainedModel, PretrainedConfig
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESMConfig(PretrainedConfig):
    model_type = "esm2"

    def __init__(self, pooling='mean', output_dim=1, fix_lms=True, lora_tune=True, break_point=20, **kwargs):
        self.pooling = pooling
        self.output_dim = output_dim
        self.fix_lms = fix_lms
        self.lora_tune = lora_tune
        self.break_point = break_point
        super().__init__(**kwargs)


class RiNALMoConfig(PretrainedConfig):
    model_type = "rinalmo_flash"

    def __init__(
        self,
        pooling='mean',
        output_dim=1,
        fix_lms=True,
        lora_tune=True,
        ddft_tune=False,
        lora_rank=8,
        lora_alpha=16,
        full_ft_last_k=2,
        train_ffn=False,
        **kwargs,
    ):
        self.pooling = pooling
        self.output_dim = output_dim
        self.fix_lms = fix_lms
        self.lora_tune = lora_tune

        # RiNALMo-specific lightweight DDFT flags
        self.ddft_tune = ddft_tune
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.full_ft_last_k = full_ft_last_k
        self.train_ffn = train_ffn
        super().__init__(**kwargs)


class ESM3Config(PretrainedConfig):
    model_type = "esm3"

    def __init__(self, break_point=20, lora_rank=16, lora_alpha=32, **kwargs):
        self.break_point = break_point
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        super().__init__(**kwargs)


class ESMCConfig(PretrainedConfig):
    model_type = "esmc"

    def __init__(self, break_point=20, lora_rank=16, lora_alpha=32, **kwargs):
        self.break_point = break_point
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        super().__init__(**kwargs)


class LoRAESM(PreTrainedModel):
    config_class = ESMConfig

    def __init__(self, model, config):
        super().__init__(config)
        self.model = model

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        return self.model(tokens, repr_layers, need_head_weights, return_contacts)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16, lora_alpha=32, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scale = (lora_alpha / r) if r > 0 else 1.0

        # frozen base weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # LoRA trainable weights
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(in_features, r))
            self.lora_B = nn.Parameter(torch.empty(r, out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r=16, lora_alpha=32):
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=(linear.bias is not None),
        )
        module.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.data)
        return module

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            delta_w = (self.lora_A @ self.lora_B).t() * self.scale
            out = out + F.linear(x, delta_w, None)
        return out


class LoRARiNALMo(PreTrainedModel):
    config_class = RiNALMoConfig

    def __init__(self, model, config):
        super().__init__(config)
        self.model = model

        self.ddft_tune = getattr(config, "ddft_tune", False)
        self.lora_rank = getattr(config, "lora_rank", 8)
        self.lora_alpha = getattr(config, "lora_alpha", 16)
        self.full_ft_last_k = getattr(config, "full_ft_last_k", 2)
        self.train_ffn = getattr(config, "train_ffn", False)

        if self.ddft_tune:
            if not hasattr(self.model, "transformer") or not hasattr(self.model.transformer, "blocks"):
                raise RuntimeError("RiNALMo 未找到 model.transformer.blocks，当前结构与预期不符。")

            self._t_layers = self.model.transformer.blocks
            self.n_layers = len(self._t_layers)
            self.final_norm = self.model.transformer.final_layer_norm

            if not (0 <= self.full_ft_last_k <= self.n_layers):
                raise ValueError(
                    f"full_ft_last_k={self.full_ft_last_k} 超出范围，应在 [0, {self.n_layers}] 内。"
                )

            self._apply_ddft_strategy()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _inject_lora_into_attention(self, layer):
        attn = layer.mh_attn

        # FlashAttention path:
        # TransformerBlock.mh_attn = FlashMultiHeadSelfAttention
        # modules: Wqkv / out_proj
        if hasattr(attn, "Wqkv") and isinstance(attn.Wqkv, nn.Linear):
            attn.Wqkv = LoRALinear.from_linear(
                attn.Wqkv,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )

            if not hasattr(attn, "out_proj") or not isinstance(attn.out_proj, nn.Linear):
                raise RuntimeError("RiNALMo Flash attention 未找到 out_proj。")

            attn.out_proj = LoRALinear.from_linear(
                attn.out_proj,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
            return

        # Non-flash path:
        # TransformerBlock.mh_attn = MultiHeadSelfAttention
        # real linear layers live in mh_attn.mh_attn.{to_q,to_k,to_v,out_proj}
        if hasattr(attn, "mh_attn"):
            inner = attn.mh_attn
            for proj_name in ["to_q", "to_k", "to_v", "out_proj"]:
                if not hasattr(inner, proj_name) or not isinstance(getattr(inner, proj_name), nn.Linear):
                    raise RuntimeError(f"RiNALMo non-flash attention 未找到 {proj_name}。")
                setattr(
                    inner,
                    proj_name,
                    LoRALinear.from_linear(
                        getattr(inner, proj_name),
                        r=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                    ),
                )
            return

        raise RuntimeError("RiNALMo attention 结构与预期不符，无法注入 LoRA。")

    def _unfreeze_top_attention_only(self, layer):
        # attention path
        for p in layer.mh_attn.parameters():
            p.requires_grad = True

        # two layer norms in block
        for p in layer.attn_layer_norm.parameters():
            p.requires_grad = True
        for p in layer.out_layer_norm.parameters():
            p.requires_grad = True

    def _unfreeze_top_full_block(self, layer):
        for p in layer.parameters():
            p.requires_grad = True

    def _apply_ddft_strategy(self):
        # freeze everything first
        for p in self.model.parameters():
            p.requires_grad = False

        deep_start = self.n_layers - self.full_ft_last_k

        # shallow blocks: attention LoRA
        # top-k blocks: full block FT or attention-only FT
        for layer_idx, layer in enumerate(self._t_layers):
            if layer_idx < deep_start:
                self._inject_lora_into_attention(layer)
            else:
                if self.train_ffn:
                    self._unfreeze_top_full_block(layer)
                else:
                    self._unfreeze_top_attention_only(layer)

        # final transformer norm is trainable
        for p in self.final_norm.parameters():
            p.requires_grad = True

        # keep input embedding frozen for stability / memory
        if hasattr(self.model, "embedding"):
            for p in self.model.embedding.parameters():
                p.requires_grad = False

        # MLM head is not used in your downstream task
        if hasattr(self.model, "lm_mask_head"):
            for p in self.model.lm_mask_head.parameters():
                p.requires_grad = False

        # token dropout has no trainable params, but keep it here for clarity
        if hasattr(self.model, "token_dropout"):
            for p in self.model.token_dropout.parameters():
                p.requires_grad = False

    def forward(self, tokens, need_attn_weights=False):
        return self.model(tokens, need_attn_weights)


class LoRAESMC(PreTrainedModel):
    config_class = ESMCConfig

    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
        self.break_point = getattr(config, "break_point", 20)
        self.lora_rank = getattr(config, "lora_rank", 16)
        self.lora_alpha = getattr(config, "lora_alpha", 32)

        if not hasattr(self.model, "transformer") or not hasattr(self.model.transformer, "blocks"):
            raise RuntimeError("当前 ESMC 模型未找到 model.transformer.blocks，请确认 ESMC 版本。")

        self._t_layers = self.model.transformer.blocks
        self.n_layers = len(self._t_layers)
        self.final_norm = self.model.transformer.norm

        if not (0 <= self.break_point <= self.n_layers):
            raise ValueError(f"break_point={self.break_point} 超出范围，应在 [0, {self.n_layers}] 内。")

        self._apply_ddft_strategy()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _inject_lora_into_attention(self, layer):
        attn = layer.attn

        # ESMC attention qkv lives in layernorm_qkv[1]
        if (
            not hasattr(attn, "layernorm_qkv")
            or not isinstance(attn.layernorm_qkv, nn.Sequential)
            or len(attn.layernorm_qkv) < 2
            or not isinstance(attn.layernorm_qkv[1], nn.Linear)
        ):
            raise RuntimeError("未找到 attn.layernorm_qkv[1]，当前 ESMC attention 结构与预期不符。")

        attn.layernorm_qkv[1] = LoRALinear.from_linear(
            attn.layernorm_qkv[1],
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )

        if not hasattr(attn, "out_proj") or not isinstance(attn.out_proj, nn.Linear):
            raise RuntimeError("未找到 attn.out_proj，当前 ESMC attention 结构与预期不符。")

        attn.out_proj = LoRALinear.from_linear(
            attn.out_proj,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )

    def _apply_ddft_strategy(self):
        # freeze everything first
        for p in self.model.parameters():
            p.requires_grad = False

        # shallow: attention LoRA ; deep: full FT
        for layer_idx, layer in enumerate(self._t_layers):
            if layer_idx < self.break_point:
                self._inject_lora_into_attention(layer)
            else:
                for p in layer.parameters():
                    p.requires_grad = True

        # train final norm
        for p in self.final_norm.parameters():
            p.requires_grad = True

        # sequence head is not used in downstream task
        if hasattr(self.model, "sequence_head"):
            for p in self.model.sequence_head.parameters():
                p.requires_grad = False

        # freeze input embedding
        if hasattr(self.model, "embed"):
            for p in self.model.embed.parameters():
                p.requires_grad = False

    def encode(self, protein):
        return self.model.encode(protein)

    def forward(self, tokens, fusion_layer=None):
        out = self.model(sequence_tokens=tokens)
        res = {"embeddings": out.embeddings}

        if fusion_layer is not None:
            if not (0 <= fusion_layer < self.n_layers):
                raise ValueError(f"fusion_layer={fusion_layer} 超出范围，应在 [0, {self.n_layers - 1}] 内。")
            res["shallow_embeddings"] = out.hidden_states[fusion_layer]

        return res
