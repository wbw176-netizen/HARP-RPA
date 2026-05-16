import torch.nn as nn
import torch
import esm
import contextlib
from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from models.register import ModelRegister
from peft import (
    LoraConfig,
    get_peft_model,
)
from models.lora_tune import (
    LoRAESM,
    LoRARiNALMo,
    LoRAESMC,
    ESMConfig,
    RiNALMoConfig,
    ESMCConfig,
)
from models.components.valina_transformer import Transformer
from models.model import cat_pad, segment_cat_pad, load_esm as load_general_esm
R = ModelRegister()

def load_esm(esm_type):
    if esm_type == '650M':
        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    elif esm_type == '3B':
        model, _ = esm.pretrained.esm2_t36_3B_UR50D()
    elif esm_type == '15B':
        model, _ = esm.pretrained.esm2_t48_15B_UR50D()
    else:
        raise NotImplementedError
    feat_size = model.embed_dim
    return model, feat_size
    
def load_rinalmo(rinalmo_weights):
    config = model_config('giga')
    model = RiNALMo(config)
    # alphabet = Alphabet(**config['alphabet'])
    model.load_state_dict(torch.load(rinalmo_weights))
    feat_size = config.globals.embed_dim
    return model, feat_size

def segment_pool(input, chains, mask, pooling):
    # input shape [N', L, E], mask_shape [N', L]
    result = input.new_full([len(chains), input.shape[-1]], 0) # (N, E)
    mask_result = mask.new_full([len(chains), 1], 0) #(N, 1)
    input_flattened = input.reshape((-1, input.shape[-1])) #(N'*L, E)
    mask_flattened = mask.reshape((-1, 1)) #(N'*L, 1)
    # print("Shapes:", result.shape, mask_result.shape, input_flattened.shape, mask_flattened.shape)
    # segment_id shape (N', )
    segment_id = torch.tensor(sum([[i] * chain for i, chain in enumerate(chains)], start=[]), device=result.device, dtype=torch.int64)
    segment_id = segment_id.repeat_interleave(input.shape[1]) #(N'*L)
    result.scatter_add_(0, segment_id.unsqueeze(1).expand_as(input_flattened), input_flattened*mask_flattened)
    mask_result.scatter_add_(0, segment_id.unsqueeze(1), mask_flattened)
    mask_result.reshape((-1, ))
    
    if pooling == 'mean':
        result = result / (mask_result + 1e-10)
    
    return result

@R.register('esmc_rinalmo_seq')
@R.register('esm2_rinalmo_seq')
class ESM2RiNALMo(nn.Module):
    def __init__(self, 
                 rinalmo_weights='./weights/rinalmo_giga_pretrained.pt',
                 esm_type='650M',
                 pooling='token',
                 output_dim=1,
                 fix_lms=True,
                 lora_tune=False,
                 lora_rank=16,
                 lora_alpha=32,
                 break_point=30,
                 rinalmo_lora_tune=None,
                 rinalmo_lora_rank=None,
                 rinalmo_lora_alpha=None,
                 rinalmo_full_ft_last_k=2,
                 rinalmo_train_ffn=False,
                 rinalmo_freeze=None,
                 representation_layer=33,
                 vallina=True,
                 **kwargs
                 ):
        super(ESM2RiNALMo, self).__init__()
        self.esm, esm_feat_size, self.esm_version = load_general_esm(esm_type)
        self.rinalmo, rinalmo_feat_size = load_rinalmo(rinalmo_weights)
        self.vallina=vallina
        # if esm_feat_size != rinalmo_feat_size:
        #     self.project_layer = nn.Linear(esm_feat_size, rinalmo_feat_size)
        self.cat_size = esm_feat_size + rinalmo_feat_size
        self.feat_size = rinalmo_feat_size
        self.representation_layer = representation_layer
        self.transformer = Transformer(**kwargs['transformer'])
        self.complex_dim = kwargs['transformer']['embed_dim']
        self.proj_cplx= nn.Linear(self.feat_size, self.complex_dim)
        self.pooling = pooling
        if self.pooling == 'token':
            self.prot_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.rna_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.complex_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            nn.init.normal_(self.prot_embedding)
            nn.init.normal_(self.rna_embedding)
            nn.init.normal_(self.complex_embedding)
        self.fix_lms = fix_lms
        self.lora_tune = lora_tune
        self.break_point = break_point
        self.rinalmo_lora_tune = lora_tune if rinalmo_lora_tune is None else rinalmo_lora_tune
        self.rinalmo_lora_rank = lora_rank if rinalmo_lora_rank is None else rinalmo_lora_rank
        self.rinalmo_lora_alpha = lora_alpha if rinalmo_lora_alpha is None else rinalmo_lora_alpha
        self.rinalmo_full_ft_last_k = rinalmo_full_ft_last_k
        self.rinalmo_train_ffn = rinalmo_train_ffn
        self.rinalmo_freeze = fix_lms if rinalmo_freeze is None else rinalmo_freeze

        # ---------- protein LM ----------
        if lora_tune:
            if self.esm_version == 'esm2':
                esm_lora_config = LoraConfig(
                    r=lora_rank,
                    bias="none",
                    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
                    lora_alpha=lora_alpha
                )
                esm_config = ESMConfig()
                self.esm = LoRAESM(self.esm, esm_config)
                self.esm = get_peft_model(self.esm, esm_lora_config)
            elif self.esm_version == 'esmc':
                esm_config = ESMCConfig(
                    break_point=break_point,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                )
                self.esm = LoRAESMC(self.esm, esm_config)
                print(
                    f"[SEQ-DDFT] ESMC: layers 0-{break_point-1} attention use LoRA, "
                    f"layers {break_point}-{self.esm.n_layers-1} full fine-tune"
                )
            else:
                raise ValueError(f"Unsupported esm_version for LoRA: {self.esm_version}")
        elif fix_lms:
            for p in self.esm.parameters():
                p.requires_grad_(False)

        # ---------- RNA LM ----------
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
                f"[SEQ-DDFT] RiNALMo: shallow attention uses LoRA; last {self.rinalmo_full_ft_last_k} layers "
                + ("full fine-tune" if self.rinalmo_train_ffn else "attention + LN fine-tune")
            )
        elif self.rinalmo_freeze:
            for p in self.rinalmo.parameters():
                p.requires_grad_(False)
        self.pooling = pooling
        self.pred_head = nn.Sequential(
            nn.Linear(self.complex_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim)
        )
        if self.vallina:
            print("Using vallina version!")
            self.cat_pred_head = nn.Sequential(
            nn.Linear(self.cat_size, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim) 
            )

    def _encode_protein(self, prot_input):
        if self.esm_version == 'esmc':
            return self._encode_esmc(prot_input)
        return self.esm(
            prot_input,
            repr_layers=[self.representation_layer],
            return_contacts=False,
        )['representations'][self.representation_layer]

    def _encode_esmc(self, prot_input):
        import esm.data
        from esm.sdk.api import ESMProtein

        esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        l_padded = prot_input.shape[1]
        device = next(self.esm.parameters()).device
        all_embeddings = []

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
            tokens = protein_tensor.sequence.unsqueeze(0).to(device)

            ctx = torch.no_grad if self.fix_lms and not self.lora_tune else contextlib.nullcontext
            with ctx():
                if self.lora_tune:
                    output = self.esm(tokens)
                    emb = output["embeddings"][0, 1:-1, :]
                else:
                    raw_output = self.esm(sequence_tokens=tokens)
                    emb = raw_output.embeddings[0, 1:-1, :]

            aligned_emb = torch.zeros(
                (l_padded, emb.shape[-1]),
                device=emb.device,
                dtype=emb.dtype,
            )
            aligned_emb[valid_mask.to(aligned_emb.device)] = emb
            all_embeddings.append(aligned_emb)

        return torch.stack(all_embeddings, dim=0)

    def forward(self, input, strategy='separate'):
        prot_input = input['prot']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        na_input = input['na']
        na_chains = input['na_chains']
        na_mask = input['na_mask']
        # print("Input Shape:", prot_input.shape, na_input.shape, prot_chains, na_chains)
        with torch.cuda.amp.autocast():
            prot_embedding = self._encode_protein(prot_input)
            na_embedding = self.rinalmo(na_input)['representation']
        
        # Vallina implementation with mean pooling
        # print("Original Embedding:", prot_embedding.shape, na_embedding.shape)
        if self.vallina:
            if strategy == 'separate':
                # input shape [N', L], where N' is flexible in every batch
                prot_embedding = segment_pool(prot_embedding, prot_chains, prot_mask, pooling=self.pooling)
                na_embedding = segment_pool(na_embedding, na_chains, na_mask, pooling=self.pooling)
            else:
                if self.pooling == 'max':
                    prot_embedding = (prot_embedding * prot_mask.unsqueeze(-1)).max(dim=1)[0]
                    na_embedding = (na_embedding * na_mask.unsqueeze(-1)).max(dim=1)[0]
                else:
                    prot_embedding = (prot_embedding * prot_mask.unsqueeze(-1)).sum(dim=1)
                    na_embedding = (na_embedding * na_mask.unsqueeze(-1)).sum(dim=1)
                    if self.pooling == 'mean':
                        # Prot_mask: [N, L]
                        prot_mask_sum = prot_mask.sum(dim=1, keepdim=True)
                        na_mask_sum = na_mask.sum(dim=1, keepdim=True)
                        prot_embedding = prot_embedding / (prot_mask_sum + 1e-10)
                        na_embedding = na_embedding / (na_mask_sum + 1e-10)
            complex_embedding = torch.cat([prot_embedding, na_embedding], dim=1)
            output = self.cat_pred_head(complex_embedding)
            output = output.squeeze(1)
        else:   
            prot_embedding = prot_embedding.float()
            na_embedding = na_embedding.float()
            # print("Original Embedding:", prot_embedding, na_embedding)
            max_len = input['pos_atoms'].shape[1]
            # Adjust the embeddings from LMs for CFormer
            if 'patch_idx' in input:
                patch_idx = input['patch_idx']
            else:
                patch_idx = None
            
            if strategy == 'separate':
                # input shape [N', L], where N' is flexible in every batch
                out_embedding, masks = segment_cat_pad(prot_embedding, prot_chains, prot_mask, na_embedding, na_chains, na_mask, max_len, patch_idx)
                assert out_embedding.shape[0] == input['size']
            else:
                out_embedding, masks = cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx)
                assert out_embedding.shape[0] == input['size']
                
            out_embedding = self.proj_cplx(out_embedding)
            key_padding_mask = ~masks
            
            if self.pooling == 'token':
                mask_special = torch.zeros((len(out_embedding), 1), device=out_embedding.device, dtype=key_padding_mask.dtype)
                cplx_embed = self.complex_embedding.repeat(len(out_embedding), 1, 1)
                prot_embed = self.prot_embedding.repeat(len(out_embedding), 1, 1)
                rna_embed = self.rna_embedding.repeat(len(out_embedding), 1, 1)
                out_embedding = torch.cat([cplx_embed, prot_embed, rna_embed, out_embedding], dim=1)
                key_padding_mask = torch.cat([mask_special, mask_special, mask_special, key_padding_mask], dim=1)
                
            output, _ = self.transformer(out_embedding, key_padding_mask=key_padding_mask, need_attn_weights=False)

            if self.pooling == 'token':
                complex_embedding = output[:, 0, :].squeeze(1)
            else:
                complex_embedding = (output * (~key_padding_mask).unsqueeze(-1)).sum(dim=1)
                if self.pooling == 'mean':
                    # Prot_mask: [N, L]
                    seq_mask_sum = (~key_padding_mask).sum(dim=1, keepdim=True)
                    complex_embedding = complex_embedding / (seq_mask_sum + 1e-10)

            output = self.pred_head(complex_embedding)
            output = output.squeeze(1)
        
        return output

        
        
            
            
