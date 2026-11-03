"""Reapply atom-availability masks after the existing backbone selector."""
import torch
from ._base import register_transform

def apply_mask(data,atom_names):
    ids=torch.nonzero(data['identifier']==1,as_tuple=False).flatten()
    seq=''.join(data['rna_seqs'])
    if len(seq)!=len(ids) or data['mask_atoms'].shape[1]!=4:
        raise ValueError('Expected full RNA sequence and four backbone atoms')
    idx=torch.tensor([[14+atom_names.index(a) for a in
        ('P',"C4'","C1'",'N9' if nt in 'AG' else 'N1')] for nt in seq],
        device=ids.device,dtype=torch.long)
    available=data['mask_heavyatom'][ids[:,None],idx].bool()
    data['mask_atoms']=data['mask_atoms'].clone();data['mask_atoms'][ids]=available
    data['pos_atoms']=data['pos_atoms'].clone()
    data['pos_atoms'][ids]=torch.where(available[...,None],data['pos_atoms'][ids],0.)
    return data

@register_transform('ms2_mask_template_backbone')
class MaskTemplateBackbone:
    def __call__(self,data):
        from data.rna.base_constants import RNA_ATOMS
        return apply_mask(data,RNA_ATOMS)
