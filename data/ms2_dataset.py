"""MS2 reference-geometry adapter; the HARP-RPA model is unchanged."""
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from data.register import DataRegister
from data.rna.base_constants import RNA_ATOMS, RNA_NUCLEOTIDES
from data.structure_dataset import _process_structure, _pairwise_min_atom_dist
from data.transforms import get_transform

REFERENCE='ACAUGAGGAUCACCCAUGU'
MISSING=(1,2,3,17,18,19)

def insert_zero(tensor,index):
    return torch.cat((tensor[:index],tensor.new_zeros((1,)+tensor.shape[1:]),tensor[index:]))

@DataRegister().register('ms2_template_dataset')
class MS2Dataset(Dataset):
    def __init__(self,dataframe,data_root,col_na='RNA sequences',
                 col_label='△G(kcal/mol)',transform=None,**kwargs):
        self.df=dataframe.reset_index(drop=True).copy()
        self.col_label=col_label
        self.sequences=[];self.labels=[]
        for _,row in self.df.iterrows():
            chain,seq=str(row[col_na]).split(':',1)
            if chain!='R' or len(seq)!=19 or not set(seq)<=set('ACGU'):
                raise ValueError('Expected full R:19nt RNA sequence')
            if any(seq[p-1]!=REFERENCE[p-1] for p in MISSING):
                raise ValueError('Candidate changes an unresolved reference position')
            self.sequences.append(seq);self.labels.append(float(row[col_label]))
        self.transform=get_transform(transform)
        base=_process_structure(Path(data_root)/'MS2_1ZDH_ABR_candidate.pdb',
            'MS2_1ZDH_ABR_candidate',valid_prot_chains=['A','B'],valid_rna_chains=['R'])
        if base is None:raise ValueError('Template parsing failed')
        observed=''.join(REFERENCE[p-1] for p in range(1,20) if p not in MISSING)
        if base['rna_seqs']!=[observed]:
            raise ValueError(f'Unexpected parsed RNA: {base["rna_seqs"]}')
        if list(map(len,base['prot_seqs']))!=[129,129]:
            raise ValueError('Expected 129 residues in each MS2 protein chain')
        for _,row in self.df.iterrows():
            expected=[part.split(':',1)[1] for part in str(row['Protein sequences']).split(',')]
            if base['prot_seqs']!=expected:raise ValueError('CSV/template protein sequence mismatch')
        self.template=self.complete(base)
        self.rna_indices=torch.nonzero(self.template['identifier']==1,as_tuple=False).flatten()
        if len(self.rna_indices)!=19:raise ValueError('Incorrect completed RNA indexing')

    @staticmethod
    def complete(base):
        data={k:v.clone() if torch.is_tensor(v) else v for k,v in base.items()}
        observed=torch.nonzero(data['identifier']==1,as_tuple=False).flatten()
        if len(observed)!=13:raise ValueError('Expected 13 observed RNA positions')
        rna_chain=int(data['chain_nb'][observed[0]])
        for p in MISSING:
            start=int(torch.nonzero(data['identifier']==1,as_tuple=False)[0])
            idx=start+p-1
            for key in ('res_nb','chain_nb','identifier','restype','seq_mask',
                        'pos_heavyatom','mask_heavyatom','atom64_positions','atom64_mask'):
                data[key]=insert_zero(data[key],idx)
            data['res_nb'][idx]=p;data['chain_nb'][idx]=rna_chain
            data['identifier'][idx]=1;data['seq_mask'][idx]=True
        rna_idx=torch.nonzero(data['identifier']==1,as_tuple=False).flatten()
        # Distances use actual reference atoms before creating a generic N anchor.
        data['atom_min_dist']=_pairwise_min_atom_dist(data['pos_heavyatom'],data['mask_heavyatom'])
        n1,n9=(14+RNA_ATOMS.index(a) for a in ('N1','N9'))
        for p,idx in enumerate(rna_idx,start=1):
            nt=REFERENCE[p-1]
            data['restype'][idx]=RNA_NUCLEOTIDES.index(nt)+21;data['res_nb'][idx]=p
            if p in MISSING:
                if data['mask_heavyatom'][idx].any():raise ValueError('Phantom missing-position atoms')
                continue
            # Select the chemically correct TEMPLATE glycosidic atom.
            anchor=n9 if nt in 'AG' else n1
            if not data['mask_heavyatom'][idx,anchor]:raise ValueError('Reference glycosidic atom absent')
            xyz=data['pos_heavyatom'][idx,anchor].clone()
            # Two aliases support variant purine/pyrimidine selection. They are
            # shared reference proxies, NOT observations of both variant atoms.
            for slot in (n1,n9):
                data['pos_heavyatom'][idx,slot]=xyz;data['mask_heavyatom'][idx,slot]=True
        data['rna_seqs']=[REFERENCE];data['seq']=''.join(data['prot_seqs'])+REFERENCE
        data['max_prot_length']=max(map(len,data['prot_seqs']));data['max_na_length']=19
        return data

    def __len__(self):return len(self.sequences)

    def __getitem__(self,index):
        # Transforms are allowed to mutate tensors: do not contaminate the next variant.
        data={k:v.clone() if torch.is_tensor(v) else v for k,v in self.template.items()}
        seq=self.sequences[index]
        for nt,idx in zip(seq,self.rna_indices):data['restype'][idx]=RNA_NUCLEOTIDES.index(nt)+21
        sample='ms2_'+hashlib.sha256(seq.encode('ascii')).hexdigest()[:16]
        data.update(seq=''.join(data['prot_seqs'])+seq,rna_seqs=[seq],
                    labels=self.labels[index],complex=sample,id=sample)
        return self.transform(data) if self.transform is not None else data
