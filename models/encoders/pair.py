# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from utils.geometry import angstrom_to_nm, pairwise_dihedrals
# from models.encoders.layers import AngularEncoding


# class ResiduePairEncoder(nn.Module):

#     def __init__(self, feat_dim, max_num_atoms, max_aa_types=30, max_relpos=32):
#         super().__init__()
#         self.max_num_atoms = max_num_atoms
#         self.max_aa_types = max_aa_types
#         self.max_relpos = max_relpos
#         self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)
#         self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)

#         self.aapair_to_distcoef = nn.Embedding(self.max_aa_types * self.max_aa_types, max_num_atoms * max_num_atoms)
#         # nn.init.zeros_(self.aapair_to_distcoef.weight)
#         self.distance_embed = nn.Sequential(
#             nn.Linear(max_num_atoms * max_num_atoms, feat_dim), nn.ReLU(),
#             nn.Linear(feat_dim, feat_dim), nn.ReLU(),
#         )

#         self.dihedral_embed = AngularEncoding()
#         feat_dihed_dim = self.dihedral_embed.get_out_dim(2)  # Phi and Psi

#         infeat_dim = feat_dim + feat_dim + feat_dim + feat_dihed_dim
#         self.out_mlp = nn.Sequential(
#             nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
#             nn.Linear(feat_dim, feat_dim), nn.ReLU(),
#             nn.Linear(feat_dim, feat_dim),
#         )

#     def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms):
#         """
#         Args:
#             aa: (N, L).
#             res_nb: (N, L).
#             chain_nb: (N, L).
#             pos_atoms:  (N, L, A, 3)
#             mask_atoms: (N, L, A)
#         Returns:
#             (N, L, L, feat_dim)
#         """
#         N, L = aa.size()
#         mask_residue = mask_atoms[:, :, 1]  # (N, L)
#         mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]

#         # Pair identities
#         aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]  # (N, L, L)
#         feat_aapair = self.aa_pair_embed(aa_pair)

#         # Relative positions
#         same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
#         relpos = torch.clamp(
#             res_nb[:, :, None] - res_nb[:, None, :],
#             min=-self.max_relpos, max=self.max_relpos,
#         )  # (N, L, L)

#         feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:, :, :, None]

#         # Distances
#         d = angstrom_to_nm(torch.linalg.norm(
#             pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :],
#             dim=-1, ord=2,
#         )).reshape(N, L, L, -1)  # (N, L, L, A*A)
#         c = F.softplus(self.aapair_to_distcoef(aa_pair))  # (N, L, L, A*A)
#         d_gauss = torch.exp(-1 * c * d ** 2)
#         mask_atom_pair = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(N, L, L, -1)
#         feat_dist = self.distance_embed(d_gauss * mask_atom_pair)

#         # Orientations
#         dihed = pairwise_dihedrals(pos_atoms)  # (N, L, L, 2)
#         feat_dihed = self.dihedral_embed(dihed)
#         # if torch.isnan(feat_dihed).any():
#         #     print("Found nan in dihed!")
#         # All
#         feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
#         feat_all = self.out_mlp(feat_all)  # (N, L, L, F)
#         feat_all = feat_all * mask_pair[:, :, :, None]
#         if torch.isnan(feat_all).any():
#             # print("??:", torch.isnan(aa_pair).any())
#             print("Let's check:", torch.isnan(feat_aapair).any(), torch.isnan(feat_relpos).any(), torch.isnan(feat_dist).any(), torch.isnan(feat_dihed).any())
#             print("Let's check 2:", torch.isnan(aa).any(), torch.isnan(res_nb).any(), torch.isnan(chain_nb).any(), torch.isnan(pos_atoms).any(), torch.isnan(mask_atoms).any())
#         return feat_all

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geometry import angstrom_to_nm, pairwise_dihedrals
from models.encoders.layers import AngularEncoding

SUPER_PROT_IDX = 27
SUPER_RNA_IDX = 28
SUPER_CPLX_IDX = 29


class ResiduePairEncoder(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=30, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types * self.max_aa_types, max_num_atoms * max_num_atoms)
        # nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms * max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2)  # Phi and Psi

        infeat_dim = feat_dim + feat_dim + feat_dim + feat_dihed_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, identifier=None):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, 1]  # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]

        # Pair identities
        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]  # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair)

        # Relative positions
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
        relpos = torch.clamp(
            res_nb[:, :, None] - res_nb[:, None, :],
            min=-self.max_relpos, max=self.max_relpos,
        )  # (N, L, L)

        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:, :, :, None]

        # Distances
        d = angstrom_to_nm(torch.linalg.norm(
            pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :],
            dim=-1, ord=2,
        )).reshape(N, L, L, -1)  # (N, L, L, A*A)
        c = F.softplus(self.aapair_to_distcoef(aa_pair))  # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * c * d ** 2)
        mask_atom_pair = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(N, L, L, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)

        # Orientations
        dihed = pairwise_dihedrals(pos_atoms)  # (N, L, L, 2)
        feat_dihed = self.dihedral_embed(dihed)
        # if torch.isnan(feat_dihed).any():
        #     print("Found nan in dihed!")
        # All
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
        feat_all = self.out_mlp(feat_all)  # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]
        if torch.isnan(feat_all).any():
            # print("??:", torch.isnan(aa_pair).any())
            print("Let's check:", torch.isnan(feat_aapair).any(), torch.isnan(feat_relpos).any(), torch.isnan(feat_dist).any(), torch.isnan(feat_dihed).any())
            print("Let's check 2:", torch.isnan(aa).any(), torch.isnan(res_nb).any(), torch.isnan(chain_nb).any(), torch.isnan(pos_atoms).any(), torch.isnan(mask_atoms).any())
        return feat_all


class HeteroResiduePairEncoder(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=30, max_relpos=32, num_relations=8):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.num_relations = num_relations

        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)
        self.relation_embed = nn.Embedding(num_relations, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types * self.max_aa_types, max_num_atoms * max_num_atoms)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms * max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )
        self.distance_stats_embed = nn.Sequential(
            nn.Linear(3, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2)

        infeat_dim = feat_dim + feat_dim + feat_dim + feat_dim + feat_dim + feat_dihed_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.last_relation_ids = None

    def _build_relation_ids(self, aa, chain_nb, identifier=None):
        same_chain = chain_nb[:, :, None] == chain_nb[:, None, :]

        is_cplx_special = aa == SUPER_CPLX_IDX
        is_prot_special = aa == SUPER_PROT_IDX
        is_rna_special = aa == SUPER_RNA_IDX

        if identifier is None:
            is_protein = ((aa >= 0) & (aa < 21)) | is_prot_special
            is_rna = ((aa >= 21) & (aa < 26)) | is_rna_special
        else:
            is_protein = (identifier == 0) | is_prot_special
            is_rna = (identifier == 1) | is_rna_special

        protein_pair = is_protein[:, :, None] & is_protein[:, None, :]
        rna_pair = is_rna[:, :, None] & is_rna[:, None, :]

        relation_ids = torch.full_like(chain_nb[:, :, None].expand(-1, -1, chain_nb.size(1)), 4)
        relation_ids = torch.where(protein_pair & same_chain, torch.zeros_like(relation_ids), relation_ids)
        relation_ids = torch.where(protein_pair & (~same_chain), torch.ones_like(relation_ids), relation_ids)
        relation_ids = torch.where(rna_pair & same_chain, torch.full_like(relation_ids, 2), relation_ids)
        relation_ids = torch.where(rna_pair & (~same_chain), torch.full_like(relation_ids, 3), relation_ids)

        prot_special_pair = is_prot_special[:, :, None] | is_prot_special[:, None, :]
        rna_special_pair = is_rna_special[:, :, None] | is_rna_special[:, None, :]
        cplx_special_pair = is_cplx_special[:, :, None] | is_cplx_special[:, None, :]

        relation_ids = torch.where(prot_special_pair, torch.full_like(relation_ids, 6), relation_ids)
        relation_ids = torch.where(rna_special_pair, torch.full_like(relation_ids, 7), relation_ids)
        relation_ids = torch.where(cplx_special_pair, torch.full_like(relation_ids, 5), relation_ids)
        return relation_ids

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, identifier=None):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            identifier: (N, L), 0 for protein and 1 for RNA.
        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, 1] if mask_atoms.size(-1) > 1 else mask_atoms[:, :, 0]
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]

        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]
        feat_aapair = self.aa_pair_embed(aa_pair)

        same_chain = chain_nb[:, :, None] == chain_nb[:, None, :]
        relpos = torch.clamp(
            res_nb[:, :, None] - res_nb[:, None, :],
            min=-self.max_relpos, max=self.max_relpos,
        )
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:, :, :, None]

        relation_ids = self._build_relation_ids(aa, chain_nb, identifier=identifier)
        feat_relation = self.relation_embed(relation_ids)

        d = angstrom_to_nm(torch.linalg.norm(
            pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :],
            dim=-1, ord=2,
        )).reshape(N, L, L, -1)
        mask_atom_pair = (
            mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]
        ).reshape(N, L, L, -1).float()

        c = F.softplus(self.aapair_to_distcoef(aa_pair))
        d_gauss = torch.exp(-1 * c * d ** 2)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)

        large_value = torch.full_like(d, 1e4)
        d_masked = torch.where(mask_atom_pair.bool(), d, large_value)
        min_dist = d_masked.min(dim=-1)[0]
        min_dist = torch.where(mask_atom_pair.sum(dim=-1) > 0, min_dist, torch.zeros_like(min_dist))
        mean_dist = (d * mask_atom_pair).sum(dim=-1) / (mask_atom_pair.sum(dim=-1) + 1e-6)
        valid_fraction = mask_atom_pair.mean(dim=-1)
        feat_dist_stats = self.distance_stats_embed(
            torch.stack([min_dist, mean_dist, valid_fraction], dim=-1)
        )

        dihed = pairwise_dihedrals(pos_atoms)
        feat_dihed = self.dihedral_embed(dihed)

        feat_all = torch.cat(
            [feat_aapair, feat_relpos, feat_relation, feat_dist, feat_dist_stats, feat_dihed],
            dim=-1,
        )
        feat_all = self.out_mlp(feat_all)
        feat_all = feat_all * mask_pair[:, :, :, None]
        self.last_relation_ids = relation_ids
        return feat_all
