from typing import Union
import logging

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin
from ..radial_basis import BesselBasis, FourierBasis
from ..cutoffs import PolynomialCutoff


@compile_mode("script")
class SphericalHarmonicEdgeAttrs(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        irreps_edge_sh: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
    ):
        super().__init__()
        self.out_field = out_field

        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_sh = self.sh(edge_vec)
        data[self.out_field] = edge_sh
        return data


@compile_mode("script")
class SphericalHarmonicEdgeAttrs_spin_soc(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs that are not invariant but equivalent.
    spherical harmonic projections of edge vectors
    Include all edge properties that are not invariant including rij, (si, rij X si, si X sj, rij * si)

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        l_max: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        parity: bool = True,
        time_reversal: bool = True,
    ):
        # Consider parity and time reversal symmetry
        super().__init__()
        self.out_field = out_field
        self.parity = parity
        self.time_reversal = time_reversal

        # Build irreps for different symmetry first
        lmax = l_max
        irreps_edge_sh_00 = o3.Irreps.spherical_harmonics(lmax=lmax, p=1, t=1)
        irreps_edge_sh_10 = o3.Irreps.spherical_harmonics(lmax=lmax, p=-1, t=1)
        irreps_edge_sh_01 = o3.Irreps.spherical_harmonics(lmax=lmax, p=1, t=-1)
        irreps_edge_sh_11 = o3.Irreps.spherical_harmonics(lmax=lmax, p=-1, t=-1)

        # Whatever include soc, rij will be used to feature the edge
        # Rij SphericalHarmonics
        if self.parity:
            self.sh_rij = o3.SphericalHarmonics(
                irreps_edge_sh_10,
                edge_sh_normalize,
                edge_sh_normalization,
                parity=True,
                time_reversal=False,
            )
        else:
            self.sh_rij = o3.SphericalHarmonics(
                irreps_edge_sh_00,
                edge_sh_normalize,
                edge_sh_normalization,
                parity=False,
                time_reversal=False,
            )
        # Only when considering soc, (si X sj, si, and rij X si) will be used to feature the edge
        # si X sj SphericalHarmonics
        self.sh_si_sj = o3.SphericalHarmonics(
            irreps_edge_sh_00,
            edge_sh_normalize,
            edge_sh_normalization,
            parity=False,
            time_reversal=False,
        )

        # si SphericalHarmonics
        if self.time_reversal:
            self.sh_si = o3.SphericalHarmonics(
                irreps_edge_sh_01,
                False,
                edge_sh_normalization,
                parity=False,
                time_reversal=True,
            )
        else:
            self.sh_si = o3.SphericalHarmonics(
                irreps_edge_sh_00,
                False,
                edge_sh_normalization,
                parity=False,
                time_reversal=False,
            )

        # rij X si SphericalHarmonics
        if self.time_reversal:
            if self.parity:
                self.sh_rij_si = o3.SphericalHarmonics(
                    irreps_edge_sh_11,
                    edge_sh_normalize,
                    edge_sh_normalization,
                    parity=True,
                    time_reversal=True,
                )
            else:
                self.sh_rij_si = o3.SphericalHarmonics(
                    irreps_edge_sh_01,
                    edge_sh_normalize,
                    edge_sh_normalization,
                    parity=False,
                    time_reversal=True,
                )
        else:
            if self.parity:
                self.sh_rij_si = o3.SphericalHarmonics(
                    irreps_edge_sh_10,
                    edge_sh_normalize,
                    edge_sh_normalization,
                    parity=True,
                    time_reversal=False,
                )
            else:
                self.sh_rij_si = o3.SphericalHarmonics(
                    irreps_edge_sh_00,
                    edge_sh_normalize,
                    edge_sh_normalization,
                    parity=False,
                    time_reversal=False,
                )

        # rij X si X sj SphericalHarmonics
        if self.parity:
            self.sh_rij_si_sj_vector = o3.SphericalHarmonics(
                irreps_edge_sh_10,
                edge_sh_normalize,
                edge_sh_normalization,
                parity=True,
                time_reversal=False,
            )
        else:
            self.sh_rij_si_sj_vector = o3.SphericalHarmonics(
                irreps_edge_sh_00,
                edge_sh_normalize,
                edge_sh_normalization,
                parity=False,
                time_reversal=False,
            )

        # rij * (si X sj) scalar
        irrep_rij_si_sj_scalar = o3.Irreps([(1, (0, (1 - 2 * self.parity), 1))])

        irrep_rij_si_dot = o3.Irreps([(1, (0, -1, -1))])
        irreps_edge_sh = (
            self.sh_rij.irreps_out
            + self.sh_si_sj.irreps_out
            + self.sh_si.irreps_out
            + self.sh_rij_si.irreps_out
            + irrep_rij_si_dot
            + self.sh_rij_si_sj_vector.irreps_out
            + irrep_rij_si_sj_scalar
        )

        (
            irreps_edge_sh,
            _,
            _,
            self.sort_list,
        ) = irreps_edge_sh.sort_array()  # sort the irreps
        # Overall, 7 scalar are considered.
        # Move two first scalar to the last two scalar for its parity and time reversal
        irreps_edge_sh = irreps_edge_sh

        irreps_edge_sh = list(irreps_edge_sh)  # Change to list first for index
        tmp = irreps_edge_sh[:2]
        tmp.reverse()
        irreps_edge_sh[:5] = irreps_edge_sh[2:7]
        irreps_edge_sh[5:7] = tmp
        self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)  # Turn back to Irreps

        self.scalar_sort = [2, 3, 4, 5, 6, 1, 0]
        # dim_list = [lr.dim for (_, lr) in self.irreps_edge_sh]
        # self.irreps_edge_sh, self.p, _ = self.irreps_edge_sh.sort()
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
            include_spin=True,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        rij_sh = self.sh_rij(edge_vec)

        col = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        row = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        spin = data[AtomicDataDict.SPIN_NONLEAF_KEY]

        si_sj = torch.cross(spin[row], spin[col], dim=-1)
        si_sj_sh = self.sh_si_sj(si_sj)

        si_sh = self.sh_si(spin[row])

        rij_si = torch.cross(edge_vec, spin[row], dim=-1)
        rij_si_sh = self.sh_rij_si(rij_si)

        rij_si_dot = torch.sum(edge_vec * spin[row], dim=-1).unsqueeze(-1)

        rij_si_sj_vector = torch.cross(edge_vec, si_sj, dim=-1)
        rij_si_sj_vector_sh = self.sh_rij_si_sj_vector(rij_si_sj_vector)

        sh_rij_si_sj_scalar = torch.sum(edge_vec * si_sj, dim=-1).unsqueeze(-1)

        edge_sh = torch.cat(
            [
                rij_sh,
                si_sj_sh,
                si_sh,
                rij_si_sh,
                rij_si_dot,
                rij_si_sj_vector_sh,
                sh_rij_si_sj_scalar,
            ],
            dim=-1,
        )
        edge_sh = edge_sh[:, self.sort_list]
        edge_sh[:, :7] = edge_sh[:, self.scalar_sort]

        data[self.out_field] = edge_sh
        return data


@compile_mode("script")
class SphericalHarmonicEdgeAttrs_spin_soc_simple(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs that are not invariant but equivalent.
    spherical harmonic projections of edge vectors
    Include all edge properties that are not invariant including rij, si, sj

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        l_max: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        parity: bool = True,
        time_reversal: bool = True,
    ):
        # Consider parity and time reversal symmetry
        super().__init__()
        self.out_field = out_field
        self.parity = parity
        self.time_reversal = time_reversal

        # Build irreps for different symmetry first
        lmax = l_max
        irreps_edge_sh_00 = o3.Irreps.spherical_harmonics(lmax=lmax, p=1, t=1)
        irreps_edge_sh_10 = o3.Irreps.spherical_harmonics(lmax=lmax, p=-1, t=1)
        irreps_edge_sh_01 = o3.Irreps.spherical_harmonics(lmax=lmax, p=1, t=-1)
        irreps_edge_sh_11 = o3.Irreps.spherical_harmonics(lmax=lmax, p=-1, t=-1)

        # Whatever include soc, rij will be used to feature the edge
        # Rij SphericalHarmonics
        if self.parity:
            self.sh_rij = o3.SphericalHarmonics(
                irreps_edge_sh_10,
                edge_sh_normalize,
                edge_sh_normalization,
                parity=True,
                time_reversal=False,
            )
        else:
            self.sh_rij = o3.SphericalHarmonics(
                irreps_edge_sh_00,
                edge_sh_normalize,
                edge_sh_normalization,
                parity=False,
                time_reversal=False,
            )

        # si SphericalHarmonics
        if self.time_reversal:
            self.sh_si = o3.SphericalHarmonics(
                irreps_edge_sh_01,
                False,
                edge_sh_normalization,
                parity=False,
                time_reversal=True,
            )
        else:
            self.sh_si = o3.SphericalHarmonics(
                irreps_edge_sh_00,
                False,
                edge_sh_normalization,
                parity=False,
                time_reversal=False,
            )

        irreps_edge_sh = (
            self.sh_rij.irreps_out
            + self.sh_si.irreps_out
            + self.sh_si.irreps_out
        )

        (
            irreps_edge_sh,
            _,
            _,
            self.sort_list,
        ) = irreps_edge_sh.sort_array()  # sort the irreps
        # Overall, 3 scalar are considered.
        # Move two first scalar to the last two scalar for its parity and time reversal

        self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)  # Turn back to Irreps

        # dim_list = [lr.dim for (_, lr) in self.irreps_edge_sh]
        # self.irreps_edge_sh, self.p, _ = self.irreps_edge_sh.sort()
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
            include_spin=True,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        rij_sh = self.sh_rij(edge_vec)

        col = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        row = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        spin = data[AtomicDataDict.SPIN_NONLEAF_KEY]

        si_sh = self.sh_si(spin[col])
        
        sj_sh = self.sh_si(spin[row])

        edge_sh = torch.cat(
            [
                rij_sh,
                si_sh,
                sj_sh,
            ],
            dim=-1,
        )
        edge_sh = edge_sh[:, self.sort_list]

        data[self.out_field] = edge_sh
        return data


@compile_mode("script")
class SphericalHarmonicEdgeAttrs_spin_nosoc(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs that are not invariant but equivalent.
    spherical harmonic projections of edge vectors
    Include all edge properties that are not invariant including si, sj not considering SOC

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        l_max (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        l_max: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        time_reversal: bool = True,
    ):
        # Consider parity and time reversal symmetry
        super().__init__()
        self.out_field = out_field
        self.time_reversal = time_reversal

        # Build irreps for different symmetry first
        lmax = l_max
        irreps_edge_sh_00 = o3.Irreps.spherical_harmonics(lmax=lmax, p=1, t=1)
        irreps_edge_sh_01 = o3.Irreps.spherical_harmonics(lmax=lmax, p=1, t=-1)

        # Whatever include soc, rij will be used to feature the edge
        # si SphericalHarmonics
        if self.time_reversal:
            self.sh_si = o3.SphericalHarmonics(
                irreps_edge_sh_01,
                False,
                edge_sh_normalization,
                parity=False,
                time_reversal=True,
            )
        else:
            self.sh_si = o3.SphericalHarmonics(
                irreps_edge_sh_00,
                False,
                edge_sh_normalization,
                parity=False,
                time_reversal=False,
            )

        irreps_edge_sh = self.sh_si.irreps_out + self.sh_si.irreps_out

        (
            irreps_edge_sh,
            _,
            _,
            self.sort_list,
        ) = irreps_edge_sh.sort_array()  # sort the irreps
        # Overall, 3 scalar are considered.
        # Move two first scalar to the last two scalar for its parity and time reversal

        self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)  # Turn back to Irreps

        # dim_list = [lr.dim for (_, lr) in self.irreps_edge_sh]
        # self.irreps_edge_sh, self.p, _ = self.irreps_edge_sh.sort()
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
            include_spin=True,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        col = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        row = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        spin = data[AtomicDataDict.SPIN_NONLEAF_KEY]

        si_sh = self.sh_si(spin[col])
        
        sj_sh = self.sh_si(spin[row])

        edge_sh = torch.cat(
            [
                si_sh,
                sj_sh,
            ],
            dim=-1,
        )
        edge_sh = edge_sh[:, self.sort_list]

        data[self.out_field] = edge_sh
        return data


@compile_mode("script")
class RadialBasisEdgeEncoding(GraphModuleMixin, torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
        out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.cutoff = cutoff(**cutoff_kwargs)
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(self.basis.num_basis, (0, 1))])},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        edge_length_embedded = (
            self.basis(edge_length) * self.cutoff(edge_length)[:, None]
        )
        data[self.out_field] = edge_length_embedded
        return data


@compile_mode("script")
class RadialBasisEdgeEncoding_spin(GraphModuleMixin, torch.nn.Module):
    out_field: str
    """
    Construct edge embedding that are invariant scaler including |rij|, (normalized spin: si * sj)
    """

    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
        out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.cutoff = cutoff(**cutoff_kwargs)
        self.out_field = out_field
        irreps_out = o3.Irreps([(2 * self.basis.num_basis, (0, 1))])
        self.basis_sisj = FourierBasis(
            r_max=1.01,
            r_min=-1.01,  # Avoid boundary problem of bessel basis
            num_basis=self.basis.num_basis,
        )

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_out},
            include_spin=True,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        edge_embedded = self.basis(edge_length)

        spin = data[AtomicDataDict.SPIN_NONLEAF_KEY]
        col = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        row = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        # Normalize spin first
        # spin_norm = torch.norm(spin, dim=-1).unsqueeze(-1)
        # node_lspin = spin_norm > 1e-6
        # spin = torch.nn.functional.normalize(spin, dim=-1)  # * node_lspin
        # Consider not to normalzie？

        sisj_dot = torch.sum(spin[row] * spin[col], dim=-1)
        spin_square = torch.sum(spin * spin, dim=-1).unsqueeze(-1)

        edge_sisj_embedded = (
            self.basis_sisj(sisj_dot) * spin_square[col] * spin_square[row]
        )

        edge_embedded = (
            torch.cat([edge_embedded, edge_sisj_embedded], dim=-1)
            * self.cutoff(edge_length)[:, None]
        )

        data[self.out_field] = edge_embedded
        return data


@compile_mode("script")
class EdgeScalarAddSpin(GraphModuleMixin, torch.nn.Module):
    out_field: str
    """
    Concatenate edge invariant scaler with spin: si * sj
    """

    def __init__(
        self,
        cutoff=PolynomialCutoff,
        cutoff_kwargs={},
        field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        num_spin_basis: int = 0,
        irreps_in=None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            include_spin=True,
        )

        in_dim = self.irreps_in[self.field][0].mul

        if num_spin_basis == 0:
            if in_dim < 10:
                num_spin_basis = 10
            else:
                num_spin_basis = in_dim

        self.cutoff = cutoff(**cutoff_kwargs)
        self.basis_sisj = FourierBasis(
            r_max=1.01,
            r_min=-1.01,  # Avoid boundary problem of bessel basis
            num_basis=num_spin_basis,
        )

        assert len(self.irreps_in[self.field]) == 1
        assert self.irreps_in[self.field][0].ir == (0, 1, 1)  # scalars

        self.irreps_out[self.out_field] = o3.Irreps([(in_dim + num_spin_basis, (0, 1))])

        self.irreps_out[AtomicDataDict.EDGE_SISJ_KEY] = o3.Irreps([(1, (0, 1))])

        logging.info(
            f"Edge_Scalar_add_spin: Initialize spin edge based on updated invariant edge with {num_spin_basis} basis"
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_embedded = data[self.field]

        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]

        data = AtomicDataDict.with_edge_sisj(data)
        spin = data[AtomicDataDict.SPIN_NONLEAF_KEY]
        col = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        row = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        sisj_dot = data[AtomicDataDict.EDGE_SISJ_KEY]
        spin_square = torch.sum(spin * spin, dim=-1).unsqueeze(-1)

        edge_sisj_embedded = (
            self.basis_sisj(sisj_dot.view(-1))
            * spin_square[col]
            * spin_square[row]
            * self.cutoff(edge_length)[:, None]
        )

        edge_embedded = torch.cat([edge_embedded, edge_sisj_embedded], dim=-1)

        data[self.out_field] = edge_embedded
        return data
