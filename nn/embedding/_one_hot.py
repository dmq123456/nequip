import torch
import torch.nn.functional
import numpy as np

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin


@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_types
        ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        return data


@compile_mode("script")
class OneHotAtomEncoding_spin(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.
    If spin is include, the norm of spin will also be encoded.
    Only invariant scalars are included here. (zi, |si|)

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {
            AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types + 1, (0, 1))])
        }
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out, include_spin=True)
        self.irreps_out[AtomicDataDict.SPIN_NONLEAF_KEY] = Irreps("1eo")

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_types
        ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)

        # data[AtomicDataDict.SPIN_NONLEAF_KEY] = torch.nn.functional.normalize(data[AtomicDataDict.SPIN_KEY], dim=-1)
        data[AtomicDataDict.SPIN_NONLEAF_KEY] = data[AtomicDataDict.SPIN_KEY]
        spin = torch.nn.functional.normalize(data[AtomicDataDict.SPIN_NONLEAF_KEY], dim=-1)

        spin_norm = torch.sum(spin * spin, dim=-1).unsqueeze(-1)
        one_hot = torch.cat([one_hot, spin_norm], dim=-1)
        data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot
        if self.set_features:  # Possible bug here? Not influence though.
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        return data


@compile_mode("script")
class AtomEmbedding(GraphModuleMixin, torch.nn.Module):
    """Copmute an continue embedding based on one-hot floating point encoding of atoms' discrete atom types.
    From [GemNet](https://arxiv.org/abs/2106.08903) AtomEmbedding

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = True,
        dimension: int = 0,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        if dimension == 0:
            self.dimension = num_types
        else:
            self.dimension = dimension
        self.embedding_net = torch.nn.Embedding(self.num_types, self.dimension)
        torch.nn.init.uniform_(self.embedding_net.weight, a=-np.sqrt(3), b=np.sqrt(3))
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.dimension, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        # one_hot = torch.nn.functional.one_hot(
        #     type_numbers, num_classes=self.num_types
        # ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        embedding = self.embedding_net(type_numbers).to(
            device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype
        )

        data[AtomicDataDict.NODE_ATTRS_KEY] = embedding
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = embedding
        return data


@compile_mode("script")
class SpeciesEnergyEmbedding(GraphModuleMixin, torch.nn.Module):
    """Species energy based on Atoms' atom types.

    Args:

    """

    num_types: int
    add_collect: bool

    def __init__(
        self,
        num_types: int,
        irreps_in=None,
        add_collect: bool = True,
    ):
        super().__init__()
        self.num_types = num_types
        self.add_collect = add_collect
        self.embedding_net = torch.nn.Embedding(self.num_types, 1)
        torch.nn.init.uniform_(self.embedding_net.weight, a=-np.sqrt(3), b=np.sqrt(3))
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        node_energy = self.embedding_net(type_numbers).to(
            device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype
        )

        if self.add_collect:
            data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = (
                data[AtomicDataDict.PER_ATOM_ENERGY_KEY] + node_energy
            )
        else:
            data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = (
                data[AtomicDataDict.PER_ATOM_ENERGY_KEY] * 0 + node_energy
            )

        return data
