from typing import Optional
import logging

from e3nn import o3

from nequip.data import AtomicDataDict, AtomicDataset
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
    OneHotAtomEncoding_spin,
    RadialBasisEdgeEncoding_spin,
    SphericalHarmonicEdgeAttrs_spin_soc,
    SphericalHarmonicEdgeAttrs_spin_soc_simple,
)

from . import builder_utils


def SimpleIrrepsConfig(config, prefix: Optional[str] = None):
    """Builder that pre-processes options to allow "simple" configuration of irreps."""

    # We allow some simpler parameters to be provided, but if they are,
    # they have to be correct and not overridden
    simple_irreps_keys = [
        "l_max",
        "parity",
        "num_features",
        "time_reversal",
        "include_spin",
        "soc",
    ]
    real_irreps_keys = [
        "chemical_embedding_irreps_out",
        "feature_irreps_hidden",
        "irreps_edge_sh",
        "conv_to_output_hidden_irreps_out",
    ]

    prefix = "" if prefix is None else f"{prefix}_"

    has_simple: bool = any(
        (f"{prefix}{k}" in config) or (k in config) for k in simple_irreps_keys
    )
    has_full: bool = any(
        (f"{prefix}{k}" in config) or (k in config) for k in real_irreps_keys
    )
    assert has_simple or has_full

    config["num_features"] = config.get("num_features", 32)
    config["parity"] = config.get("parity", True)
    config["include_spin"] = config.get("include_spin", False)
    include_spin = config.get(f"{prefix}include_spin", config["include_spin"])

    config["time_reversal"] = config.get("time_reversal", config["include_spin"])
    config["soc"] = config.get("soc", config["include_spin"])

    update = {}
    if has_simple or include_spin:
        # nothing to do if not
        lmax = config.get(f"{prefix}l_max", config["l_max"])
        parity = config.get(f"{prefix}parity", config["parity"])
        time_reversal = (
            config.get(f"{prefix}time_reversal", config["time_reversal"])
            and include_spin
        )  # Consider time reversal only when including spin
        num_features = config.get(f"{prefix}num_features", config["num_features"])
        update[f"{prefix}chemical_embedding_irreps_out"] = repr(
            o3.Irreps([(num_features, (0, 1))])  # n scalars
        )

        if include_spin:
            update[
                f"{prefix}irreps_edge_sh"
            ] = lmax  # Require irreps_edge_sh to be integer if include spin
        else:
            update[f"{prefix}irreps_edge_sh"] = repr(
                o3.Irreps.spherical_harmonics(lmax=lmax, p=-1 if parity else 1)
            )

        update[f"{prefix}feature_irreps_hidden"] = repr(
            o3.Irreps(
                [
                    (num_features, (l, p, t))
                    for p in ((1, -1) if parity else (1,))
                    for t in ((1, -1) if time_reversal else (1,))
                    for l in range(lmax + 1)
                ]
            )
        )
        update[f"{prefix}conv_to_output_hidden_irreps_out"] = repr(
            # num_features // 2  scalars
            o3.Irreps([(max(1, num_features // 2), (0, 1))])
        )

    # check update is consistant with config
    # (this is necessary since it is not possible
    #  to delete keys from config, so instead of
    #  making simple and full styles mutually
    #  exclusive, we just insist that if full
    #  and simple are provided, full must be
    #  consistant with simple)
    for k, v in update.items():
        if k in config:
            assert (
                config[k] == v
            ), f"For key {k}, the full irreps options had value `{config[k]}` inconsistant with the value derived from the simple irreps options `{v}`"
        config[k] = v


def EnergyModel(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:
    """Base default energy model archetecture.

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.
    """
    logging.debug("Start building the network model")

    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    num_layers = config.get("num_layers", 3)
    include_spin = config.get("include_spin", False)
    soc = config.get("soc", include_spin)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding_spin if include_spin else OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs_spin_soc
        if (include_spin and soc)
        else SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding_spin
        if include_spin
        else RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    # add convnet layers
    # insertion preserves order
    for layer_i in range(num_layers):
        layers[f"layer{layer_i}_convnet"] = ConvNetLayer

    # .update also maintains insertion order
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # -- output block --
            "conv_to_output_hidden": AtomwiseLinear,
            "output_hidden_to_scalar": (
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY),
            ),
        }
    )

    layers["total_energy_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )
