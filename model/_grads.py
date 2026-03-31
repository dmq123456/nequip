from nequip.nn import GraphModuleMixin, GradientOutput
from nequip.nn import PartialForceOutput as PartialForceOutputModule
from nequip.nn import StressOutput as StressOutputModule
from nequip.nn import ParaStressOutput as ParaStressOutputModule
from nequip.nn import ParaStressBECOutput as ParaStressBECOutputModule
from nequip.nn import StressForceSpinOutput as StressForceSpinOutputModule
from nequip.nn import ParaStressForceSpinOutput as ParaStressForceSpinOutputModule
from nequip.data import AtomicDataDict
from nequip.nn import PartialSpinForceOutput as PartialSpinForceOutputModule
from nequip.nn import ForceSpinForceOutput as ForceSpinForceOutputModule


def ForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces to a model that predicts energy.

    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """
    if AtomicDataDict.FORCE_KEY in model.irreps_out:
        raise ValueError("This model already has force outputs.")
    return GradientOutput(
        func=model,
        of=AtomicDataDict.TOTAL_ENERGY_KEY,
        wrt=AtomicDataDict.POSITIONS_KEY,
        out_field=AtomicDataDict.FORCE_KEY,
        sign=-1,  # force is the negative gradient
    )


def PartialForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and partial forces to a model that predicts energy.

    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.PARTIAL_FORCE_KEY in model.irreps_out
    ):
        raise ValueError("This model already has force outputs.")
    return PartialForceOutputModule(func=model)


def StressForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and stresses to a model that predicts energy.

    Args:
        model: the model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``StressOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.STRESS_KEY in model.irreps_out
    ):
        raise ValueError("This model already has force or stress outputs.")
    return StressOutputModule(func=model)


def ParaStressForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and stresses to a model that predicts energy.

    Args:
        model: the model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``StressOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.STRESS_KEY in model.irreps_out
    ):
        raise ValueError("This model already has force or stress outputs.")
    return ParaStressOutputModule(func=model)


def ParaStressForceBECOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and stresses and Born effective charge to a model that predicts energy and dipole.

    Args:
        model: the model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``StressOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.STRESS_KEY in model.irreps_out
        or AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY in model.irreps_out
    ):
        raise ValueError("This model already has force or stress outputs.")
    return ParaStressBECOutputModule(func=model)

def StressForceSpinOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and stresses and spin forces to a model that predicts energy.

    Args:
        model: the model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``StressOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.STRESS_KEY in model.irreps_out
        or AtomicDataDict.SPIN_FORCE_KEY in model.irreps_out
    ):
        raise ValueError(
            "This model already has force or stress or spin forces outputs."
        )
    return StressForceSpinOutputModule(func=model)


def ParaStressForceSpinOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and stresses and spin forces to a model that predicts energy.

    Args:
        model: the model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``StressOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.STRESS_KEY in model.irreps_out
        or AtomicDataDict.SPIN_FORCE_KEY in model.irreps_out
    ):
        raise ValueError(
            "This model already has force or stress or spin forces outputs."
        )
    return ParaStressForceSpinOutputModule(func=model)


def SpinForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add spin forces to a model that predicts energy.

    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """
    if AtomicDataDict.SPIN_FORCE_KEY in model.irreps_out:
        raise ValueError("This model already has spin force outputs.")
    return GradientOutput(
        func=model,
        of=AtomicDataDict.TOTAL_ENERGY_KEY,
        wrt=AtomicDataDict.SPIN_KEY,
        out_field=AtomicDataDict.SPIN_FORCE_KEY,
        sign=-1,  # spin force is the positive gradient
        retain_graph=True,
        include_spin=True,
    )


def ForceSpinForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and spin forces to a model that predicts energy.

    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.SPIN_FORCE_KEY in model.irreps_out
        or AtomicDataDict.FORCE_KEY in model.irreps_out
    ):
        raise ValueError("This model already has spin force outputs.")
    return ForceSpinForceOutputModule(
        func=model,
    )


def PartialSpinForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and partial forces to a model that predicts energy.

    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.SPIN_FORCE_KEY in model.irreps_out
        or AtomicDataDict.SPIN_PARTIAL_FORCE_KEY in model.irreps_out
    ):
        raise ValueError("This model already has force outputs.")
    return PartialSpinForceOutputModule(func=model)
