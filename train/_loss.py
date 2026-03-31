import inspect
import logging

import torch
import torch.nn
from torch_runstats.scatter import scatter, scatter_mean

from nequip.data import AtomicDataDict
from nequip.utils import instantiate_from_cls_name


class SimpleLoss:
    """wrapper to compute weighted loss function

    Args:

    func_name (str): any loss function defined in torch.nn that
        takes "reduction=none" as init argument, uses prediction tensor,
        and reference tensor for its call functions, and outputs a vector
        with the same shape as pred/ref
    params (str): arguments needed to initialize the function above

    Return:

    if mean is True, return a scalar; else return the error matrix of each entry
    """

    def __init__(self, func_name: str, params: dict = {}):
        params = dict(params)
        self.ignore_nan = params.pop("ignore_nan", False)
        self.exclude_indices = self._parse_exclude_indices(
            params.pop("exclude_indices", [])
        )
        func, _ = instantiate_from_cls_name(
            torch.nn,
            class_name=func_name,
            prefix="",
            positional_args=dict(reduction="none"),
            optional_args=params,
            all_args={},
        )
        self.func_name = func_name
        self.func = func

    @staticmethod
    def _parse_exclude_indices(exclude_indices):
        if exclude_indices in (None, []):
            return tuple()
        if isinstance(exclude_indices, (list, tuple)) and len(exclude_indices) > 0:
            if all(isinstance(idx, int) for idx in exclude_indices):
                return (tuple(exclude_indices),)
            return tuple(tuple(int(i) for i in idx) for idx in exclude_indices)
        raise TypeError("exclude_indices must be a list of indices")

    def _make_component_mask(self, loss: torch.Tensor):
        if len(self.exclude_indices) == 0:
            return None
        if loss.ndim <= 1:
            raise ValueError(
                "exclude_indices requires the target to have at least one component dimension"
            )
        component_shape = tuple(loss.shape[1:])
        mask = torch.ones(component_shape, dtype=torch.bool, device=loss.device)
        for index in self.exclude_indices:
            if len(index) != len(component_shape):
                raise ValueError(
                    f"exclude index {index} does not match component shape {component_shape}"
                )
            mask[index] = False
        if not torch.any(mask):
            raise ValueError("exclude_indices removes all components")
        return mask.reshape(-1)

    def _select_components(self, tensor: torch.Tensor):
        if len(self.exclude_indices) == 0:
            return tensor
        flat = tensor.reshape(tensor.shape[0], -1)
        return flat[:, self._make_component_mask(tensor)]

    def _reduce(self, loss: torch.Tensor, valid_mask: torch.Tensor, mean: bool):
        selected_loss = self._select_components(loss)
        selected_valid = self._select_components(valid_mask.to(loss.dtype))
        if mean:
            denom = selected_valid.sum()
            if torch.eq(denom, 0).item():
                return selected_loss.sum() * 0.0
            return (selected_loss * selected_valid).sum() / denom
        if len(self.exclude_indices) == 0:
            return loss * valid_mask.to(loss.dtype)
        return selected_loss * selected_valid

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        valid_mask = torch.ones_like(ref[key], dtype=torch.bool)
        has_nan = self.ignore_nan and torch.isnan(ref[key]).any()
        if has_nan:
            valid_mask = ref[key] == ref[key]
            target = torch.nan_to_num(ref[key], nan=0.0)
        else:
            target = ref[key]
        loss = self.func(pred[key], target)
        return self._reduce(loss=loss, valid_mask=valid_mask, mean=mean)


class PerAtomLoss(SimpleLoss):
    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        has_nan = self.ignore_nan and torch.isnan(ref[key]).any()
        N = torch.bincount(ref[AtomicDataDict.BATCH_KEY])
        target = ref[key]
        valid_mask = torch.ones_like(target, dtype=torch.bool)
        len_size = len(target.shape)
        expanded_N = N
        for _ in range(len_size - 1):
            expanded_N = expanded_N.unsqueeze(-1)
        if has_nan:
            valid_mask = ref[key] == ref[key]
            loss = (
                self.func(pred[key], torch.nan_to_num(ref[key], nan=0.0))
                / expanded_N
            )
            if self.func_name == "MSELoss":
                loss = loss / expanded_N
            assert loss.shape == pred[key].shape  # [atom, dim]
        else:
            loss = self.func(pred[key], ref[key])
            loss = loss / expanded_N
            if self.func_name == "MSELoss":
                loss = loss / expanded_N
            assert loss.shape == pred[key].shape  # [atom, dim]
        return self._reduce(loss=loss, valid_mask=valid_mask, mean=mean)


class PerSpeciesLoss(SimpleLoss):
    """Compute loss for each species and average among the same species
    before summing them up.

    Args same as SimpleLoss
    """

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        if not mean:
            raise NotImplementedError("Cannot handle this yet")

        has_nan = self.ignore_nan and torch.isnan(ref[key]).any()

        if has_nan:
            valid_mask = ref[key] == ref[key]
            per_atom_loss = self.func(pred[key], torch.nan_to_num(ref[key], nan=0.0))
        else:
            valid_mask = torch.ones_like(ref[key], dtype=torch.bool)
            per_atom_loss = self.func(pred[key], ref[key])

        flat_loss = per_atom_loss.reshape(per_atom_loss.shape[0], -1)
        flat_valid = valid_mask.reshape(valid_mask.shape[0], -1).to(per_atom_loss.dtype)
        if len(self.exclude_indices) > 0:
            component_mask = self._make_component_mask(per_atom_loss)
            flat_loss = flat_loss[:, component_mask]
            flat_valid = flat_valid[:, component_mask]

        denom = flat_valid.sum(dim=1)
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        per_atom_loss = (flat_loss * flat_valid).sum(dim=1) / safe_denom

        spe_idx = pred[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        if has_nan:
            per_species_loss = scatter(per_atom_loss, spe_idx, dim=0)
            assert per_species_loss.ndim == 1  # [type]

            N = scatter((denom > 0).to(per_atom_loss.dtype), spe_idx, dim=0)
            N = N.reciprocal()
            N_species = ((N == N).int()).sum()
            assert N.ndim == 1  # [type]

            per_species_loss = (per_species_loss * N).sum() / N_species

            return per_species_loss

        else:
            assert per_atom_loss.ndim == 1

            # offset species index by 1 to use 0 for nan
            _, inverse_species_index = torch.unique(spe_idx, return_inverse=True)

            per_species_loss = scatter_mean(per_atom_loss, inverse_species_index, dim=0)
            assert per_species_loss.ndim == 1  # [type]

            return per_species_loss.mean()


def find_loss_function(name: str, params):
    """
    Search for loss functions in this module

    If the name starts with PerSpecies, return the PerSpeciesLoss instance
    """

    wrapper_list = dict(
        perspecies=PerSpeciesLoss,
        peratom=PerAtomLoss,
    )

    if isinstance(name, str):
        for key in wrapper_list:
            if name.lower().startswith(key):
                logging.debug(f"create loss instance {wrapper_list[key]}")
                return wrapper_list[key](name[len(key) :], params)
        return SimpleLoss(name, params)
    elif inspect.isclass(name):
        return SimpleLoss(name, params)
    elif callable(name):
        return name
    else:
        raise NotImplementedError(f"{name} Loss is not implemented")
