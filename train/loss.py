import logging
from typing import Union, List

import torch.nn
from ._loss import find_loss_function
from ._key import ABBREV
from nequip.data import AtomicDataDict

from torch_runstats import RunningStats, Reduction


class Loss:
    """
    assemble loss function based on key(s) and coefficient(s)

    Args:
        coeffs (dict, str): keys with coefficient and loss function name

    Example input dictionaries

    ```python
    'total_energy'
    ['total_energy', 'forces']
    {'total_energy': 1.0}
    {'total_energy': (1.0)}
    {'total_energy': (1.0, 'MSELoss'), 'forces': (1.0, 'L1Loss', param_dict)}
    {'total_energy': (1.0, user_define_callables), 'force': (1.0, 'L1Loss', param_dict)}
    {'total_energy': (1.0, 'MSELoss'),
     'force': (1.0, 'Weighted_L1Loss', param_dict)}
    ```

    The loss function can be a loss class name that is exactly the same (case sensitive) to the ones defined in torch.nn.
    It can also be a user define class type that
        - takes "reduction=none" as init argument
        - uses prediction tensor and reference tensor for its call functions,
        - outputs a vector with the same shape as pred/ref

    """

    def __init__(
        self,
        coeffs: Union[dict, str, List[str]],
        coeff_schedule: str = "constant",
        ldetach_method: bool = False,
        lauto_loss: bool = False,
    ):

        self.coeff_schedule = coeff_schedule
        self.ldetach_method = ldetach_method
        self.lauto_loss = lauto_loss
        self.coeffs = {}
        self.funcs = {}
        self.keys = []

        mseloss = find_loss_function("MSELoss", {})
        if isinstance(coeffs, str):
            self.coeffs[coeffs] = 1.0
            self.funcs[coeffs] = mseloss
        elif isinstance(coeffs, list):
            for key in coeffs:
                self.coeffs[key] = 1.0
                self.funcs[key] = mseloss
        elif isinstance(coeffs, dict):
            for key, value in coeffs.items():
                logging.debug(f" parsing {key} {value}")
                coeff = 1.0
                func = "MSELoss"
                func_params = {}
                if isinstance(value, (float, int)):
                    coeff = value
                elif isinstance(value, str) or callable(value):
                    func = value
                elif isinstance(value, (list, tuple)):
                    # list of [func], [func, param], [coeff, func], [coeff, func, params]
                    if isinstance(value[0], (float, int)):
                        coeff = value[0]
                        if len(value) > 1:
                            func = value[1]
                        if len(value) > 2:
                            func_params = value[2]
                    else:
                        func = value[0]
                        if len(value) > 1:
                            func_params = value[1]
                else:
                    raise NotImplementedError(
                        f"expected float, list or tuple, but get {type(value)}"
                    )
                logging.debug(f" parsing {coeff} {func}")
                self.coeffs[key] = coeff
                self.funcs[key] = find_loss_function(
                    func,
                    func_params,
                )
        else:
            raise NotImplementedError(
                f"loss_coeffs can only be str, list and dict. got {type(coeffs)}"
            )

        for key, coeff in self.coeffs.items():
            self.coeffs[key] = torch.as_tensor(coeff, dtype=torch.get_default_dtype())
            self.keys += [key]

        if self.lauto_loss:
            print("Use Automatic Weighted Loss")
            self.awl = AutomaticWeightedLoss(len(self.keys))
        elif self.ldetach_method:
            print("Use Detached Loss")
        else:
            print("Use Original Loss")

    def __call__(self, pred: dict, ref: dict):

        loss = 0.0
        contrib = {}
        if self.lauto_loss:
            loss_list = []
        for key in self.coeffs:
            _loss = self.funcs[key](
                pred=pred,
                ref=ref,
                key=key,
                mean=True,
            )
            contrib[key] = _loss
            _loss = self.coeffs[key] * _loss
            if self.ldetach_method:
                _loss = _loss / (_loss.detach() + 1e-8)
            elif self.lauto_loss:
                loss_list.append(_loss)
            loss = loss + _loss
        contrib["awl"] = loss
        if self.lauto_loss:
            loss, _ = self.awl(loss_list)
        return loss, contrib


class LossStat:
    """
    The class that accumulate the loss function values over all batches
    for each loss component.

    Args:

    keys (null): redundant argument

    """

    def __init__(self, loss_instance=None):
        self.loss_stat = {
            "total": RunningStats(
                dim=tuple(), reduction=Reduction.MEAN, ignore_nan=False
            )
        }
        self.ignore_nan = {}
        if loss_instance is not None:
            for key, func in loss_instance.funcs.items():
                self.ignore_nan[key] = (
                    func.ignore_nan if hasattr(func, "ignore_nan") else False
                )

    def __call__(self, loss, loss_contrib):
        """
        Args:

        loss (torch.Tensor): the value of the total loss function for the current batch
        loss (Dict(torch.Tensor)): the dictionary which contain the loss components
        """

        results = {}

        results["loss"] = self.loss_stat["total"].accumulate_batch(loss).item()

        # go through each component
        for k, v in loss_contrib.items():

            # initialize for the 1st batch
            if k not in self.loss_stat:
                self.loss_stat[k] = RunningStats(
                    dim=tuple(),
                    reduction=Reduction.MEAN,
                    ignore_nan=self.ignore_nan.get(k, False),
                )
                device = v.get_device()
                self.loss_stat[k].to(device="cpu" if device == -1 else device)

            results["loss_" + ABBREV.get(k, k)] = (
                self.loss_stat[k].accumulate_batch(v).item()
            )
        return results

    def reset(self):
        """
        Reset all the counters to zero
        """

        for v in self.loss_stat.values():
            v.reset()

    def to(self, device):
        for v in self.loss_stat.values():
            v.to(device=device)

    def current_result(self):
        results = {
            "loss_" + ABBREV.get(k, k): v.current_result().item()
            for k, v in self.loss_stat.items()
            if k != "total"
        }
        results["loss"] = self.loss_stat["total"].current_result().item()
        return results


class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl([loss1, loss2])
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(
                1 + self.params[i] ** 2
            )
        coefficient = 0.5 / (self.params[: len(x)] ** 2)
        return loss_sum, coefficient
