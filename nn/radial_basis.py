from typing import Optional
import math

import torch

from torch import nn

from e3nn.math import soft_one_hot_linspace
from e3nn.util.jit import compile_mode


@compile_mode("trace")
class bessel_basis(nn.Module):
    r_max: float
    r_min: float
    num_basis: int

    def __init__(
        self,
        r_max: float,
        r_min: Optional[float] = 0.0,
        num_basis: int = 8,
    ):
        super().__init__()
        self.r_max = r_max  # torch.FloatTensor(r_max)
        self.r_min = r_min  # torch.FloatTensor(r_min)
        self.num_basis = num_basis  # torch.IntTensor(num_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bessel_soft_one_hot_linspace(
            x,
            start=self.r_min,
            end=self.r_max,
            number=self.num_basis,
        )

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (torch.randn(5),)} for _ in range(n)]


def bessel_soft_one_hot_linspace(
    x: torch.Tensor, start: float, end: float, number: int
):
    x = x[..., None] - start
    c = end - start
    bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
    out = math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x

    return out


@compile_mode("trace")
class gaussian_basis(nn.Module):
    r_max: float
    r_min: float
    num_basis: int

    def __init__(
        self,
        r_max: float,
        r_min: Optional[float] = 0.0,
        num_basis: int = 8,
    ):
        super().__init__()
        self.r_max = r_max  # torch.FloatTensor(r_max)
        self.r_min = r_min  # torch.FloatTensor(r_min)
        self.num_basis = num_basis  # torch.IntTensor(num_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gaussian_soft_one_hot_linspace(
            x,
            start=self.r_min,
            end=self.r_max,
            number=self.num_basis,
        )

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (torch.randn(5),)} for _ in range(n)]


def gaussian_soft_one_hot_linspace(
    x: torch.Tensor, start: float, end: float, number: int
):
    values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
    step = values[1] - values[0]
    diff = (x[..., None] - values) / step
    out = diff.pow(2).neg().exp().div(1.12)

    return out


class BesselBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)

        return self.prefactor * (numerator / x.unsqueeze(-1))


class FourierBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, r_min, num_basis=8, trainable=False):
        r"""Radial Fourier Basis


        Parameters
        ----------
        r_max : float
            Maximum number

        r_min : float
            Minimal number

        num_basis : int
            Number of Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(FourierBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.r_min = float(r_min)

        self.prefactor = 1 / math.sqrt(0.25 + num_basis / 2)

        fourier_weights = (
            torch.linspace(start=r_min, end=r_max, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.fourier_weights = nn.Parameter(fourier_weights)
        else:
            self.register_buffer("fourier_weights", fourier_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        x = (x[..., None] - self.r_min) / (self.r_max - self.r_min)
        numerator = torch.cos(self.fourier_weights * x)

        return self.prefactor * numerator


# class GaussianBasis(nn.Module):
#     r_max: float

#     def __init__(self, r_max, r_min=0.0, num_basis=8, trainable=True):
#         super().__init__()

#         self.trainable = trainable
#         self.num_basis = num_basis

#         self.r_max = float(r_max)
#         self.r_min = float(r_min)

#         means = torch.linspace(self.r_min, self.r_max, self.num_basis)
#         stds = torch.full(size=means.size, fill_value=means[1] - means[0])
#         if self.trainable:
#             self.means = nn.Parameter(means)
#             self.stds = nn.Parameter(stds)
#         else:
#             self.register_buffer("means", means)
#             self.register_buffer("stds", stds)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = (x[..., None] - self.means) / self.stds
#         x = x.square().mul(-0.5).exp() / self.stds  # sqrt(2 * pi)
