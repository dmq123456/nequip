from typing import Union, Callable, Dict
import torch
from torch_runstats.scatter import scatter
import os
import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.optimize import BFGS, QuasiNewton
from ase.constraints import ExpCellFilter
from ase.io.trajectory import Trajectory

from nequip.data import AtomicData, AtomicDataDict
from .nequip_calculator import NequIPCalculator
from nequip.train import Trainer


# def nequip_latent_opt_calculator(model, **kwargs):
#     """Build ASE Calculator directly from deployed model."""
#     return NequIP_Latent_Opt_Calculator.from_deployed_model(model, **kwargs)


class NequIP_Latent_Opt_Calculator(NequIPCalculator):
    """NequIP ASE Calculator for Latent space optimization."""

    implemented_properties = ["energy", "energies", "forces", "stress", "free_energy"]

    def __init__(
        self,
        model: torch.jit.ScriptModule,
        # model_path: str,
        r_max: float,
        device: Union[str, torch.device],
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        spin_units_to_ub: float = 1.0,
        transform: Callable = lambda x: x,
        latent_target: Union[None, torch.Tensor] = None,
        opt_coeff: float = 1.0,
        **kwargs
    ):
        # model, _ = Trainer.load_model_from_training_session(
        #     model_path, model_name="best_model.pth", device="cpu"
        # )
        NequIPCalculator.__init__(
            self,
            model=model,
            r_max=r_max,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            spin_units_to_ub=spin_units_to_ub,
            transform=transform,
            **kwargs
        )
        self.latent_target = latent_target  # this is the target latent vector
        self.opt_coeff = (
            opt_coeff  # this is the coefficient for the latent vector optimization
        )

    def set_latent_target(self, latent_target):
        self.latent_target = latent_target

    def set_opt_coeff(self, opt_coeff):
        self.opt_coeff = opt_coeff

    def compare_latent_atom(self, atoms):
        latent_atoms = self.generate_latent_vec(atoms)
        latent_diff = np.sqrt(np.sum((latent_atoms - self.latent_target) ** 2))
        latent_norm = np.sqrt(np.sum((self.latent_target) ** 2))
        return latent_diff, latent_norm

    def generate_latent_vec(self, atoms):
        # prepare data
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]
        data = self.transform(data)
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)
        out = self.model(data)
        latent_vec = (
            out[AtomicDataDict.EDGE_FEATURES_KEY]
            .mean(0)
            .detach()
            .squeeze(-1)
            .cpu()
            .numpy()
        )
        return latent_vec

    def backward_latent(self, atoms):
        # Use the normalization of the difference of latent_target and latent_predict to scale the energy and force

        # prepare data
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]
        data = self.transform(data)
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)
        data = AtomicDataDict.with_batch(data)
        batch = data[AtomicDataDict.BATCH_KEY]
        num_batch: int = int(batch.max().cpu().item()) + 1

        pos = data[AtomicDataDict.POSITIONS_KEY]
        has_cell: bool = AtomicDataDict.CELL_KEY in data
        if has_cell:
            orig_cell = data[AtomicDataDict.CELL_KEY]
            # Make the cell per-batch
            cell = orig_cell.view(-1, 3, 3).expand(num_batch, 3, 3)
            data[AtomicDataDict.CELL_KEY] = cell
        else:
            # torchscript
            orig_cell = self._empty
            cell = self._empty
        pos.requires_grad_(True)
        data[AtomicDataDict.POSITIONS_KEY] = pos
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        data[AtomicDataDict.EDGE_VECTORS_KEY].requires_grad_(True)

        self.model.train()
        out = self.model.model.func(data)
        latent_predict = out[AtomicDataDict.EDGE_FEATURES_KEY].mean(0)
        latent_target = torch.tensor(self.latent_target).type_as(latent_predict)
        energy = (
            torch.sqrt(torch.sum((latent_predict - latent_target) ** 2))
            * self.opt_coeff
        )
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = energy

        grads = torch.autograd.grad(
            [energy],
            [pos, data[AtomicDataDict.EDGE_VECTORS_KEY]],
            create_graph=False,  # needed to allow gradients of this output during training
        )
        forces = grads[0]
        if forces is None:
            # condition needed to unwrap optional for torchscript
            assert False, "failed to compute forces autograd"
        forces = torch.neg(forces)
        data[AtomicDataDict.FORCE_KEY] = forces

        # Store virial
        vector_force = grads[1]
        if vector_force is None:
            # condition needed to unwrap optional for torchscript
            assert False, "failed to compute vector_force autograd"
        edge_virial = torch.einsum(
            "zi,zj->zij", vector_force, data[AtomicDataDict.EDGE_VECTORS_KEY]
        )
        edge_virial = (edge_virial + edge_virial.transpose(-1, -2)) / 2  # symmetric
        atom_virial = scatter(
            edge_virial,
            data[AtomicDataDict.EDGE_INDEX_KEY][0],
            dim=0,
            reduce="sum",
            dim_size=len(pos),
        )
        virial = scatter(atom_virial, batch, dim=0, reduce="sum")

        if virial is None:
            assert False, "failed to compute virial autograd"

        if has_cell:
            volume = torch.einsum(
                "zi,zi->z",
                cell[:, 0, :],
                torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
            ).unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
        else:
            stress = self._empty  # torchscript
        data[AtomicDataDict.STRESS_KEY] = stress
        virial = torch.neg(virial)
        atom_virial = torch.neg(atom_virial)
        data[AtomicDataDict.VIRIAL_KEY] = virial
        data[AtomicDataDict.ATOM_VIRIAL_KEY] = atom_virial

        self.model.eval()

        return data

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Calculate properties based on latent space vector.

        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        out = self.backward_latent(atoms)

        # only store results the model actually computed to avoid KeyErrors
        if AtomicDataDict.TOTAL_ENERGY_KEY in out:
            self.results["energy"] = self.energy_units_to_eV * (
                out[AtomicDataDict.TOTAL_ENERGY_KEY]
                .detach()
                .cpu()
                .numpy()
                .reshape(tuple())
            )
            # "force consistant" energy
            self.results["free_energy"] = self.results["energy"]
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
            self.results["energies"] = self.energy_units_to_eV * (
                out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                .detach()
                .squeeze(-1)
                .cpu()
                .numpy()
            )
        if AtomicDataDict.FORCE_KEY in out:
            # force has units eng / len:
            self.results["forces"] = (
                self.energy_units_to_eV / self.length_units_to_A
            ) * out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        if AtomicDataDict.STRESS_KEY in out:
            stress = out[AtomicDataDict.STRESS_KEY].detach().cpu().numpy()
            stress = stress.reshape(3, 3) * (
                self.energy_units_to_eV / self.length_units_to_A**3
            )
            # ase wants voigt format
            stress_voigt = full_3x3_to_voigt_6_stress(stress)
            self.results["stress"] = stress_voigt

    def optimize_latent(
        self,
        atoms,
        opt_coeff_init=1,
        factor=2,
        step=100,
        relax_cell=False,
        l_line=False,
        working_dir="./",
    ):
        coeff = opt_coeff_init

        if relax_cell:
            molecule = ExpCellFilter(atoms)
            traj = Trajectory(os.path.join(working_dir, "optimize.traj"), "w", molecule)
        else:
            molecule = atoms

        molecule.set_calculator(calc=self)
        name = "optimization"
        optimize_file = os.path.join(working_dir, name)

        if l_line:
            optimizer = QuasiNewton(
                molecule,
                trajectory="%s.traj" % optimize_file,
                restart="%s.pkl" % optimize_file,
            )
        else:
            optimizer = BFGS(
                molecule,
                trajectory="%s.traj" % optimize_file,
                restart="%s.pkl" % optimize_file,
            )

        if relax_cell:
            optimizer.attach(traj)

        for i in range(10):
            self.set_opt_coeff(coeff)
            optimizer.run(0.0, step)
            coeff = coeff * factor

    @classmethod
    def from_training_session(
        self,
        train_dir: str,
        device: str = "cpu",
        model_name="best_model.pth",
        deploy_net_name: str = "jit.pth",
    ):
        """
        Load a trained model from a training session.

        :param train_dir: str, path to training session directory
        :param device: str, device to load model on
        :return: ASE calculator
        """
        calc = self.from_deployed_model(
            model_path=train_dir + "/" + deploy_net_name, set_global_options=True
        )
        calc.model, _ = Trainer.load_model_from_training_session(
            train_dir, model_name="best_model.pth", device="cpu"
        )
        return calc
