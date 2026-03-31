import nequip
from nequip.ase import nequip_calculator, NequIPCalculator
from ase.io import read, write
import numpy as np
from nequip.data import AtomicData, AtomicDataDict
from sklearn.metrics import mean_absolute_error
import pandas as pd
from nequip.train import Trainer
import torch

from typing import Optional
import sys
import argparse
import logging
import textwrap
from pathlib import Path
import contextlib
from tqdm.auto import tqdm

import ase.io

import torch

from nequip.data import AtomicData, Collater, dataset_from_config, register_fields
from nequip.scripts.deploy import load_deployed_model, R_MAX_KEY
from nequip.scripts._logger import set_up_script_logger
from nequip.scripts.train import default_config, check_code_version
from nequip.utils._global_options import _set_global_options
from nequip.train import Trainer, Loss, Metrics
from nequip.utils import load_file, instantiate, Config


def main(args=None, running_as_script: bool = True):
    # in results dir, do: nequip-deploy build --train-dir . deployed.pth
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Compute the Heienberg coefficients of a model on a structure.

            The model can specified in individual YAML config files, or a training session can be indicated with `--train-dir`.
            In order of priority, the global settings (dtype, TensorFloat32, etc.) are taken from:
              (1) the model config (for a training session),
              or (2) the defaults.

            Prints only the heisenberg coefficients stdout; all other information is `logging.debug`ed to stderr.
            
            Store heisenberg coefficients in csv

            WARNING: Please note that results of CUDA models are rarely exactly reproducible, and that even CPU models can be nondeterministic.
            """
        )
    )
    parser.add_argument(
        "--train-dir",
        help="Path to a working directory from a training session.",
        type=Path,
        default="./",
    )
    parser.add_argument(
        "--model",
        help="A deployed or pickled NequIP model to load. If omitted, defaults to `best_model.pth` in `train_dir`.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--structure-path",
        help="A structure file to load data from. If omitted, `POSCAR` in `train_dir` will be used",
        type=Path,
        default="./POSCAR",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        help="Try to have PyTorch use deterministic algorithms. Will probably fail on GPU/CUDA.",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--log",
        help="log file to store all the metrics and screen logging.debug",
        type=Path,
        default="log",
    )
    parser.add_argument(
        "-z",
        "--magz",
        help="atomic number of atoms with magnetization, eg. 8,83 , default: 0,",
        type=str,
        default="0",
    )
    # Something has to be provided
    # See https://stackoverflow.com/questions/22368458/how-to-make-argparse-logging.debug-usage-when-no-option-is-given-to-the-code
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    # Parse the args
    args = parser.parse_args(args=args)

    magz = [int(item) for item in args.magz.split(",")]

    if args.train_dir:
        if args.model is None:
            args.model = args.train_dir / "jit.pth"

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if running_as_script:
        set_up_script_logger(args.log)
    logger = logging.getLogger("nequip-evaluate")
    logger.setLevel(logging.INFO)

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(
            "WARNING: please note that models running on CUDA are usually nondeterministc and that this manifests in the final test errors; for a _more_ deterministic result, please use `--device cpu`",
        )

    if args.use_deterministic_algorithms:
        logger.info(
            "Telling PyTorch to try to use deterministic algorithms... please note that this will likely error on CUDA/GPU"
        )
        torch.use_deterministic_algorithms(True)

    calc = NequIPCalculator.from_deployed_model(
        model_path=args.model, device=device, set_global_options=True
    )
    atom = read(args.structure_path)
    spin = np.random.normal(size=[atom.positions.shape[0], 3])
    spin = spin / np.linalg.norm(spin, axis=-1)[:, None]
    atom.set_initial_magnetic_moments(spin)

    atom.set_calculator(calc=calc)
    data = AtomicData.from_ase(atoms=atom, r_max=calc.r_max)
    data = calc.transform(data)
    data = AtomicData.to_AtomicDataDict(data)
    out = calc.model(data)

    if AtomicDataDict.EDGE_HB_COEFF_KEY in out:
        hb = out[AtomicDataDict.EDGE_HB_COEFF_KEY]
    elif AtomicDataDict.EDGE_HB_MATRIX_KEY in out:
        hb = out[AtomicDataDict.EDGE_HB_MATRIX_KEY]
    else:
        raise NotImplementedError("No HB coeff or matrix found in output")

    edge_index = out[AtomicDataDict.EDGE_INDEX_KEY]
    z = out[AtomicDataDict.ATOM_TYPE_KEY]
    edge_dist = out[AtomicDataDict.EDGE_LENGTH_KEY]
    pos = out[AtomicDataDict.POSITIONS_KEY]

    print("---EDGE_HB---")
    print("z1,  z2,  pos1,  pos2,  length,  HB")
    for i in range(hb.size(0)):
        src, dst = edge_index[:, i]
        if z[src] in magz and z[dst] in magz:
            print(int(z[src]), int(z[dst]), pos[src].detach().numpy(), pos[dst].detach().numpy(), float(edge_dist[i]))
            print(hb[i].detach().numpy())
    
    if AtomicDataDict.NODE_SPINANI_MATRIX_KEY in out:
        print("---NODE_SPINANI_MATRIX---")
        ani = out[AtomicDataDict.NODE_SPINANI_MATRIX_KEY]
        for i in range(pos.size(0)):
            if z[i] in magz:
                print(int(z[i]), pos[i].detach().numpy())
                print(ani[i].detach().numpy())

    if AtomicDataDict.EDGE_BIQUAD_COEFF_KEY in out:
        print("---EDGE_BIQUAD_COEFF---")
        print("z1,  z2,  pos1,  pos2,  length,  EDGE_BIQUAD_COEFF")
        biquad = out[AtomicDataDict.EDGE_BIQUAD_COEFF_KEY]
        for i in range(hb.size(0)):
            src, dst = edge_index[:, i]
            if z[src] in magz and z[dst] in magz:
                print(int(z[src]), int(z[dst]), pos[src].detach().numpy(), pos[dst].detach().numpy(), float(edge_dist[i]))
                print(biquad[i].detach().numpy())


if __name__ == "__main__":
    main()
