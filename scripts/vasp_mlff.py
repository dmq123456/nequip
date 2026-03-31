import argparse
import textwrap
from ase.io import write, read
import ase
from tqdm import trange
import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np


def mlabn_load_n(df: pd.DataFrame, atom_base: ase.Atoms, index: int):
    """Load ML_ABN file and return ase.Atoms"""
    atom = atom_base.copy()
    cell_index = df[df[0].isin(["Primitive"])].index[index]
    cell = df.iloc[cell_index + 2 : cell_index + 5, 0:3].values
    position = df.iloc[cell_index + 8 : cell_index + 8 + len(atom), 0:3].values
    energy = df.iloc[cell_index + 11 + len(atom), 0]
    force = df.iloc[
        cell_index + 15 + len(atom) : cell_index + 15 + 2 * len(atom), 0:3
    ].values
    stress_xxyyzz = df.iloc[cell_index + 20 + 2 * len(atom), 0:3].values
    stress_xyyzzx = df.iloc[cell_index + 24 + 2 * len(atom), 0:3].values
    stress_og = np.concatenate((stress_xxyyzz, stress_xyyzzx))  # In kbar
    # Transform stress to ASE format
    stress_arr = -np.array(stress_og, dtype=np.float32)
    stress_arr = stress_arr[[0, 1, 2, 4, 5, 3]] * 1e-1 * ase.units.GPa

    atom.set_cell(cell)
    atom.set_positions(position)
    atom.calc = SinglePointCalculator(
        energy=energy,
        forces=force,
        stress=stress_arr,
        atoms=atom,
    )
    return atom


def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Read ML_ABN file and store it into `extxyz` dataset which can be loaded by ASEdataset"""
        )
    )
    parser.add_argument(
        "-m",
        "--mlabn-path",
        help="ML_ABN path, default: ./ML_ABN",
        type=str,
        default="./ML_ABN",
    )
    parser.add_argument(
        "-p",
        "--poscar-path",
        help="POSCAR path related to ML_ABN, default: ./POSCAR",
        type=str,
        default="./POSCAR",
    )
    parser.add_argument(
        "-o",
        "--out-path",
        help="ata.xyz path to store, default: ./data.extxyz",
        type=str,
        default="./data.extxyz",
    )

    # Parse the args
    args = parser.parse_args(args=args)
    mlabn_path = args.mlabn_path
    poscar_path = args.poscar_path
    output_file_path = args.out_path

    print(f"Reading ML_ABN file from {mlabn_path}...")
    df = pd.read_table(
        mlabn_path, header=None, sep="\s+", names=range(10), on_bad_lines=None
    )
    data_num = int(df.iloc[4, 0])
    print(f"Find {data_num} data.")

    print(f"Read POSCAR data from {poscar_path}")
    atom_base = read(poscar_path)

    atoms_list = []
    for i in trange(data_num):
        atoms = mlabn_load_n(df=df, atom_base=atom_base, index=i)
        atoms_list.append(atoms)
    print(f"Store {data_num} data in {output_file_path}")
    write(output_file_path, atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
