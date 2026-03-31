import argparse
import textwrap
from ase.io import write, read
import ase
from tqdm import trange
from nequip.utils.data_outcar_spin import get_outcar_magnetization_born, bec_logout_convergence
from os.path import join as osjoin
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex", "bright"])

# Add BEC from OUTCAR.$i into atoms
def add_bec2data(outcar_path: str, atoms: ase.Atoms):
    spin, lconvergence, lmagnetization, born, lborn, piezo_tensor, lpiezo = get_outcar_magnetization_born(
        outcar_path
    )
    if lconvergence:
        assert lborn == True, f"bec data is not included in {outcar_path}"
        assert lpiezo == True, f"bec data is not included in {outcar_path}"

        atoms_outcar = read(outcar_path)

        assert judge_same_atoms(
            atoms, atoms_outcar
        ), f"atoms in {outcar_path} and {atoms} are not the same"

        atoms.set_array("born_effective_charges", born)
        atoms.info["piezoelectric"] = piezo_tensor
        return atoms
    else:
        return None


# judge if the atoms in OUTCAR.$i and data.extxyz are the same
def judge_same_atoms(atoms1: ase.Atoms, atoms2: ase.Atoms):
    a = atoms1.arrays
    b = atoms2.arrays

    # print((atoms1.cell - atoms2.cell))
    # print((a['positions'] - b['positions']))

    return (
        len(atoms1) == len(atoms2)
        and ((a["positions"] - b["positions"]) < 1e-5).all()
        and (a["numbers"] == b["numbers"]).all()
        and ((atoms1.cell - atoms2.cell) < 1e-5).all()
        and (atoms1.pbc == atoms2.pbc).all()
    )

def plot_bec_data(data):
    bec_all = []
    for i in trange(len(data)):
        d = data[i]
        bec = d.get_array("born_effective_charges")
        bec_all.append(bec)
    bec_all = np.concatenate(bec_all)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        data = bec_all[:, i]
        print(f"Min, Max {i}:", data.min(), data.max())
        label = f'bec_{i}'
        ax.hist(data, bins=20, color='purple', edgecolor='white', facecolor='purple', alpha=0.5, density=False)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel(label)
        ax.set_ylabel('Number (total:{})'.format(len(data)))
    plt.tight_layout()
    plt.savefig('bec_plot.png', dpi=600)
    plt.show()

def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Read born effective charge and piezoelectric tensor from OUTCAR.$i in a folder and add it into `extxyz` dataset"""
        )
    )
    parser.add_argument(
        "-d",
        "--dataset-path",
        help="original data.extxyz without born effective charge, default: ./data.extxyz",
        type=str,
        default="./data.extxyz",
    )
    parser.add_argument(
        "-o",
        "--out-path",
        help="ata.xyz path to store, default: ./data_bec.extxyz",
        type=str,
        default="./data_bec.extxyz",
    )
    parser.add_argument(
        "-f",
        "--folder",
        help="folder path where store all the OUTCAR.$i and $i coorespond to the order in original data.extxyz, default: ./result",
        type=str,
        default="./result",
    )
    parser.add_argument(
        "-c",
        "--criteria",
        help="the convergence criteria in log.out for bec calculation, default: 1e-2",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--cpu",
        help="Number of cpu used, default: 1",
        type=int,
        default=1,
    )

    # Parse the args
    args = parser.parse_args(args=args)
    dataset_path = args.dataset_path
    folder = args.folder
    criteria = args.criteria
    output_file_path = args.out_path
    cpu = args.cpu

    print(f"Reading original dataset file from {dataset_path} with {cpu} cpus...")
    dataset_og = read(dataset_path, index=":")
    data_num = len(dataset_og)
    print(f"Find {data_num} data.")

    print(f"Read OUTCAR.$i data and check convergence based on log.out.$i from {folder}")
    atoms_list = []
    if cpu == 1:
        for i in trange(data_num):
            lconvergence = bec_logout_convergence(osjoin(folder, "log.out." + str(i)), criteria)
            if lconvergence:
                atoms = add_bec2data(osjoin(folder, "OUTCAR." + str(i)), dataset_og[i].copy())
                if atoms is not None:
                    atoms_list.append(atoms)
    else:
        from multiprocessing import Pool
        from functools import partial
        from tqdm import tqdm
        
        bec_logout_convergence_partial = partial(bec_logout_convergence, criteria=criteria)
        
        path_list = [osjoin(folder, "log.out." + str(i)) for i in range(data_num)]
        
        with Pool(processes=cpu) as pool:
            iter = pool.imap(bec_logout_convergence_partial, path_list)
            for idx, lconvergence in tqdm(
                enumerate(iter), total=len(path_list)
            ):
                if lconvergence:
                    atoms = add_bec2data(osjoin(folder, "OUTCAR." + str(idx)), dataset_og[idx].copy())
                    if atoms is not None:
                        atoms_list.append(atoms)
        
    data_num_new = len(atoms_list)
    print(f"Store {data_num_new} data in {output_file_path}")
    write(output_file_path, atoms_list, format="extxyz")
    plot_bec_data(atoms_list)


if __name__ == "__main__":
    main()
