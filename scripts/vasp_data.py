import argparse
import textwrap
from nequip.utils import read_vasp_atom
import os
from ase.io import write
from tqdm import trange


def walk_folder_outcar(root_path, fname="OUTCAR"):
    file_list = []
    for home, dirs, files in os.walk(root_path):
        for file in files:
            if fname in file:
                file_list.append(os.path.join(home, file))
    return file_list


def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Search for `*fname*` in a folder and process it into `extxyz` dataset which can be loaded by ASEdataset"""
        )
    )
    parser.add_argument(
        "-p",
        "--path",
        help="folder path to search for *fname*, default: ./",
        type=str,
        default="./",
    )
    parser.add_argument(
        "-o",
        "--out-path",
        help="data.xyz path to store, default: ./data.extxyz",
        type=str,
        default="./data.extxyz",
    )
    parser.add_argument(
        "-m",
        "--lmagnet",
        help="read magnetization from fname, use -m to activate, default: false",
        action="store_true",
    )
    parser.add_argument(
        "-z",
        "--nomagz",
        help="atomic number of atoms without magnetization, eg. 8,83 , default: 0,",
        type=str,
        default="0",
    )
    parser.add_argument(
        "-n",
        "--min-mag-norm",
        help="magnetization lower than min_mag_norm will be set to zero, default: 0.5",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-c",
        "--cpu",
        help="Number of cpu used, default: 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-b",
        "--lborn",
        help="read Born effective charge from fname, use -b to activate, default: false",
        action="store_true",
    )
    parser.add_argument(
        "--fname",
        help="search file name including `fname`, default: OUTCAR",
        type=str,
        default="OUTCAR",
    )
    parser.add_argument(
        "-f",
        "--lforce_convergence",
        help="read all results no matter if it reaches convergence, use -f to activate, default: false",
        action="store_true",
    )

    # Parse the args
    args = parser.parse_args(args=args)
    folder_path = args.path
    output_file_path = args.out_path
    min_mag_norm = args.min_mag_norm
    lmagnetization_data = args.lmagnet
    lforce_convergence = args.lforce_convergence
    lborn = args.lborn
    nomagz = [int(item) for item in args.nomagz.split(",")]
    fname = args.fname
    cpu = args.cpu
    print("Search *{}* in {} with {} cpus".format(fname, folder_path, cpu))
    print("Store output in {}".format(output_file_path))
    if lmagnetization_data:
        print("Read magnetization from {}".format(fname))
        if nomagz != [0]:
            print(
                "Magnetization of atoms with atomic number {} set to zero as non-magnetic atoms. The rest are magnetic atoms.".format(
                    nomagz
                )
            )
        else:
            print(
                "Magnetization Norm of atoms under {} set to zero as non-magnetic atoms. The rest are magnetic atoms.".format(
                    min_mag_norm
                )
            )
    if lborn:
        print("Read born effective charges from {}".format(fname))

    outcar_paths = walk_folder_outcar(folder_path, fname=args.fname)

    atoms_list = []
    if cpu == 1:
        for i in trange(len(outcar_paths)):
            path = outcar_paths[i]
            atoms, lconvergence = read_vasp_atom(
                path,
                non_mag_atomic_number=nomagz,
                min_mag_norm=min_mag_norm,
                lmagnetization_data=lmagnetization_data,
                lnormalize_spin=True,
                lread_born=lborn,
                lread_piezo=lborn,
                lforce_convergence=lforce_convergence,
                lread_OSZICAR=True,
            )
            if lconvergence:
                atoms_list.append(atoms)
    else:
        from multiprocessing import Pool
        from functools import partial
        from tqdm import tqdm

        read_vasp_atom_partial = partial(
            read_vasp_atom,
            non_mag_atomic_number=nomagz,
            min_mag_norm=min_mag_norm,
            lmagnetization_data=lmagnetization_data,
            lnormalize_spin=True,
            lread_born=lborn,
            lread_piezo=lborn,
            lforce_convergence=lforce_convergence,
            lread_OSZICAR=True,
        )
        with Pool(processes=cpu) as pool:
            iter = pool.imap(read_vasp_atom_partial, outcar_paths)
            for idx, (atoms, lconvergence) in tqdm(
                enumerate(iter), total=len(outcar_paths)
            ):
                if lconvergence:
                    atoms_list.append(atoms)

    write(output_file_path, atoms_list, format="extxyz")
    print("Find {} {} files in {}.".format(len(outcar_paths), fname, folder_path))
    print(
        "{} of them have reached convergence and store in {}.".format(
            len(atoms_list), output_file_path
        )
    )


if __name__ == "__main__":
    main()
