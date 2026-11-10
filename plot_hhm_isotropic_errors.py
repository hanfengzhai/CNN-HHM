# /path/to/digital_rock/rock_images_segmented/B1/size_1200/skin_0/size_600
# /path/to/digital_rock/rock_images_segmented/B1/size_1200/skin_0/size_300
import os, sys, argparse, glob, torch, numpy as np, atomman as am
import pandas as pd, seaborn as sns
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score
from fft_homogenization import compute_homogenized_stiffness
from util_visualization import plotting_params, TerminalColors as colors
from util_machine_learning import MLDRP, compute_moduli_atomman
sys.path.append('data')
from order import read_rock_matrices, write_matrices_to_dat
from eval_cnn_hhm_moduli import homogenize_fullcube_stiffness, split_stiffness_file
plt.rcParams.update(plotting_params)
np.random.seed(2025); torch.manual_seed(2025); torch.cuda.manual_seed(2025)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate errors in homogenized stiffness matrices using isotropic assumption.")
    parser.add_argument("--fullcube_stiff_dir", type=str,
                        default="/path/to/digital_rock/FFT_python/real_rock_parameters/two_phase/runs_geodict/B1/size_900/skin_0/",
                        help="Directory containing true fullcube stiffness .dat files (stiffness_300.dat)")
    parser.add_argument("--dat_file", type=str, # /path/to/digital_rock/FFT_python/real_rock_parameters/two_phase/runs_geodict/B1/size_900/skin_0
                        default="data/stiffness_B1_900_75.dat",)
    # -------------------------------------------------------
    parser.add_argument("--basecube_size", type=int, default=900)
    parser.add_argument("--bigcube_size", type=int, default=300)
    parser.add_argument("--subcube_size", type=int, nargs='+', default=75)
    parser.add_argument("--avg_scheme", type=str, default="Hill", help="Average scheme for FFT homogenization (Hill, Voigt, Reuss)")
    parser.add_argument("--mat_type", type=str, default="isotropic",)
    parser.add_argument("--figures_dir", type=str, default="figures", help="Directory to save figures")
    parser.add_argument("--med_stiff_dir", type=str, default="data_hhm", help="storing intermediate stiffness files")
    return parser.parse_args()

def process_stiffness_isotropic(input_path: str, output_path: str):
    with open(input_path, 'r') as f:
        all_lines = f.readlines()
    nlines = len(all_lines)
    if nlines < 8:
        raise RuntimeError("Input file is too short to contain a 6×6 block.")

    isotropic_blocks = []
    comments = []
    for start in range(0, nlines, 10):
        chunk = all_lines[start : start + 10]
        if len(chunk) < 8:
            break  # not enough lines for another block
        comment_line = chunk[0].rstrip("\n")
        comments.append(comment_line)
        mat = []
        try:
            for rowtxt in chunk[2:8]:
                vals = [float(x) for x in rowtxt.strip().split()]
                if len(vals) != 6:
                    raise ValueError
                mat.append(vals)
        except ValueError:
            comments.pop()
            continue

        Cij = np.array(mat, dtype=float)
        # Compute Hill‐averaged K, G via atomman
        Cij_symm = .5 * (Cij + Cij.T)  # Ensure symmetry
        C_am    = am.ElasticConstants(Cij=Cij_symm)
        K_hill  = C_am.bulk("Hill")
        G_hill  = C_am.shear("Hill")

        lam = K_hill - (2.0/3.0)*G_hill
        C11 = lam + 2.0*G_hill
        C12 = lam
        C44 = G_hill
        C_iso = np.array([[ C11,  C12,  C12,  0.0,  0.0,  0.0 ],
                          [ C12,  C11,  C12,  0.0,  0.0,  0.0 ],
                          [ C12,  C12,  C11,  0.0,  0.0,  0.0 ],
                          [ 0.0,  0.0,  0.0,  C44,  0.0,  0.0 ],
                          [ 0.0,  0.0,  0.0,  0.0,  C44,  0.0 ],
                          [ 0.0,  0.0,  0.0,  0.0,  0.0,  C44 ],], dtype=float)
        isotropic_blocks.append(C_iso)
    with open(output_path, 'w') as fout:
        for idx, C_iso in enumerate(isotropic_blocks):
            fout.write(f"{comments[idx]}\n")
            fout.write("\n")
            for row in C_iso:
                fout.write("    ".join(f"{val:12.4f}" for val in row) + "\n")
            fout.write("\n\n")
    print(f"Processed {len(isotropic_blocks)} block(s); output written to '{output_path}'.")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.med_stiff_dir, exist_ok=True)
    process_stiffness_isotropic(input_path=args.dat_file, output_path=f'{args.med_stiff_dir}/stiffness_B1_900_75_isotropic.dat')

    split_stiffness_file(input_path=args.dat_file,
                         output_dir=f'{args.med_stiff_dir}/anisotropic',
                         chunk_size=int((args.bigcube_size / args.subcube_size)**3),)
    split_stiffness_file(input_path=f'{args.med_stiff_dir}/stiffness_B1_900_75_isotropic.dat',
                         output_dir=f'{args.med_stiff_dir}/isotropic',
                         chunk_size=int((args.bigcube_size / args.subcube_size)**3),)
    split_stiffness_file(input_path=f'{args.fullcube_stiff_dir}/stiffness_{args.bigcube_size}.dat',
                         output_dir=f'{args.med_stiff_dir}/dns',
                         chunk_size=1, target_size= args.bigcube_size)
    # (1) HHM using isotropic assumption
    bulk_large_iso, shear_large_iso = [], []
    for i in tqdm(range(1,int(args.basecube_size / args.bigcube_size)**3+1), desc="Processing isotropic stiffness"):
        stiff_file = os.path.join(f'{args.med_stiff_dir}/isotropic', f"stiffness_{args.subcube_size}_{i}.dat")
        bulk_large, shear_large = homogenize_fullcube_stiffness(Path(stiff_file), rock_id=i,
                                                                data_dir=f'{args.med_stiff_dir}/iso_300',
                                                                mat_type=args.mat_type,
                                                                avg_scheme=args.avg_scheme)
        bulk_large_iso.append(bulk_large); shear_large_iso.append(shear_large)
    bulk_large_iso, shear_large_iso = np.array(bulk_large_iso).flatten(), np.array(shear_large_iso).flatten()

    # (2) HHM with anisotropic stiffness using FFT
    bulk_large_aniso, shear_large_aniso = [], []
    for i in tqdm(range(1,int(args.basecube_size / args.bigcube_size)**3+1), desc="Processing anisotropic stiffness"):
        stiff_file = os.path.join(f'{args.med_stiff_dir}/anisotropic', f"stiffness_{args.subcube_size}_{i}.dat")
        bulk_large, shear_large = homogenize_fullcube_stiffness(Path(stiff_file), rock_id=i,
                                                                data_dir=f'{args.med_stiff_dir}/aniso_300',
                                                                mat_type=args.mat_type,
                                                                avg_scheme=args.avg_scheme)
        bulk_large_aniso.append(bulk_large); shear_large_aniso.append(shear_large)
    bulk_large_aniso, shear_large_aniso = np.array(bulk_large_aniso).flatten(), np.array(shear_large_aniso).flatten()

    # (3) Large rock stiffness matrices
    fullcube_stiff_file = os.path.join(args.fullcube_stiff_dir, f"stiffness_{args.bigcube_size}.dat")
    bulk_large_dns, shear_large_dns = [], []
    for i in tqdm(range(1,int(args.basecube_size / args.bigcube_size)**3+1), desc="Processing DNS stiffness"):
        stiff_file = os.path.join(f'{args.med_stiff_dir}/dns', f"stiffness_{args.bigcube_size}_{i}.dat")
        bulk_large, shear_large = homogenize_fullcube_stiffness(Path(stiff_file), rock_id=i,
                                                                data_dir=f'{args.med_stiff_dir}/dns_300',
                                                                mat_type=args.mat_type,
                                                                avg_scheme=args.avg_scheme)
        bulk_large_dns.append(bulk_large); shear_large_dns.append(shear_large)
    bulk_large_dns, shear_large_dns = np.array(bulk_large_dns).flatten(), np.array(shear_large_dns).flatten()

    for file in glob.glob("*.dat"):
        os.remove(file)
    error_hhm_iso = np.array([np.abs(bulk_large_dns - bulk_large_iso), np.abs(shear_large_dns - shear_large_iso)])
    error_hhm_aniso = np.array([np.abs(bulk_large_dns - bulk_large_aniso), np.abs(shear_large_dns - shear_large_aniso)])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(error_hhm_iso[0], label='HHM w/ isotropic', color='blue')
    plt.plot(error_hhm_aniso[0], label='HHM w/ anisotropic', color='red', linestyle='-.')
    plt.yscale('log')
    plt.ylabel('Bulk modulus error (GPa)')
    plt.xlabel('Rock ID')
    plt.subplot(1, 2, 2)
    plt.plot(error_hhm_iso[1], label='HHM w/ isotropic', color='blue')
    plt.plot(error_hhm_aniso[1], label='HHM w/ anisotropic', color='red', linestyle='-.')
    plt.yscale('log')
    plt.ylabel('Shear modulus error (GPa)')
    plt.xlabel('Rock ID')
    plt.legend()
    plt.tight_layout()
    os.makedirs(args.figures_dir, exist_ok=True)
    plt.savefig(f"{args.figures_dir}/hhm_isotropic_errors.png", dpi=300)
    plt.close()
    print(f'---> saved figure to {args.figures_dir}/hhm_isotropic_errors.png')
    
    error_iso_aniso = np.abs(error_hhm_iso - error_hhm_aniso)
    plt.figure(figsize=(5, 5))
    plt.plot(error_iso_aniso[0], label=r'Bulk modulus error ($K^{\rm iso}$ - $K^{\rm aniso}$)', color='blue')
    plt.plot(error_iso_aniso[1], label=r'Shear modulus error ($G^{\rm iso}$ - $G^{\rm aniso}$)', color='red', linestyle='-.')
    plt.yscale('log')
    plt.ylim(1e-8, 1e-5)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{args.figures_dir}/hhm_iso_aniso_errors.png", dpi=300)
    plt.close()
    print(f'---> saved figure to {args.figures_dir}/hhm_iso_aniso_errors.png')
    