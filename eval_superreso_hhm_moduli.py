import matplotlib.pyplot as plt
import os, re, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm
# from util_fft_homogenization import init_color_object
from fft_homogenization import compute_homogenized_stiffness
from matplotlib.ticker import NullFormatter
from util_elasticity import process_homogenization, compute_moduli_from_y_nu
from util_visualization import figures_hhm_moduli_eval, plotting_params
# ================ manual import parameters ================
plt.rcParams.update(plotting_params)
# ===========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Validate Gassmann equation for HHM on generated superresolution rocks")
    parser.add_argument("-K_sat",   "--K_sat",   default=80,                    type=float)
    parser.add_argument("-Y_min",   "--Y_min",   default=95.29411764705883,     type=float)
    parser.add_argument("-nu_min",  "--nu_min",  default=0.058823529411764705,   type=float)
    parser.add_argument("-Y_air",   "--Y_air",   default=2.903225806452e-04,     type=float)
    parser.add_argument("-nu_air",  "--nu_air",  default=0.4516129032258,       type=float)
    parser.add_argument("-Y_wat",   "--Y_wat",   default=2.999954546143e-04,     type=float)
    parser.add_argument("-nu_wat",  "--nu_wat",  default=0.4999772730716,       type=float)
    parser.add_argument("-phi",     "--phi",     default=0.18913,                type=float) # porosity
    parser.add_argument("-avg_scheme", "--avg_scheme", default='Hill', type=str)
    parser.add_argument("-method", "--method", default='OT', type=str)
    parser.add_argument("-mat_type", "--mat_type", default='anisotropic', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    global params
    params = vars(parse_args())
    params['phi'] = params['phi']
    params['K_min'], params['G_min'] = compute_moduli_from_y_nu(params['Y_min'], params['nu_min'])
    params['K_dry_ref'], params['G_dry_ref'] = compute_moduli_from_y_nu(params['Y_air'], params['nu_air'])
    params['K_wat'], params['G_wat'] = compute_moduli_from_y_nu(params['Y_wat'], params['nu_wat'])

    vox_dir = Path("data/")
    method, mat_type = params['method'], params['mat_type']
    avg_scheme = params['avg_scheme']
    all_results = {}

    for size_big in [400, 800]:
        size_smalls = [100, 200, 400] if size_big == 400 else [100, 200, 400, 800]
        res = process_homogenization(vox_dir, method, mat_type, avg_scheme, size_big, size_smalls)
        all_results[size_big] = res
    
    fig_dir = Path("figures/"); fig_dir.mkdir(exist_ok=True)
    fig_hhm_mod = figures_hhm_moduli_eval(fig_dir="figures")

    x400, x800 = all_results[400]['size_smalls'], all_results[800]['size_smalls']
    for filling in ['dry', 'sat']:
        filling_key = 'Saturated' if filling == 'sat' else 'Dry'
        fig, (ax_b, ax_s) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

        # ================= Bulk subplot ================
        y400_bulk, y800_bulk = all_results[400][f'K_{filling}'], all_results[800][f'K_{filling}']
        fig_hhm_mod.plot_hhm_fit_ax(ax_b,
            x400, y400_bulk, x800, y800_bulk,
            ylabel=r'$K$ [GPa]',
            labels=(rf'{filling_key} rock: $\mathbb{{R}}^{{400}}$', 
                    rf'{filling_key} rock: $\mathbb{{R}}^{{800}}$'),
            fit_labels=(None, None), marker_label='(a)')

        # ================= Shear subplot =================
        y400_shear, y800_shear = all_results[400][f'G_{filling}'], all_results[800][f'G_{filling}']
        fig_hhm_mod.plot_hhm_fit_ax(ax_s,
            x400, y400_shear, x800, y800_shear,
            ylabel=r'$G$ [GPa]',
            labels=(rf'{filling_key} rock: $\mathbb{{R}}^{{400}}$', 
                    rf'{filling_key} rock: $\mathbb{{R}}^{{800}}$'),
            fit_labels=(None, None), marker_label='(b)')

        plt.tight_layout()
        out = f"{fig_dir}/moduli_comparison_{filling}_{avg_scheme}.png"
        plt.savefig(out, dpi=300); plt.close()
        print(f"---> saved figure {out}")

    for fn in Path(".").glob("*.dat"):
        fn.unlink()