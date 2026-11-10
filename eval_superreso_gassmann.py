import argparse, os, numpy as np
from pathlib import Path
from fft_homogenization import compute_homogenized_stiffness
from util_elasticity import compute_moduli_from_y_nu, gassmann_bulk, process_homogenization
from util_visualization import figures_gassmann_eval
import warnings
warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*", category=UserWarning)

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
    return parser.parse_args()


if __name__ == "__main__":
    global params
    params = vars(parse_args())
    params['phi'] = params['phi']
    params['K_min'], params['G_min'] = compute_moduli_from_y_nu(params['Y_min'], params['nu_min'])
    params['K_dry_ref'], params['G_dry_ref'] = compute_moduli_from_y_nu(params['Y_air'], params['nu_air'])
    params['K_wat'], params['G_wat'] = compute_moduli_from_y_nu(params['Y_wat'], params['nu_wat'])

    fig_gassmann = figures_gassmann_eval(fig_dir="figures")

    vox_dir = Path("data/")
    method, mat_type = 'OT', 'anisotropic'
    avg_scheme = params['avg_scheme']
    all_results = {}

    for size_big in [400, 800]:
        size_smalls = [100, 200, 400] if size_big == 400 else [100, 200, 400, 800]
        res = process_homogenization(vox_dir, method, mat_type, avg_scheme, size_big, size_smalls)
        # attach error calculations
        res['K_err'] = [abs(gassmann_bulk(K_dry, params['K_min'], params['phi'], params['K_wat']) - K_sat) / K_sat
                        for K_dry, K_sat in zip(res['K_dry'], res['K_sat'])]
        res['G_err'] = [abs(G_dry - G_sat) / G_sat for G_dry, G_sat in zip(res['G_dry'], res['G_sat'])]
        # prediction arrays
        res['K_pred'] = [gassmann_bulk(K_dry, params['K_min'], params['phi'], params['K_wat'])
                         for K_dry in res['K_dry']]
        res['G_pred'] = res['G_dry']

        fig_gassmann.plot_rel_errors(res, size_big, avg_scheme, params)
        fig_gassmann.plot_moduli_comparison(res, size_big, avg_scheme, params)
        all_results[size_big] = res

    fig_gassmann.plot_combined_rel_err(all_results)

    for fn in Path(".").glob("*.dat"):
        fn.unlink()