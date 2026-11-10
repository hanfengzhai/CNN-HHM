import os, argparse, tqdm, numpy as np, atomman as am
from pathlib import Path
from util_fft_homogenization import FFT_GOOSE

def compute_homogenized_stiffness(stiffness_voxel_file: str,
        stiffness_save_file: str, mat_type: str = "anisotropic",
        avg_scheme: str = "Hill", return_moduli=True):

    fft_solver = FFT_GOOSE(stiffness_voxel_file=stiffness_voxel_file)
    fft_solver.setup()
    homogenized_stiffness = fft_solver.solve_elasticity(mat_type=mat_type, avg_scheme=avg_scheme)
    homogenized_stiffness = 0.5 * (homogenized_stiffness + homogenized_stiffness.T)

    C = am.ElasticConstants(Cij=homogenized_stiffness)
    bulk_voigt  = C.bulk("Voigt")
    bulk_reuss  = C.bulk("Reuss")
    bulk_hill   = C.bulk("Hill")
    shear_voigt = C.shear("Voigt")
    shear_reuss = C.shear("Reuss")
    shear_hill  = C.shear("Hill")

    with stiffness_save_file.open(mode="a") as fid:
        fid.write(f"# Using {avg_scheme} averaged stiffness for each voxel \n\n")
        np.savetxt(fid, homogenized_stiffness, fmt="%12.4f %12.4f %12.4f %12.4f %12.4f %12.4f",)
        fid.write(f"\n#         Voigt         Reuss          Hill\n")
        fid.write(f"Bulk     {bulk_voigt:0.4f}       {bulk_reuss:0.4f}       {bulk_hill:0.4f}\n")
        fid.write(f"Shear    {shear_voigt:0.4f}       {shear_reuss:0.4f}       {shear_hill:0.4f}")
    
    if return_moduli: # for Gassman theory testing
        if avg_scheme == "Hill":
            return bulk_hill, shear_hill
        elif avg_scheme == "Voigt":
            return bulk_voigt, shear_voigt
        elif avg_scheme == "Reuss":
            return bulk_reuss, shear_reuss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="this program assemble the smaller images into a big one compute its average elastic moduli")
    parser.add_argument("-r", "--rock", required=True, type=str, help="rock type")
    parser.add_argument("-m", "--method", required=True, type=str, help="hstogram method")
    parser.add_argument("-S", "--size_big", required=True, type=int, help="size of the original big rock")
    parser.add_argument("-s", "--size_small", required=True, type=int, help="size of the smaller image to be assmbled")
    parser.add_argument("-p", "--padding", required=True, type=int, help="size of skin/padding used in the smaller image")
    parser.add_argument("-g", "--geodict", default=False, type=bool, help="if small image are solved using geodict")
    parser.add_argument("-o", "--filling", default='dry', type=str)
    parser.add_argument("-a", "--avg_scheme", default='Hill', type=str)
    args = parser.parse_args()

    output_dir = "results_by_user"
    os.makedirs(output_dir, exist_ok=True)
    result_dir = Path(f"{output_dir}/results_{args.size_big}_{args.size_small}")
    result_dir.mkdir(exist_ok=True)

    mat_type, avg_scheme = "anisotropic", args.avg_scheme  # relevant only for isotropic material
    stiffness_voxel_parent_dir = Path("data/")
    stiffness_voxel_file = stiffness_voxel_parent_dir.joinpath(f"stiffness_{args.method}_{args.size_big}_{args.size_small}_{args.filling}.dat")
    print(f"processing rocks of size {args.size_big/args.size_small: .0f}")

    stiffness_file = Path(f"{result_dir}/stiffness_{args.method}_{args.size_big}_{args.size_small:.0f}_{mat_type}_{args.filling}.dat")
    stiffness_file.unlink(missing_ok=True) # missing_ok=True
    compute_homogenized_stiffness(stiffness_voxel_file, stiffness_file, mat_type=mat_type, avg_scheme=avg_scheme)