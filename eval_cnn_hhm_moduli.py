import os, sys, argparse, torch, random, time, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score
from fft_homogenization import compute_homogenized_stiffness
from util_visualization import plotting_params, TerminalColors as colors
from util_machine_learning import MLDRP, resolve_checkpoint_path, compute_moduli_atomman, stiffness_symmetrize, stiffness_upper_triangle_21
from util_data_processing import load_scans_and_labels
sys.path.append('data')
from order import read_rock_matrices, write_matrices_to_dat
plt.rcParams.update(plotting_params)
seed = 2025
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dry-rock moduli: CNN+HHM inference and FFT-based homogenization")
    parser.add_argument("--largecube_size", type=int, default=300,  choices=[300, 600],) # you can select 600 only if fullrock_size is 1200
    parser.add_argument("--fullrock_size",  type=int, default=1200, choices=[1200, 900],)
    parser.add_argument("--rock_type",      type=str, default="B1", choices=["B1", "B2", "CG", "FB1", "FB2"], help="Rock type")
    parser.add_argument("--fine_tune",      action="store_true", help="Fine-tune the model")
    # ------------------ CNN model variables --------------------
    parser.add_argument("--batch_size",     type=int, default=1, help="Batch size for training")
    parser.add_argument("--benchmark_model",   action="store_true", help="Use Benchmark 1 (prior) model for homogenization")
    parser.add_argument("--homogenization", action="store_true", help="Predict effective moduli (2 outputs) from stiffness (36 outputs)")
    parser.add_argument("--hs_bound",       action="store_true", help="Predict HS fraction (2 outputs) from stiffness (36 outputs)")
    parser.add_argument("--E1",  type=float, help="Young's modulus for pore",    default=2.903225806452e-04)
    parser.add_argument("--nu1", type=float, help="Poisson ratio for pore",      default=0.4516129032258)
    parser.add_argument("--E2",  type=float, help="Young's modulus for mineral", default=95.29411764705883)
    parser.add_argument("--nu2", type=float, help="Poisson ratio for mineral",   default=0.058823529411764705)
    parser.add_argument("--model_dir",     type=str, default="model", help="Directory to save model checkpoints")
    parser.add_argument("--stop_option",   type=str, default='test') # placeholder for model deployment
    parser.add_argument("-loss_avg_step",  "--loss_avg_step", type=int, default=15)
    parser.add_argument("--early_stop",    action="store_true", default=False, help="Enable early stopping")
    parser.add_argument("--epochs",        type=int, default=50, help="Number of training epochs")
    # ----------------------------------------------------------
    parser.add_argument("--scan_shape",    type=int, nargs=3, default=[75, 75, 75])
    parser.add_argument("--subcubes_per_fullcube", type=int, default=64)
    parser.add_argument("--size_big",      type=int, default=300)
    parser.add_argument("--size_smalls",   type=int, nargs='+', default=[75], help="List of subcube sizes for FFT homogenization")
    parser.add_argument("--avg_scheme",    type=str, default="Hill", help="Average scheme for FFT homogenization (Hill, Voigt, Reuss)")
    parser.add_argument("--mat_type",      type=str, default="isotropic",)
    parser.add_argument("--figures_dir",   type=str, default="figures", help="Directory to save figures")
    parser.add_argument("--med_stiff_dir", type=str, default="data_hhm", help="storing intermediate stiffness files")
    parser.add_argument("--result_dir",    type=str, default="data", help="Directory to save results")
    return parser.parse_args()

def homogenize_fullcube_stiffness(subcube_stiff_file, fullcube_stiff_file=None, rock_id=None,
                                  data_dir=None, mat_type='anisotropic', avg_scheme='Hill'):
    '''full-cube stands for R300 rock here'''
    bulk_mods, shear_mods = [], []
    if fullcube_stiff_file is None:
        fullcube_stiff_file = Path(f"stiffness_homogenize_300.dat")
        fullcube_stiff_file.unlink(missing_ok=True)
    if rock_id is not None:
        os.makedirs(f"{data_dir}", exist_ok=True)
        fullcube_stiff_file = Path(f"{data_dir}/stiffness_homogenize_300_{rock_id}.dat")
        fullcube_stiff_file.unlink(missing_ok=True)
    try:
        bulk, shear = compute_homogenized_stiffness(subcube_stiff_file, fullcube_stiff_file,
                                                    mat_type=mat_type, avg_scheme=avg_scheme,
                                                    return_moduli=True)
    except Exception as e:
        print(f"Error <compute_homogenized_stiffness()> {subcube_stiff_file}: {e}")
        bulk, shear = 1.0, 1.0
    bulk_mods.append(bulk); shear_mods.append(shear)
    return bulk_mods, shear_mods

def split_stiffness_file(input_path: str, output_dir: str,
                         chunk_size: int = 64, target_size: int = 75):
    os.makedirs(output_dir, exist_ok=True)
    blocks, current = [], [] # parse the master file into blocks
    with open(input_path) as f:
        for line in f:
            if line.startswith('# file'):
                if current:
                    blocks.append(current)
                current = [line]
            elif current and line.strip():
                current.append(line)
        if current:
            blocks.append(current)
    total = len(blocks) # split into chunks and write files
    n_files = (total + chunk_size - 1) // chunk_size
    for i in tqdm(range(n_files), desc="Splitting blocks"):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        out_path = os.path.join(output_dir, f"stiffness_{target_size}_{i+1}.dat")
        with open(out_path, 'w') as out:
            for block in blocks[start:end]:
                out.write(block[0] + '\n')
                out.writelines(block[1:])
                out.write('\n\n')

def plot_moduli(ax, fft, cnn, xlabel, ylabel):
    r2 = r2_score(fft, cnn)
    lb = min(fft.min(), cnn.min()); ub = max(fft.max(), cnn.max())
    ax.scatter(fft, cnn, c='darkblue', label=rf'$R^2$ = {r2:.3f}')
    ax.plot([lb, ub], [lb, ub], 'r--', lw=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend()


if __name__ == "__main__":
    args = parse_args()
    chunk_size = int( (args.largecube_size / args.scan_shape[0])**3 )
    largecube_number = int( (args.fullrock_size / args.scan_shape[0])**3 / chunk_size )

    # 1. CNN inference setup (prefer Drive-downloaded size75/<rock>/)
    local_scan = os.path.join('size75', args.rock_type, 'images')
    local_dat = os.path.join('size75', args.rock_type, 'stiffness.dat')
    if os.path.isdir(local_scan) and os.path.isfile(local_dat):
        args.scan_dir = local_scan
        args.dat_file = local_dat
    else:
        args.scan_dir = os.environ.get(
            'CNN_HHM_SCAN_DIR',
            os.path.join('/path/to/digital_rock/machine_learning/bulk_shear_learning/size_75/data',
                         f'data_{args.rock_type}', 'skin_0', 'images'))
        args.dat_file = os.environ.get(
            'CNN_HHM_DAT_FILE',
            os.path.join('/path/to/digital_rock/machine_learning/bulk_shear_learning/size_75/data',
                         f'data_{args.rock_type}', 'skin_0', 'stiffness.dat'))

    ml = MLDRP(args)
    ml.load_data()
    device = ml.device
    # --------- specify the model file name based on the homogenization option ----------
    if args.homogenization:   model_file = os.path.join(args.model_dir, f"cnn3d_rock_moduli_{args.rock_type}.pt")
    elif args.hs_bound:
        if args.benchmark_model: model_file = os.path.join(args.model_dir, f"cnn3d_rock_benchmark_hs_bound_{args.rock_type}.pt")
        else:                 model_file = os.path.join(args.model_dir, f"cnn3d_rock_hs_bound_{args.rock_type}.pt")
    else:                     model_file = os.path.join(args.model_dir, f"cnn3d_rock_stiffness_{args.rock_type}.pt")
    # -----------------------------------------------------------------------------------
    model_file = resolve_checkpoint_path(model_file)
    ckpt = torch.load(model_file, map_location=device)
    ml.model.load_state_dict(ckpt)
    ml.model.to(device).eval()

    print(f"* {colors.BLUE}Loaded model from{colors.RESET} {model_file}: {ml.model}")
    if args.fine_tune:
        args.figures_dir = os.path.join(args.figures_dir, f"fine_tune_{args.rock_type}")
        args.result_dir  = os.path.join(args.result_dir, f"fine_tune_{args.rock_type}")
        os.makedirs(args.figures_dir, exist_ok=True); os.makedirs(args.result_dir, exist_ok=True)
        ml.model_dir = os.path.join(args.model_dir, f"fine_tune_{args.rock_type}")
        ml.figs_ml.fig_dir = args.figures_dir
        if os.path.isfile(f'{ml.model_dir}/cnn3d_rock_{ml.key}.pt'):
            print(f"* {colors.BLUE}Loading fine-tuned model from{colors.RESET} {ml.model_dir}/cnn3d_rock_{ml.key}.pt")
            ckpt = torch.load(f'{ml.model_dir}/cnn3d_rock_{ml.key}.pt', map_location=device)
            ml.model.load_state_dict(ckpt)
        else: # no fine-tuned model found, start training
            t_start_fine_tune = time.time()
            ml.batch_size = 8 # for training process in fine-tuning
            ml.prepare_train_test()
            ml.train_model()
            t_end_fine_tune = time.time()
            print(f"* {colors.BLUE}Fine-tuned model time:{colors.RESET} {t_end_fine_tune - t_start_fine_tune:.2f} seconds")

    # 2. Predict subcube stiffness/moduli
    os.makedirs(args.med_stiff_dir, exist_ok=True)
    ml.deploy_cnn_stiffness_pred(output_file=f'{args.med_stiff_dir}/cnn_pred',)
    # reorder the stiffness matrices in the stiffness.dat file
    if args.homogenization:   cnn_stiffness_file = f'{args.med_stiff_dir}/cnn_pred_moduli.dat'
    elif args.hs_bound:
        if args.benchmark_model: cnn_stiffness_file = f'{args.med_stiff_dir}/cnn_pred_benchmark_hs_bound_stiffness.dat'
        else:                 cnn_stiffness_file = f'{args.med_stiff_dir}/cnn_pred_hs_bound_stiffness.dat'
    else:                     cnn_stiffness_file = f'{args.med_stiff_dir}/cnn_pred_stiffness.dat'
    rock_matrices = read_rock_matrices(cnn_stiffness_file, apply_increment=False)
    write_matrices_to_dat(rock_matrices, cnn_stiffness_file)

    # (1) Using HHM to predict the moduli of size-300 "full"-cubes based on size-75 subcubes
    split_stiffness_file(input_path=cnn_stiffness_file, output_dir=f'{args.med_stiff_dir}/CNN_small',
                         chunk_size=chunk_size)
    bulk_large_cnn, shear_large_cnn = [], []
    for i in tqdm(range(1, largecube_number)):
        stiff_file_75 = os.path.join(f'{args.med_stiff_dir}/CNN_small', f"stiffness_75_{i}.dat")
        try: # avoid unsuccessful HHM calculation due to low-quality prediction from CNN (primary from Benchmark 1 -- FB2)
            bulk_large, shear_large = homogenize_fullcube_stiffness(Path(stiff_file_75), rock_id=i,
                                                                    data_dir=f'{args.med_stiff_dir}/CNN_75_homog', 
                                                                    mat_type=args.mat_type, avg_scheme=args.avg_scheme)
        except Exception as e:
            bulk_large, shear_large = 1.0, 1.0
            print(f"Error <homogenize_fullcube_stiffness()> {stiff_file_75}: {e}")
            continue
        bulk_large_cnn.append(bulk_large); shear_large_cnn.append(shear_large)
    bulk_large_cnn, shear_large_cnn = np.array(bulk_large_cnn).flatten(), np.array(shear_large_cnn).flatten()

    # (2) Load the FFT-calculated size-75 subcubes and homogenize their moduli for comparison
    split_stiffness_file(input_path=args.dat_file, output_dir=f'{args.med_stiff_dir}/FFT_small',#'data_explore_hhm/size_900_300_75', #
                         chunk_size=chunk_size)
    bulk_large_fft, shear_large_fft = [], []
    for i in tqdm(range(1, largecube_number)):
        stiff_file_75 = os.path.join(f'{args.med_stiff_dir}/FFT_small', f"stiffness_75_{i}.dat")
        bulk_large, shear_large = homogenize_fullcube_stiffness(Path(stiff_file_75), rock_id=i,
                                                                data_dir=f'{args.med_stiff_dir}/FFT_75_homog', 
                                                                mat_type=args.mat_type, avg_scheme=args.avg_scheme)
        bulk_large_fft.append(bulk_large); shear_large_fft.append(shear_large)
    bulk_large_fft, shear_large_fft = np.array(bulk_large_fft).flatten(), np.array(shear_large_fft).flatten()

    # Eliminate components equal to 1.0
    mask = (bulk_large_fft != 1.0) & (bulk_large_cnn != 1.0) & (shear_large_fft != 1.0) & (shear_large_cnn != 1.0)
    bulk_large_fft, bulk_large_cnn = bulk_large_fft[mask], bulk_large_cnn[mask]
    shear_large_fft, shear_large_cnn = shear_large_fft[mask], shear_large_cnn[mask]

    plt.figure(figsize=(9, 5))
    ax = plt.subplot(121); plot_moduli(ax, bulk_large_fft,  bulk_large_cnn,  r'$K_{\rm eff}^{\rm DNS}$ [GPa]', r'$K_{\rm eff}^{\rm CNN}$ [GPa]')
    ax = plt.subplot(122); plot_moduli(ax, shear_large_fft, shear_large_cnn, r'$G_{\rm eff}^{\rm DNS}$ [GPa]', r'$G_{\rm eff}^{\rm CNN}$ [GPa]')
    plt.tight_layout()

    if args.homogenization:   keyword = 'moduli'
    elif args.hs_bound:
        if args.benchmark_model: keyword = 'benchmark_hs_bound'
        else:                 keyword = 'hs_bound'
    else:                     keyword = 'stiffness'

    plt.savefig(os.path.join(args.figures_dir, f"cnn_hhm_moduli_{args.rock_type}_{keyword}_large{args.largecube_size}_full{args.fullrock_size}.png"), dpi=300)
    np.save(os.path.join(args.result_dir, f"cnn_hhm_moduli_{args.rock_type}_{keyword}_large{args.largecube_size}_full{args.fullrock_size}.npy"),
            np.array([bulk_large_fft, bulk_large_cnn, shear_large_fft, shear_large_cnn]))
    np.save(os.path.join(args.result_dir, f"loss_{args.rock_type}_{keyword}_large{args.largecube_size}_full{args.fullrock_size}.npy"),
            np.array([ml.train_loss_list, ml.test_loss_list]))
    print(f'---> saved figure to {args.figures_dir}/cnn_hhm_moduli_{args.rock_type}_{keyword}_large{args.largecube_size}_full{args.fullrock_size}.png')
    print(f'R2 score of bulk and shear moduli: {r2_score(bulk_large_fft, bulk_large_cnn):.3f}, {r2_score(shear_large_fft, shear_large_cnn):.3f}')

    # Stiffness model only: parity plot for 21 independent stiffness constants (symmetry-imposed)
    if keyword == 'stiffness':
        X, Y, file_list, _ = load_scans_and_labels(args.scan_dir, args.dat_file, tuple(args.scan_shape))
        rock_dict = read_rock_matrices(cnn_stiffness_file, apply_increment=False)
        pred_list = []
        for fname in file_list:
            rock_id = int(Path(fname).stem.split('_')[-1])
            pred_list.append(rock_dict[rock_id]["stiffness_matrix"])
        pred_stiff = np.array(pred_list)   # (N, 6, 6), already symmetric from deploy
        truth_stiff = stiffness_symmetrize(Y.reshape(-1, 6, 6))  # (N, 36) -> symmetrize for consistency
        truth_21 = stiffness_upper_triangle_21(truth_stiff.reshape(-1, 6, 6))
        pred_21 = stiffness_upper_triangle_21(pred_stiff)
        truth_21_flat = truth_21.flatten()
        pred_21_flat = pred_21.flatten()
        r2_21 = r2_score(truth_21_flat, pred_21_flat)
        out_path_21 = os.path.join(args.figures_dir, f"cnn_hhm_stiffness_21_parity_{args.rock_type}_large{args.largecube_size}_full{args.fullrock_size}.png")
        ml.figs_ml.plot_stiffness_21_parity(truth_21_flat, pred_21_flat, r2_21, out_path_21)
        print(f'R2 score (21 stiffness constants): {r2_21:.3f}')