# /path/to/digital_rock/rock_images_segmented/B1/size_1200/skin_0/size_600
# /path/to/digital_rock/rock_images_segmented/B1/size_1200/skin_0/size_300
import os, sys, argparse, glob, pickle, torch, numpy as np
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
    parser = argparse.ArgumentParser(description="Evaluate dry-rock moduli: CNN+HHM inference and FFT-based homogenization")
    parser.add_argument("--subcube_stiff_dir", type=str,
                        default="/path/to/digital_rock/FFT_python/real_rock_parameters/multiscale_medium_size/stiffness_files/B1/size_900_300_75/skin_0/",
                        # default="/path/to/digital_rock/FFT_python/real_rock_parameters/multiscale_medium_size/stiffness_files/B1/size_1200_300_75/skin_0/",
                        help="Directory containing subcube stiffness .dat files (e.g. stiffness_1.dat, …, stiffness_64.dat)")
    parser.add_argument("--fullcube_stiff_dir", type=str,
                        default="/path/to/digital_rock/FFT_python/real_rock_parameters/two_phase/runs_geodict/B1/size_900/skin_0/",
                        help="Directory containing true fullcube stiffness .dat files (stiffness_300.dat)")
    parser.add_argument("--scan_dir", type=str,
                        # default="/path/to/digital_rock/machine_learning/bulk_shear_learning/size_75/data/data_B1/skin_0/images",
                        default="/path/to/digital_rock/rock_images_segmented/B1/size_900/skin_0/size_75",
                        help="Directory of size-75 subcube images: Rock_75_1.raw, Rock_75_2.raw, …, Rock_75_4096.raw")
    parser.add_argument("--dat_file", type=str,
                        # default="../../../machine_learning/size_75/data/data_B1/skin_0/stiffness.dat", # just for the CNN model to run
                        default="/path/to/digital_rock/FFT_python/real_rock_parameters/two_phase/runs_geodict/B1/size_900/skin_0/stiffness_75.dat",
                        help=".dat file listing subcube stiffness labels for MLDRP.load_data()")
    # ---------- not used but put as placeholder ------------
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--model_name", type=str, default="cnn3d_rock",)
    parser.add_argument("--benchmark_model", action="store_true", help="Use Benchmark 1 (prior) model for homogenization")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--stop_option", type=str, default='train')
    parser.add_argument("--homogenization", action="store_true", help="Predict effective moduli (2 outputs) from stiffness (36 outputs)")
    parser.add_argument("--hs_bound", action="store_true", help="Predict HS fraction (2 outputs) from stiffness (36 outputs)")
    parser.add_argument("--loss_avg_step", type=int, default=10)
    parser.add_argument("--early_stop", action="store_true", default=False, help="Enable early stopping")
    parser.add_argument("--E1", type=float, help="Young's modulus for pore", default=2.903225806452e-04)
    parser.add_argument("--nu1", type=float, help="Poisson ratio for pore", default=0.4516129032258)
    parser.add_argument("--E2", type=float, help="Young's modulus for mineral", default=95.29411764705883)
    parser.add_argument("--nu2", type=float, help="Poisson ratio for mineral", default=0.058823529411764705)
    parser.add_argument("--model_dir", type=str, default="model_900", help="Directory to save model checkpoints")
    # -------------------------------------------------------
    # parser.add_argument("--model_file", type=str, default=None) # "model/cnn3d_rock_stiffness.pt")
    parser.add_argument("--scan_shape", type=int, nargs=3, default=[75, 75, 75])
    parser.add_argument("--subcubes_per_fullcube", type=int, default=64)
    parser.add_argument("--size_big", type=int, default=300)
    parser.add_argument("--size_smalls", type=int, nargs='+', default=[75], help="List of subcube sizes for FFT homogenization")
    parser.add_argument("--avg_scheme", type=str, default="Hill", help="Average scheme for FFT homogenization (Hill, Voigt, Reuss)")
    parser.add_argument("--mat_type", type=str, default="isotropic",)
    parser.add_argument("--figures_dir", type=str, default="figures", help="Directory to save figures")
    parser.add_argument("--med_stiff_dir", type=str, default="data_hhm", help="storing intermediate stiffness files")
    parser.add_argument("--recompute", action="store_true", help="Ignore cache and recompute errors (then save cache)")
    return parser.parse_args()


def get_cache_path(args):
    """Cache file path from args (model_dir + size_big) so different runs don't share cache."""
    os.makedirs(args.figures_dir, exist_ok=True)
    model_tag = Path(args.model_dir).name or "model"
    return os.path.join(args.figures_dir, f"cnn_hhm_errors_cache_size{args.size_big}_{model_tag}.pkl")


def load_cache(cache_path):
    """Load error_dict and size_big from cache. Returns (error_dict, size_big) or None if missing/invalid."""
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["error_dict"], data["size_big"]
    except Exception:
        return None


def save_cache(cache_path, error_dict, size_big):
    """Save error_dict and size_big to cache."""
    with open(cache_path, "wb") as f:
        pickle.dump({"error_dict": error_dict, "size_big": size_big}, f)
    print(f"---> cache saved: {cache_path}")

class ErrorVisualizer:
    def __init__(self, error_dict, fig_dir, size_big=300):
        self.error_dict = error_dict
        self.fig_dir = fig_dir
        self.size_big = size_big
        self.method_order = ['stiffness', 'moduli', 'hs_bound', 'benchmark_hs_bound']
        self.palette = {'cnn_hhm': 'tab:red',
                        'fft_hhm': 'tab:blue',
                        'cnn_fft': 'tab:green',}
        self.x_labels = {'stiffness': 'Model 1',
                         'moduli': 'Model 2',
                         'hs_bound': 'Model 3',
                         'benchmark_hs_bound': 'Benchmark 1',}
        self.legend_labels = {'cnn_hhm': 'HHM w/ CNN $vs.$ HHM w/ FFT',
                              'fft_hhm': 'HHM w/ FFT $vs.$ DNS w/ FFT',
                              'cnn_fft': 'HHM w/ CNN $vs.$ DNS w/ FFT',}
        self.df = self._build_df()

    def _build_df(self):
        records = []
        for method, etypes in tqdm(self.error_dict.items(), desc='Building records'):
            for err_type, arr in etypes.items():
                for i in range(2):
                    for v in arr[i]:
                        records.append({'Method': method, 'Error Type': err_type,
                            'Column': f'Column {i}', 'Value': v})
        return pd.DataFrame(records)

    def plot_errors(self):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 6))
        for ax, col, ylabel in zip(
            axes, ['Column 0', 'Column 1'],
            [r'Absolute errors for $K$ [GPa]', r'Absolute errors for $G$ [GPa]']):
            sns.swarmplot(data=self.df[self.df['Column'] == col],
                x='Method', y='Value', hue='Error Type',
                order=self.method_order, palette=self.palette, ax=ax)
            ax.set_ylabel(ylabel)
            ax.set_xticklabels(
                [self.x_labels.get(lbl.get_text(), lbl.get_text())
                 for lbl in ax.get_xticklabels()],
                rotation=45, ha='right')
            ax.set_xlabel('')
            ax.set_ylim(-0.3, 6.3)
            if col == 'Column 0':
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, [self.legend_labels.get(l, l) for l in labels], 
                          loc='upper right')
            else:
                ax.legend_.remove()
        # fig.text(0.01, 0.98, '(a)', fontsize=20, fontweight='bold', va='top', ha='left')
        plt.tight_layout()
        save_path = os.path.join(self.fig_dir, f'cnn_hhm_errors_size{self.size_big}.png')
        plt.savefig(save_path, dpi=300)
        print(f"---> saved figure: {save_path}")

    def plot_errors_boxplot(self):
        """Box plot: for each method (label), 3 boxes for the three error types (cnn_hhm, fft_hhm, cnn_fft)."""
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.5))
        hue_order = ['cnn_hhm', 'fft_hhm', 'cnn_fft']
        for ax, col, ylabel in zip(
            axes, ['Column 0', 'Column 1'],
            [r'Absolute errors for $K$ [GPa]', r'Absolute errors for $G$ [GPa]']):
            sns.boxplot(data=self.df[self.df['Column'] == col],
                x='Method', y='Value', hue='Error Type',
                order=self.method_order, hue_order=hue_order, palette=self.palette, ax=ax)
            ax.set_ylabel(ylabel)
            ax.set_xticklabels(
                [self.x_labels.get(lbl.get_text(), lbl.get_text())
                 for lbl in ax.get_xticklabels()],
                rotation=45, ha='right')
            ax.set_xlabel('')
            ax.set_ylim(-0.3, 6.3)
            if col == 'Column 0':
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, ['I', 'II', 'III'], loc='upper left', frameon=True)
            else:
                ax.legend_.remove()
        plt.tight_layout()
        save_path = os.path.join(self.fig_dir, f'cnn_hhm_errors_size{self.size_big}_boxplot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"---> saved figure: {save_path}")

    def print_error_table(self):
        comparison_map = {"CNN vs. HHM": 'cnn_hhm',
                          "HHM vs. DNS": 'fft_hhm',
                          "CNN vs. DNS": 'cnn_fft',}
        rows = []
        for label, key in comparison_map.items():
            Ek = [np.mean(self.error_dict[m][key][0]) for m in self.method_order]
            Eg = [np.mean(self.error_dict[m][key][1]) for m in self.method_order]
            rows += [[label] + [f"{v:.3f}" for v in Ek],
                     [r"$|\mathcal{E}_K|$ [GPa]"] + [""] * len(self.method_order),
                     [""] + [f"{v:.3f}" for v in Eg],
                     [r"$|\mathcal{E}_G|$ [GPa]"] + [""] * len(self.method_order),]

        headers = ["Abs. err."] + [self.x_labels[m] for m in self.method_order]
        print(tabulate(rows, headers=headers, tablefmt='grid'))


if __name__ == "__main__":
    args = parse_args()
    cache_path = get_cache_path(args)

    error_dict = None
    size_big = args.size_big
    if not args.recompute:
        loaded = load_cache(cache_path)
        if loaded is not None:
            error_dict, size_big = loaded
            print(f"---> loaded from cache: {cache_path} (plotting only)")

    if error_dict is None:
        # Full calculation
        model_list = glob.glob(os.path.join(args.model_dir, f"{args.model_name}_*.pt"))
        error_dict = {}
        for model_file in model_list:
            # Set model type from current filename so checkpoint and architecture match
            if 'hs_bound' in model_file:
                args.hs_bound = True
                args.homogenization = False
                args.benchmark_model = 'benchmark' in model_file
            elif 'moduli' in model_file:
                args.homogenization = True
                args.hs_bound = False
                args.benchmark_model = False
            else:
                args.homogenization = False
                args.hs_bound = False
                args.benchmark_model = False
            ml = MLDRP(args)
            ml.load_data()
            device = ml.device
            ckpt = torch.load(model_file, map_location=device)
            ml.model.load_state_dict(ckpt)
            ml.model.to(device).eval()
            print(f"* {colors.BLUE}Loaded model from{colors.RESET} {model_file}: {ml.model}")

            # 2. Predict subcube stiffness/moduli
            os.makedirs(args.med_stiff_dir, exist_ok=True)
            # prefix = os.path.join(args.model_dir, "size_75")
            ml.deploy_cnn_stiffness_pred(output_file=f'{args.med_stiff_dir}/cnn_pred',)
            # reorder the stiffness matrices in the stiffness.dat file
            if ml.model.mode == 'stiffness':
                cnn_stiffness_file = f'{args.med_stiff_dir}/cnn_pred_stiffness.dat'
            elif ml.model.mode == 'moduli':
                cnn_stiffness_file = f'{args.med_stiff_dir}/cnn_pred_moduli.dat'
            elif ml.model.mode == 'hs_bound':
                if args.benchmark_model:
                    cnn_stiffness_file = f'{args.med_stiff_dir}/cnn_pred_benchmark_hs_bound_stiffness.dat'
                else:
                    cnn_stiffness_file = f'{args.med_stiff_dir}/cnn_pred_hs_bound_stiffness.dat'
            rock_matrices = read_rock_matrices(cnn_stiffness_file, apply_increment=False)
            write_matrices_to_dat(rock_matrices, cnn_stiffness_file)

            # (1) Using HHM to predict the moduli of size-300 "full"-cubes based on size-75 subcubes
            split_stiffness_file(input_path=cnn_stiffness_file,
                                 output_dir=f'{args.med_stiff_dir}/CNN_small',
                                 chunk_size=64)
            bulk_large_cnn, shear_large_cnn = [], []
            for i in range(1, 28):
                stiff_file_75 = os.path.join(f'{args.med_stiff_dir}/CNN_small', f"stiffness_75_{i}.dat")
                bulk_large, shear_large = homogenize_fullcube_stiffness(Path(stiff_file_75), rock_id=i,
                                                                    data_dir=f'{args.med_stiff_dir}/CNN_75_homog',
                                                                    mat_type=args.mat_type,
                                                                    avg_scheme=args.avg_scheme)
                bulk_large_cnn.append(bulk_large); shear_large_cnn.append(shear_large)
            bulk_large_cnn, shear_large_cnn = np.array(bulk_large_cnn).flatten(), np.array(shear_large_cnn).flatten()
            # shear_large_cnn = np.array(shear_large_cnn).flatten()

            # (2) Load the FFT-calculated size-75 subcubes and homogenize their moduli for comparison
            split_stiffness_file(input_path=args.dat_file,
                                 output_dir=f'{args.med_stiff_dir}/FFT_small',
                                 chunk_size=64)
            bulk_large_75_fft, shear_large_75_fft = [], []
            for i in range(1, 28):
                stiff_file_75 = os.path.join(f'{args.med_stiff_dir}/FFT_small', f"stiffness_75_{i}.dat")
                # stiff_file_75 = os.path.join('data_explore_hhm/size_900_large_75', f"stiffness_{i}.dat")
                bulk_large, shear_large = homogenize_fullcube_stiffness(Path(stiff_file_75), rock_id=i,
                                                                        data_dir=f'{args.med_stiff_dir}/FFT_75_homog',
                                                                        mat_type=args.mat_type,
                                                                        avg_scheme=args.avg_scheme)
                bulk_large_75_fft.append(bulk_large); shear_large_75_fft.append(shear_large)
            bulk_large_hhm, shear_large_hhm = np.array(bulk_large_75_fft).flatten(), np.array(shear_large_75_fft).flatten()
            # shear_large_hhm = np.array(shear_large_75_fft).flatten()

            # (3) Load the FFT-calculated size-300 fullcube stiffness and homogenize their moduli
            fullcube_stiff_file = os.path.join(args.fullcube_stiff_dir, f"stiffness_{args.size_big}.dat")
            split_stiffness_file(input_path=fullcube_stiff_file,
                                 output_dir=f'{args.med_stiff_dir}/FFT_big',
                                 chunk_size=1, target_size=300)
            bulk_large_fft, shear_large_fft = [], []
            for i in range(1, 28):
                stiff_file_large = os.path.join(f'{args.med_stiff_dir}/FFT_big', f"stiffness_{args.size_big}_{i}.dat")
                bulk_large, shear_large = homogenize_fullcube_stiffness(Path(stiff_file_large), rock_id=i,
                                                                    data_dir=f'{args.med_stiff_dir}/FFT_300_homg',
                                                                    mat_type=args.mat_type,
                                                                    avg_scheme=args.avg_scheme)
                bulk_large_fft.append(bulk_large); shear_large_fft.append(shear_large)
            bulk_large_fft, shear_large_fft = np.array(bulk_large_fft).flatten(), np.array(shear_large_fft).flatten()
            # shear_large_fft = np.array(shear_large_fft).flatten()

            for file in glob.glob("*.dat"):
                os.remove(file)
            error_cnn_hhm = np.array([np.abs(bulk_large_cnn - bulk_large_hhm), np.abs(shear_large_cnn - shear_large_hhm)])
            error_fft_hhm = np.array([np.abs(bulk_large_fft - bulk_large_hhm), np.abs(shear_large_fft - shear_large_hhm)])
            error_cnn_fft = np.array([np.abs(bulk_large_cnn - bulk_large_fft), np.abs(shear_large_cnn - shear_large_fft)])

            base_name = os.path.splitext(os.path.basename(model_file))[0]
            key = base_name[len('cnn3d_rock_'):]
            error_dict[key] = {'cnn_hhm': error_cnn_hhm, 'fft_hhm': error_fft_hhm, 'cnn_fft': error_cnn_fft,}

        save_cache(cache_path, error_dict, args.size_big)
        size_big = args.size_big

    viz = ErrorVisualizer(error_dict, args.figures_dir, size_big)
    viz.plot_errors()
    viz.plot_errors_boxplot()
    viz.print_error_table()