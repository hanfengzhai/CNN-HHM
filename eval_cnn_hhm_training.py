import os, argparse, torch, random, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
from util_visualization import plotting_params
plt.rcParams.update(plotting_params)
seed = 2025
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate training method for CNN")
    parser.add_argument("--figures_dir", type=str, default="figures", help="Directory to save figures")
    parser.add_argument("--result_dir",  type=str, default="data", help="Directory to load results")
    parser.add_argument("--rock_type",   type=str, default="B1", choices=["B1", "B2", "CG", "FB1", "FB2"], help="Rock type")
    parser.add_argument("--fine_tune",   action="store_true", help="Use fine-tuning")
    parser.add_argument("--epochs",      type=int, default=50, help="Number of training epochs")
    return parser.parse_args()

def load_results(result_dir, key, size, rock_type="B1", fine_tune=False):
    if fine_tune:
        filename = f"fine_tune_{rock_type}/cnn_hhm_moduli_{rock_type}_{key}_large{size}_full1200.npy"
    else:
        filename = f"cnn_hhm_moduli_{rock_type}_{key}_large{size}_full1200.npy"
    return np.load(os.path.join(result_dir, filename), allow_pickle=True)

def plot_hhm(size, result_dir, figures_dir, rock_type="B1", fine_tune=False):
    keys = ["stiffness", "moduli", "hs_bound", "benchmark_hs_bound",]
    titles = ["Model 1", "Model 2", "Model 3", "Benchmark 1"]

    bulk_pairs, shear_pairs = [], []
    for key, title in zip(keys, titles):
        bx, by, sx, sy = load_results(result_dir, key, size, rock_type, fine_tune=fine_tune)
        bulk_pairs.append((bx, by, title))
        shear_pairs.append((sx, sy, title))

    def get_limits(pairs):
        x = np.concatenate([p[0] for p in pairs])
        y = np.concatenate([p[1] for p in pairs])
        return x.min() * 0.85, x.max() * 1.15, y.min() * 0.85, y.max() * 1.15

    bx0, bx1, by0, by1 = get_limits(bulk_pairs)
    sx0, sx1, sy0, sy1 = get_limits(shear_pairs)

    bulk_min, bulk_max   = min(bx0, by0), max(bx1, by1)
    shear_min, shear_max = min(sx0, sy0), max(sx1, sy1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 7.5))
    for col in range(4):
        ax = axes[0, col]
        x, y, title = bulk_pairs[col]
        r2 = r2_score(x, y)
        rmse = root_mean_squared_error(x, y)
        label = f"$R^2={r2:.3f}$\n\n$\\mathrm{{RMSE}}={rmse:.2f}\\ \mathrm{{GPa}}$"
        ax.plot(x, y, 'o', label=label, color='darkblue')
        ax.plot([bx0, bx1], [bx0, bx1], 'r-.')
        ax.set_title(title, fontsize=15)
        if col: ax.tick_params(labelleft=False)
        else:   ax.set_ylabel(r"$K_{\rm eff}^{\rm CNN}$ [GPa]")
        ax.set_xlabel(r"$K_{\rm eff}^{\rm DNS}$ [GPa]")
        
        ax.set_xlim(bulk_min, bulk_max)
        ax.set_ylim(bulk_min, bulk_max)
        ax.legend(loc='upper left', fontsize=12)
        ax.set_aspect('equal', adjustable='box')

        ax = axes[1, col]
        x, y, title = shear_pairs[col]
        r2 = r2_score(x, y)
        rmse = root_mean_squared_error(x, y)
        label = f"$R^2={r2:.3f}$\n\n$\\mathrm{{RMSE}}={rmse:.2f}\\ \mathrm{{GPa}}$"
        ax.plot(x, y, 'o', label=label, color='darkblue')
        ax.plot([sx0, sx1], [sx0, sx1], 'r-.')
        if col: ax.tick_params(labelleft=False)
        else:   ax.set_ylabel(r"$G_{\rm eff}^{\rm CNN}$ [GPa]")
        ax.set_xlabel(r"$G_{\rm eff}^{\rm DNS}$ [GPa]")

        ax.set_xlim(shear_min, shear_max)
        ax.set_ylim(shear_min, shear_max)
        ax.legend(loc='upper left', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    out = os.path.join(figures_dir, f"cnn_hhm_{rock_type}_large{size}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    tqdm.write(f"---> saved figure: {out}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    if args.fine_tune:
        for size in tqdm([300, 600], desc="(Fine-tune) HHM rock sizes"):
            plot_hhm(size, args.result_dir, f'{args.figures_dir}/fine_tune_{args.rock_type}',
                     rock_type=args.rock_type, fine_tune=args.fine_tune)
    else:
        for size in tqdm([300, 600], desc="HHM rock sizes"):
            plot_hhm(size, args.result_dir, args.figures_dir,
                     rock_type=args.rock_type, fine_tune=args.fine_tune)