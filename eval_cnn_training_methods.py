import re, argparse, numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from util_visualization import plotting_params
plt.rcParams.update(plotting_params)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CNN training times and R² values")
    parser.add_argument("-fig_dir", "--fig_dir", default="figures/", type=str, help="Directory to save figures")
    parser.add_argument("-dat_file", "--dat_file", default="DRPCNNElasMod_all.out", type=str, help="Path to the .out file")
    return parser.parse_args()

def parse_out_file(path):
    with open(path, 'r') as f:
        text = f.read()

    training = [int(t) for t in re.findall(r"Training finished in (\d+)s", text)]
    bulk_r2  = [float(r) for r in re.findall(r"R2 for Bulk modulus:\s*([0-9.]+)", text)]
    shear_r2 = [float(r) for r in re.findall(r"R2 for Shear modulus:\s*([0-9.]+)", text)]
    return training, bulk_r2, shear_r2

def plot_cnn_eval_bar(methods, values, ylabel, filename, yscale=None, y_lim=None, option='R2'):
    if option == 'Time':
        plt.figure(figsize=(5, 6))
        bars = plt.bar(methods, values, color='blue',
                       edgecolor='black', linewidth=1, width=0.35)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2,
                     height, f"{val:.0f}",
                     ha='center', va='bottom', fontsize=15)

        plt.ylabel(ylabel)
        if y_lim is not None: plt.ylim(y_lim)
        if yscale: plt.yscale(yscale)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    elif option == 'R2':
        r2_bulk, r2_shear = values[0], values[1]
        fig, ax = plt.subplots(figsize=(5,6))
        ind   = np.arange(len(methods))
        width = 0.25
        ax.bar(ind - width/2, r2_bulk,  width,
            color='darkblue',  edgecolor='black', linewidth=1,
            label=r'$R^2$ for $K$')
        ax.bar(ind + width/2, r2_shear, width,
            color='darkred',   edgecolor='black', linewidth=1,
            label=r'$R^2$ for $G$')
        ax.set_ylabel(r'$R^2$')
        ax.set_ylim(0.90, 1.00)
        ax.set_xticks(ind)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    print(f"---> saved figure {filename}")

def plot_cnn_moduli_comparison(npz_files, methods, filename):
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    for i, (npz_path, method) in enumerate(zip(npz_files, methods)):
        data = np.load(npz_path)
        axK, axG = axes[0, i], axes[1, i]

        # Bulk modulus
        axK.plot(data['bulk_hs_lower_x'], data['bulk_hs_lower_y'], color='red', linestyle='-', label='HS bounds')
        axK.plot(data['bulk_hs_upper_x'], data['bulk_hs_upper_y'], color='red', linestyle='-')
        axK.scatter(data['bulk_truth'], data['bulk_pred'], color='darkblue', s=10, 
                    label=rf'$R^2={r2_score(data["bulk_truth"], data["bulk_pred"]):.3f}$')
        axK.plot([0, 40], [0, 40], '-.', color='C1', lw=2)
        axK.set_title(method, fontsize=15)
        if i == 0: axK.set_ylabel(r'$K_{\rm eff}^{\rm CNN}$ [GPa]')
        axK.set_xlabel(r'$K_{\rm eff}^{\rm DNS}$ [GPa]')

        # Shear modulus
        axG.plot(data['shear_hs_lower_x'], data['shear_hs_lower_y'], color='red', linestyle='-', label='HS bounds')
        axG.plot(data['shear_hs_upper_x'], data['shear_hs_upper_y'], color='red', linestyle='-')
        axG.scatter(data['shear_truth'], data['shear_pred'], color='darkblue', s=10, 
                    label=rf'$R^2={r2_score(data["shear_truth"], data["shear_pred"]):.3f}$')
        axG.plot([0, 50], [0, 50], '-.', color='C1', lw=2)
        if i == 0: axG.set_ylabel(r'$G_{\rm eff}^{\rm CNN}$ [GPa]')
        axG.set_xlabel(r'$G_{\rm eff}^{\rm DNS}$ [GPa]')
        axK.legend(loc='upper left'); axG.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"---> saved figure {filename}")


if __name__ == "__main__":
    args = parse_args()
    methods = ["Model 1", "Model 2", "Model 3",
               "Benchmark 1", "Benchmark 2"]
    npz_files = [f"{args.fig_dir}/cnn_moduli_scatter_stiffness.npz",
                 f"{args.fig_dir}/cnn_moduli_scatter_moduli.npz",
                 f"{args.fig_dir}/cnn_moduli_scatter_hs_bound.npz",
                 f"{args.fig_dir}/cnn_moduli_scatter_benchmark_hs_bound.npz"]
    moduli_labels = ["Model 1", "Model 2", "Model 3", "Benchmark 1"]

    training_times, r2_bulk, r2_shear = parse_out_file(args.dat_file)

    # ─── manually add Benchmark 1 results ───
    training_times.append(14307)  # Training completed in 14307 seconds
    r2_bulk.append(0.976)
    r2_shear.append(0.979)
    # ────────────────────────────────────
    plot_cnn_eval_bar(methods, training_times, ylabel="Training Time [s]",
                      filename=f"{args.fig_dir}/eval_cnn_training_times.png",
                      yscale="log", option='Time')
    plot_cnn_eval_bar(methods, [r2_bulk, r2_shear], ylabel=r"$R^2$ for Moduli",
                      filename=f"{args.fig_dir}/eval_cnn_r2.png",
                      y_lim=(0.9, 1.0), option='R2')
    plot_cnn_moduli_comparison(npz_files, moduli_labels,
                               filename=f"{args.fig_dir}/eval_cnn_moduli_comparison.png")