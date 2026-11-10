import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import NullFormatter
from pathlib import Path

global plotting_params
plotting_params = {"font.family": "serif", "font.serif": ["Libertinus Serif"],
                   "mathtext.fontset": "stix", # custom
                   "xtick.labelsize": 15, "ytick.labelsize": 15,
                   "axes.labelsize": 15, "legend.fontsize": 15}
plt.rcParams.update(plotting_params)

class TerminalColors:
    """A class to add colors to terminal output."""

    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'

    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'

    @staticmethod
    def colored(text, *colors):
        """Applies the given colors to the text."""
        color_codes = "".join(colors)
        return f"{color_codes}{text}{TerminalColors.RESET}"

class figures_gassmann_eval:
    """
    Class for plotting Gassmann's equation evaluation figures.
    """
    def __init__(self, fig_dir: str):
        # self.fig_dir = fig_dir
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        from util_elasticity import gassmann_bulk
        self.gassmann_bulk = gassmann_bulk

    def plot_rel_errors(self, results: dict, size_big: int, avg_scheme: str, params: dict,
                        figure_name=None, x_ticks=[100, 200, 400, 800], marker_pos=(-0.2, -0.1)):
        """
            Plot and save relative error figures for bulk and shear moduli.
        """
        if size_big == 400:
            x_ticks = [100, 200, 400]
        xs = results['size_smalls']
        K_err = [abs(self.gassmann_bulk(K_dry, K_min, porosity, K_wat) - K_sat) / K_sat
                for K_dry, K_sat, K_min, porosity, K_wat in zip(
                    results['K_dry'], results['K_sat'],
                    [params['K_min']]*len(xs), [params['phi']]*len(xs), [params['K_wat']]*len(xs))]
        G_err = [abs(G_dry - G_sat) / G_sat
                for G_dry, G_sat in zip(results['G_dry'], results['G_sat'])]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(xs, K_err, marker='o', linestyle='-.', color='black')
        ax1.set_xlabel(r'$n$')
        ax1.set_ylabel(r'Relative error for $K$ [-]')
        ax1.set_yscale('log'); ax1.set_xscale('log')
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(map(str, x_ticks))
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.text(*marker_pos, '(a)', transform=ax1.transAxes,
                fontsize=17, weight='semibold')

        ax2.plot(xs, G_err, marker='o', linestyle='-.', color='black')
        ax2.set_xlabel(r'$n$')
        ax2.set_ylabel(r'Relative error for $G$ [-]')
        ax2.set_yscale('log'); ax2.set_xscale('log')
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(map(str, x_ticks))
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.text(*marker_pos, '(b)', transform=ax2.transAxes,
                fontsize=17, weight='semibold')
        plt.tight_layout()
        if figure_name is None:
            figure_name = f'{self.fig_dir}/gassmann_rel_err_{avg_scheme}_{size_big}.png'
        plt.savefig(figure_name, dpi=300)
        print(f'---> save plot to {figure_name}')
        plt.close()


    def plot_moduli_comparison(self, results: dict, size_big: int, avg_scheme: str, params: dict,
                               method=None, figure_name=None,
                               x_ticks=[100, 200, 400, 800], marker_pos=(-0.2, -0.1)):
        """
        Plot and save bulk and shear moduli comparison with inset axes.
        """
        if size_big == 400:
            x_ticks = [100, 200, 400]
        xs = results['size_smalls']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(xs, results['K_sat'], marker='o', markeredgecolor='black', markersize=7.5,
                lw=2, color='darkblue', label='Saturated (HHM + FFT)')
        ax1.plot(xs, [self.gassmann_bulk(K_dry, params['K_min'], params['phi'], params['K_wat'])
                    for K_dry in results['K_dry']], marker='o', markeredgecolor='black', markersize=7.5,
                lw=2, color='darkred', label='Saturated (Gassmann Eq.)')
        ax1.set_xlabel(r'$n$')
        ax1.set_ylabel(r'$K$ [GPa]')
        ax1.set_yscale('log'); ax1.set_xscale('log')
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(map(str, x_ticks))
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.text(*marker_pos, '(a)', transform=ax1.transAxes,
                fontsize=17, weight='semibold')
        ax1.legend(loc='lower left')

        ax2.plot(xs, results['G_sat'], marker='o', markeredgecolor='black', markersize=7.5,
                lw=2, color='darkblue', label='Saturated (HHM + FFT)')
        ax2.plot(xs, results['G_dry'], marker='o', markeredgecolor='black', markersize=7.5,
                lw=2, color='darkred', label='Saturated (Gassmann Eq.)')
        ax2.set_xlabel(r'$n$')
        ax2.set_ylabel(r'$G$ [GPa]')
        ax2.set_yscale('log'); ax2.set_xscale('log')
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(map(str, x_ticks))
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.text(*marker_pos, '(b)', transform=ax2.transAxes,
                fontsize=17, weight='semibold')
        ax2.legend(loc='lower left')

        # Insets
        for ax, M_sat, M_dry in zip([ax1, ax2], [results['K_pred'], results['G_pred']], [results['K_dry'], results['G_dry']]):
            ax_ins = inset_axes(ax, width="50%", height="50%",
                                bbox_to_anchor=(0.35, 0.35, 0.63, 0.63),
                                bbox_transform=ax.transAxes)
            ax_ins.plot(xs, M_sat, marker='o', lw=2, markeredgecolor='black', markersize=7.5,
                        color='darkred')
            ax_ins.plot(xs, M_dry, marker='*', ls='--', lw=1, color='red', label='Dry')
            ax_ins.set_yscale('log'); ax_ins.set_xscale('log')
            ax_ins.set_xticks(x_ticks)
            ax_ins.set_xticklabels(map(str, x_ticks))
            ax_ins.xaxis.set_minor_formatter(NullFormatter())
            ax_ins.legend(frameon=False) #fontsize=12, 
        plt.tight_layout()

        if figure_name is None:
            if method is None: method = 'OT'
            figure_name = f'{self.fig_dir}/gassmann_mod_comparison_{method}_{avg_scheme}_{size_big}.png'
        plt.savefig(figure_name, dpi=300)
        print(f'---> save plot to {figure_name}')
        plt.close()


    def plot_combined_rel_err(self, all_results: dict, 
                            figure_name=None, 
                            x_ticks=[100, 200, 400, 800],
                            marker_pos=(-0.2, -0.1)):
        """
        Plot and save combined relative error and moduli comparisons across rock sizes.
        """
        if figure_name is None:
            fig_name1 = f'{self.fig_dir}/gassmann_all_rel_err.png'
            fig_name2 = f'{self.fig_dir}/gassmann_all_moduli_comparison.png'
            figure_name = [fig_name1, fig_name2]

        # ----------------------------- Relative error comparison -----------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        for size_big, style, clr in zip([400, 800], ['-', '--'], ['darkblue', 'darkred']):
            res = all_results[size_big]
            ax1.plot(res['size_smalls'], res['K_err'], marker='o', markeredgecolor='black', markersize=7.5,
                    linestyle=style, color=clr, label=rf'Gassmann vs. FFT: $\mathbb{{R}}^{{{size_big}}}$')
            ax2.plot(res['size_smalls'], res['G_err'], marker='s', markeredgecolor='black', markersize=7.5,
                    linestyle=style, color=clr, label=rf'Gassmann vs. FFT: $\mathbb{{R}}^{{{size_big}}}$')

        ax1.set_xlabel(r'$n$')
        ax1.set_ylabel(r'Relative error for $K$ [-]')
        ax1.set_yscale('log'); ax1.set_xscale('log')
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(map(str, x_ticks))
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.text(*marker_pos, '(a)', transform=ax1.transAxes,
                fontsize=17, weight='semibold')
        ax1.legend()

        ax2.set_xlabel(r'$n$')
        ax2.set_ylabel(r'Relative error for $G$ [-]')
        ax2.set_yscale('log'); ax2.set_xscale('log')
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(map(str, x_ticks))
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.text(*marker_pos, '(b)', transform=ax2.transAxes,
                fontsize=17, weight='semibold')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(figure_name[0], dpi=300)
        print(f'---> save plot to {figure_name[0]}')
        plt.close()

        # ----------------------------- Combined moduli comparison -----------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        for size_big, style, clr_fft, clr_gasmn in zip([400, 800], ['-', '--'], ['darkblue', 'blue'], ['darkred', 'red']):
            res = all_results[size_big]
            lw=2 if style=='-' else 1
            ax1.plot(res['size_smalls'], res['K_sat'], marker='o', markeredgecolor='black', markersize=7.5,
                    linestyle=style, color=clr_fft, lw=lw, label=rf'FFT + HHM ($\mathbb{{R}}^{{{size_big}}}$)')
            ax1.plot(res['size_smalls'], res['K_pred'], marker='s', markeredgecolor='black', markersize=7.5,
                    linestyle=style, color=clr_gasmn, lw=lw, label=rf'Gassmann ($\mathbb{{R}}^{{{size_big}}}$)')
            ax2.plot(res['size_smalls'], res['G_sat'], marker='o', markeredgecolor='black', markersize=7.5,
                    linestyle=style, color=clr_fft, lw=lw, label=rf'FFT + HHM ($\mathbb{{R}}^{{{size_big}}}$)')
            ax2.plot(res['size_smalls'], res['G_pred'], marker='s', markeredgecolor='black', markersize=7.5,
                    linestyle=style, color=clr_gasmn, lw=lw, label=rf'Gassmann ($\mathbb{{R}}^{{{size_big}}}$)')

        ax1.set_xlabel(r'$n$')
        ax1.set_ylabel(r'$K$ [GPa]')
        ax1.set_yscale('log'); ax1.set_xscale('log')
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(map(str, x_ticks))
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.text(*marker_pos, '(a)', transform=ax1.transAxes,
                fontsize=17, weight='semibold')
        ax1.legend()

        ax2.set_xlabel(r'$n$')
        ax2.set_ylabel(r'$G$ [GPa]')
        ax2.set_yscale('log'); ax2.set_xscale('log')
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(map(str, x_ticks))
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.text(*marker_pos, '(b)', transform=ax2.transAxes,
                fontsize=17, weight='semibold')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(figure_name[1], dpi=300)
        print(f'---> save plot to {figure_name[1]}')
        plt.close()


class figures_hhm_moduli_eval:
    """
    Class for plotting HHM moduli evaluation figures.
    """
    def __init__(self, fig_dir: str):
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def fit_inverse_model_hhm(self, x, y):
        """
        Fit y = M + C / x via least-squares on (1/x, y).
        Returns intercept M and slope C.
        """
        inv = 1.0 / np.array(x)
        C, M = np.polyfit(inv, y, deg=1)
        return M, C

    def plot_hhm_fit_ax(self, ax,
                    x1, y1, x2, y2, ylabel,
                    eqn_pos=(0.05, 0.05),
                    marker_pos=(-0.2, -0.1),
                    labels=(r'$\mathbb{{R}}^{{400}}$', r'$\mathbb{{R}}^{{800}}$'),
                    fit_labels=('400 fit', '800 fit'),
                    marker_label='(a)'):
        """
        Plot two series (x1,y1) and (x2,y2) on `ax`, fit each to M + C/n,
        and overlay both raw points and fitted curves on log–log axes.
        """
        # 1) Fit both
        M1, C1 = self.fit_inverse_model_hhm(x1, y1)
        M2, C2 = self.fit_inverse_model_hhm(x2, y2)

        # 2) Smooth x for drawing the fit curves
        lo = min(min(x1), min(x2)) * 0.9
        hi = max(max(x1), max(x2)) * 1.1
        x_fit = np.linspace(lo, hi, 200)
        fit1 = M1 + C1 / x_fit
        fit2 = M2 + C2 / x_fit

        # 3) Plot raw data
        ax.plot(x1, y1, 'o-', markeredgecolor='black', markersize=7.5,
                label=labels[0], lw=1.5, color='blue')
        ax.plot(x2, y2, 's-', markeredgecolor='black', markersize=7.5,
                label=labels[1], lw=1.5, color='red')

        # 4) Plot fits
        ax.plot(x_fit, fit1, '-.', color='darkblue',
                label=rf'$M_\infty^{{\rm LR}}={M1:.2f}$, $C^{{\rm LR}}={C1:.2f}$', alpha=0.7)
        ax.plot(x_fit, fit2, '-.', color='darkred',
                label=rf'$M_\infty^{{\rm SR}}={M2:.2f}$, $C^{{\rm SR}}={C2:.2f}$', alpha=0.7)

        # 5) Axes styling
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(sorted(set(x1) | set(x2)))
        ax.set_xticklabels([str(int(xi)) for xi in sorted(set(x1) | set(x2))])
        ax.set_xlabel(r'$n$')
        ax.set_ylabel(ylabel)

        # 6) Equation + marker
        ax.text(*eqn_pos, r'$G \leftarrow M_\infty + C n^{-1}$', transform=ax.transAxes, fontsize=17)
        ax.text(*marker_pos, marker_label, transform=ax.transAxes,
                fontsize=17, weight='semibold')
        ax.legend(loc='upper right')

    def intersect_series(self, R1, R2, key):
        x1, y1 = all_results[R1]['size_smalls'], all_results[R1][key]
        x2, y2 = all_results[R2]['size_smalls'], all_results[R2][key]
        common_x = sorted(set(x1) & set(x2))
        y1c = [ y1[x1.index(xi)] for xi in common_x ]
        y2c = [ y2[x2.index(xi)] for xi in common_x ]
        return common_x, y1c, y2c


class figures_machine_learning:

    def __init__(self, fig_dir: str):
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def plot_pred_w_uncertainty(self, x_train, y_train, x_test, y_test, 
        r2_train, r2_test, n_bins=25, out_path="{self.fig_dir}/pred_vs_benchm.png"):
        """
        x_train, y_train: 1D arrays of benchmark & prediction for training set
        x_test,  y_test : 1D arrays for testing set
        r2_train, r2_test: R² values for training & testing
        n_bins : number of bins to use for the confidence band
        out_path : path to save the figure
        """

        plt.figure(figsize=(5, 5))
        # 45° line
        overall_min = min(x_train.min(), y_train.min(), x_test.min(), y_test.min()) - 0.75 * x_train.mean()
        overall_max = max(x_train.max(), y_train.max(), x_test.max(), y_test.max()) + 0.75 * x_train.mean()
        plt.plot([overall_min, overall_max], [overall_min, overall_max], 'k--', linewidth=2, alpha=0.5)

        plt.scatter(x_train, y_train, s=10, alpha=0.5, color='blue', label=rf"Training set: $R^2 = {r2_train:.3f}$")
        plt.scatter(x_test, y_test, s=10, alpha=0.5, color='red', label=rf"Testing set: $R^2 = {r2_test:.3f}$")

        train_xc, train_mean, train_std = self.compute_binned_stats(x_train, y_train, n_bins)
        test_xc, test_mean, test_std = self.compute_binned_stats(x_test, y_test, n_bins)

        ci_factor = 1.96 # 95% CI band: mean ± 1.96 * std

        plt.fill_between(train_xc, # Plot training band
                train_mean - ci_factor * train_std,
                train_mean + ci_factor * train_std,
                color='blue', alpha=0.15)

        plt.fill_between(test_xc, # Plot testing band
                test_mean - ci_factor * test_std,
                test_mean + ci_factor * test_std,
                color='red', alpha=0.15) # label="Test 95% CI"

        plt.xlabel("Benchmark moduli [GPa]")#, fontsize=15)
        plt.ylabel("Predicted moduli [GPa]")#, fontsize=15)
        plt.xlim(overall_min, overall_max)
        plt.ylim(overall_min, overall_max)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        print(f"---> save figure as {out_path}")
        plt.close()

    def compute_binned_stats(self, x, y, nbins): # used to compute the confidence band for figure
        bins = np.linspace(x.min(), x.max(), nbins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        mean_vals = np.zeros(nbins)
        std_vals  = np.zeros(nbins)

        for i in range(nbins):
            in_bin = (x >= bins[i]) & (x < bins[i+1])
            if not np.any(in_bin): # If no data in this bin, set mean & std to NaN
                mean_vals[i] = np.nan
                std_vals[i] = np.nan
            else:
                mean_vals[i] = np.mean(y[in_bin])
                std_vals[i]  = np.std(y[in_bin])
        return bin_centers, mean_vals, std_vals

    def plot_stiffness_21_parity(self, truth_21_flat, pred_21_flat, r2, out_path,
                                xlabel=r"$C_{ij}^{\rm DNS}$ [GPa]", ylabel=r"$C_{ij}^{\rm CNN}$ [GPa]"):
        """
        Parity plot for the 21 independent stiffness constants (ground truth vs prediction).
        truth_21_flat, pred_21_flat: 1D arrays of length N*21.
        r2: R² score. out_path: path to save the figure.
        """
        plt.figure(figsize=(5.5, 5.5))
        lb = min(truth_21_flat.min(), pred_21_flat.min())
        ub = max(truth_21_flat.max(), pred_21_flat.max())
        margin = (ub - lb) * 0.05 if ub > lb else 1.0
        lb, ub = lb - margin, ub + margin
        plt.plot([lb, ub], [lb, ub], 'r--', lw=2, alpha=0.7)
        plt.scatter(truth_21_flat, pred_21_flat, s=10, alpha=0.1, c='darkblue', label=rf'$R^2 = {r2:.3f}$')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(lb, ub)
        plt.ylim(lb, ub)
        plt.legend(loc='upper left')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"---> saved figure: {out_path}")
        plt.close()

    def plot_loss_curve(self, train_loss, test_loss, fig_key=None):
        plt.figure(figsize=(5, 5))
        plt.plot(train_loss, linewidth=6, color='black', alpha=0.5)
        plt.plot(train_loss, linewidth=2, color='blue', label="Train Loss")
        plt.plot(test_loss, linewidth=6, color='black', alpha=0.5)
        plt.plot(test_loss, linewidth=2, color='red', label="Test Loss")
        plt.xlabel("Epochs"); plt.ylabel("Loss")
        plt.yscale("log")

        plt.legend()
        plt.tight_layout()
        fig_key = 'stiffness' if fig_key is None else fig_key
        plt.savefig(f"{self.fig_dir}/loss_{fig_key}.png", dpi=300)
        print(f"---> save figure as {self.fig_dir}/loss_{fig_key}.png")
        plt.close()

    def plot_eval_pred(self, all_truth_flat, all_preds_flat, r2, full_rock=400): 
        # !!! archived function: currently this function replaced by the w/ uncertainty plot
        import scipy.stats as stats
        plt.figure(figsize=(5.5, 5))

        overall_min = min(all_truth_flat.min(), all_preds_flat.min()) - 0.75 * all_truth_flat.mean()
        overall_max = max(all_truth_flat.max(), all_preds_flat.max()) + 0.75 * all_truth_flat.mean()
        plt.plot([overall_min, overall_max], [overall_min, overall_max], 'k-.', linewidth=2, alpha=0.5)

        xy = np.vstack([all_truth_flat, all_preds_flat])
        z = stats.gaussian_kde(xy)(xy)
        plt.scatter(all_truth_flat, all_preds_flat, c=z, s=10, alpha=0.5, cmap='jet', norm=LogNorm())

        n_bins, ci_factor = 25, 1.96
        xc_eval, mean_eval, std_eval = self.compute_binned_stats(all_truth_flat, all_preds_flat, n_bins)
        plt.fill_between(xc_eval,
                mean_eval - ci_factor * std_eval,
                mean_eval + ci_factor * std_eval,
                color='blue', alpha=0.35, label=r"$95\%$ CI")

        cbar = plt.colorbar()
        cbar.set_label("")
        plt.xlabel("Benchmark moduli [GPa]")#, fontsize=15)
        plt.ylabel("Predicted moduli [GPa]")#, fontsize=15)
        plt.title(rf"Evaluation on $\rm \mathbb{{R}}^{{{full_rock}}}$ Rocks: $R^2={r2:.3f}$")#, fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/pred_eval_{full_rock}.png", dpi=300)
        print(f"---> save figure as {self.fig_dir}/pred_eval_{full_rock}.png")
        plt.close()


def visualize_3d_rock(raw_file, shape=(75, 75, 75), threshold=0.5, save_path=f"rock_3d.png"):
    # !!! archived function: for future demonstration of 3D rockj visualization
    """
    Load a .raw file, visualize the 3D rock with shading, and save the figure.
    Args:
        raw_file (str): Path to the .raw file.
        shape (tuple): Shape of the 3D rock, default (75, 75, 75).
        threshold (float): Threshold for binarizing rock data (0-1 scale for transparency).
        save_path (str): Path to save the figure.
    """
    # Load raw file data
    data = np.fromfile(raw_file, dtype=np.uint8).reshape(shape)
    x_rock, y_rock, z_rock = np.where(data == 1)
    x_void, y_void, z_void = np.where(data == 0)
    # Create figure
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    # Plot rock (red) and voids (black) with shading
    ax.scatter(x_void, y_void, z_void, c="darkblue", alpha=0.2, s=5, marker="o")  
    ax.scatter(x_rock, y_rock, z_rock, c="darkred", alpha=0.5, s=10, marker="o")  
    # Set labels and view
    ax.view_init(elev=15, azim=25)  # Adjust viewing angle
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"---> save figure as {save_path}")