"""Per-subcube prediction RMSE vs porosity for four CNN targets (2x2 subplots, e.g. B1, size 75).

Stiffness-tensor CNN: RMSE is on Hill bulk K and shear G (two scalars), not on all 36 components,
so it matches the moduli / HS-bound panels.
"""
import argparse, os, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from util_visualization import plotting_params
from util_elasticity import compute_moduli_atomman
from util_machine_learning import resolve_checkpoint_path, MLDRP, stiffness_symmetrize

plt.rcParams.update(plotting_params)

_MULTISCALE_DIR = os.path.dirname(os.path.abspath(__file__))

# (cache_key, legend label, flags, color) -- cache_key stable across label edits
CONFIGS = [
    ("stiffness", "Model 1", dict(homogenization=False, hs_bound=False, benchmark_model=False), "red"),
    ("moduli", "Model 2", dict(homogenization=True, hs_bound=False, benchmark_model=False), "blue"),
    ("hs_bound", "Model 3", dict(homogenization=False, hs_bound=True, benchmark_model=False), "tab:green"),
    ("benchmark_hs_bound", "Benchmark 1", dict(homogenization=False, hs_bound=True, benchmark_model=True), "black"),
]


def _canonical_path(p):
    if not p:
        return p
    p = os.path.expanduser(p)
    if not os.path.isabs(p):
        p = os.path.join(_MULTISCALE_DIR, p)
    return os.path.realpath(os.path.normpath(p))


def per_sample_rmse(pred, truth):
    return np.sqrt(np.mean((pred - truth) ** 2, axis=1))


def rmse_kg_from_stiffness_36(pred, truth):
    """Hill K, G from 6×6 stiffness (flattened 36); RMSE per sample over the two moduli."""
    pred = stiffness_symmetrize(pred)
    truth = stiffness_symmetrize(truth)
    n = pred.shape[0]
    pred_kg = np.array([compute_moduli_atomman(pred[i].reshape(6, 6)) for i in range(n)])
    truth_kg = np.array([compute_moduli_atomman(truth[i].reshape(6, 6)) for i in range(n)])
    return per_sample_rmse(pred_kg, truth_kg)


def infer_predictions(ml):
    dl = DataLoader(ml.full_dataset, batch_size=ml.batch_size, shuffle=False)
    parts = []
    with torch.no_grad():
        for scans, porosities, _ in tqdm(dl, desc="Inference"):
            scans = scans.to(ml.device)
            if ml.model.mode == "hs_bound":
                out = ml.model(scans, porosities.to(ml.device))
            else:
                out = ml.model(scans)
            parts.append(out.cpu().numpy())
    pred = np.concatenate(parts, axis=0)
    truth = np.asarray(ml.full_dataset.labels, dtype=np.float64)
    if pred.shape != truth.shape:
        sys.exit(f"pred {pred.shape} vs truth {truth.shape}")
    if ml.model.mode == "stiffness":
        pred = stiffness_symmetrize(pred)
    return pred, truth


def checkpoint_path(model_dir, rock_type, homogenization, hs_bound, benchmark_model):
    if homogenization:
        name = f"cnn3d_rock_moduli_{rock_type}.pt"
    elif hs_bound:
        name = (f"cnn3d_rock_benchmark_hs_bound_{rock_type}.pt" if benchmark_model
                else f"cnn3d_rock_hs_bound_{rock_type}.pt")
    else:
        name = f"cnn3d_rock_stiffness_{rock_type}.pt"
    return os.path.join(model_dir, name)


def default_paths(rock_type):
    root = os.path.join("..", "..", "..", "machine_learning", "size_75", "data", f"data_{rock_type}", "skin_0")
    return os.path.join(root, "images"), os.path.join(root, "stiffness.dat")


def base_meta(cli):
    scan_c = _canonical_path(cli.scan_dir)
    dat_c = _canonical_path(cli.dat_file)
    model_c = _canonical_path(cli.model_dir)
    meta = {
        "rock_type": cli.rock_type,
        "scan_dir": scan_c,
        "dat_file": dat_c,
        "scan_shape": tuple(cli.scan_shape),
        "model_dir": model_c,
        "stiffness_rmse_metric": "kg",
    }
    if os.path.isfile(dat_c):
        st = os.stat(dat_c)
        meta["dat_fingerprint"] = (dat_c, st.st_mtime, st.st_size)
    else:
        meta["dat_fingerprint"] = None
    return meta


def checkpoint_fingerprints(cli):
    """Stable keys; paths canonical so Slurm cwd does not matter."""
    fp = {}
    mdir = _canonical_path(cli.model_dir)
    for key, _label, flags, _c in CONFIGS:
        p = checkpoint_path(mdir, cli.rock_type, flags["homogenization"], flags["hs_bound"],
                            flags["benchmark_model"])
        if os.path.isfile(p):
            p = os.path.realpath(p)
            st = os.stat(p)
            fp[key] = (p, st.st_mtime, st.st_size)
        else:
            fp[key] = None
    return fp


def compute_series(cli):
    series = []
    fps = checkpoint_fingerprints(cli)
    for _key, label, flags, color in CONFIGS:
        ns = argparse.Namespace(
            rock_type=cli.rock_type, model_dir=cli.model_dir, scan_dir=cli.scan_dir, dat_file=cli.dat_file,
            scan_shape=cli.scan_shape, batch_size=cli.batch_size, homogenization=flags["homogenization"],
            hs_bound=flags["hs_bound"], benchmark_model=flags["benchmark_model"], E1=cli.E1, nu1=cli.nu1,
            E2=cli.E2, nu2=cli.nu2, figures_dir=cli.figures_dir, stop_option="test", loss_avg_step=15,
            early_stop=False, epochs=50, model_name="cnn3d_rock", lr=1e-3, train_ratio=0.8, key=None)
        ckpt = checkpoint_path(cli.model_dir, cli.rock_type, flags["homogenization"], flags["hs_bound"],
                               flags["benchmark_model"])
        if not os.path.isfile(ckpt):
            tqdm.write(f"Skip {label}: missing {ckpt}")
            continue
        ml = MLDRP(ns)
        ml.load_data()
        ml.model.load_state_dict(torch.load(resolve_checkpoint_path(ckpt), map_location=ml.device))
        ml.model.to(ml.device).eval()
        pred, truth = infer_predictions(ml)
        phi = np.asarray(ml.full_dataset.porosities, dtype=np.float64).ravel()
        if ml.model.mode == "stiffness":
            rmse = rmse_kg_from_stiffness_36(pred, truth)
        else:
            rmse = per_sample_rmse(pred, truth)
        series.append((_key, label, color, phi, rmse))
    return series, base_meta(cli), fps


def save_cache(path, series, meta, fps):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"series": series, "meta": meta, "fingerprints": fps}, f)


def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)


LEGACY_FP_KEY_MAP = {
    "Stiffness tensor": "stiffness",
    "Stiffness tensor (K, G)": "stiffness",
    "Homogenized moduli": "moduli",
    "HS-bound": "hs_bound",
    "Benchmark 1": "benchmark_hs_bound",
    "HS-bound (this work)": "hs_bound",
    "HS-bound (our method)": "hs_bound",
    "HS-bound (Benchmark 1)": "benchmark_hs_bound",
    "Model 1": "stiffness",
    "Model 2": "moduli",
    "Model 3": "hs_bound",
    "Benchmark 1": "benchmark_hs_bound",
}


def _series_by_key(series):
    by_key = {}
    for item in series:
        if len(item) == 5:
            key, _label, _color, phi, rmse = item
        else:
            label, _color, phi, rmse = item
            key = LEGACY_FP_KEY_MAP.get(label, label)
        by_key[key] = (phi, rmse)
    return by_key


def _fingerprints_rekey(fp):
    if not fp or "stiffness" in fp:
        return fp
    return {LEGACY_FP_KEY_MAP.get(k, k): v for k, v in fp.items()}


def _fingerprint_mtime_size_equal(cached_fp, current_fp):
    ck = _fingerprints_rekey(cached_fp or {})
    for key, cur in (current_fp or {}).items():
        c = ck.get(key)
        if cur is None and c is None:
            continue
        if cur is None or c is None:
            return False
        if c[1:] != cur[1:]:
            return False
    return True


def cache_valid(data, cli):
    try:
        cur = base_meta(cli)
        cm = data.get("meta") or {}
        if cm.get("rock_type") != cur["rock_type"]:
            return False
        if tuple(cm.get("scan_shape", ())) != cur["scan_shape"]:
            return False
        cdf, ndf = cm.get("dat_fingerprint"), cur.get("dat_fingerprint")
        if cdf is not None and ndf is not None:
            if cdf[1:] != ndf[1:]:
                return False
        if os.path.basename(cm.get("model_dir", "")) != os.path.basename(cur["model_dir"]):
            return False
        if cm.get("stiffness_rmse_metric") != "kg":
            return False
        if not _fingerprint_mtime_size_equal(data.get("fingerprints"), checkpoint_fingerprints(cli)):
            return False
    except Exception:
        return False
    return True


def _global_xy_limits(series):
    """Same x and y range for all panels; y uses log scale (positive RMSE only)."""
    if not series:
        return 0.0, 1.0, 1e-5, 10.0
    by_key = _series_by_key(series)
    phis = [np.asarray(phi, dtype=np.float64).ravel() for phi, _ in by_key.values()]
    rmses = [np.asarray(rmse, dtype=np.float64).ravel() for _, rmse in by_key.values()]
    all_phi = np.concatenate(phis)
    all_rmse = np.concatenate(rmses)
    x_lo, x_hi = float(np.min(all_phi)), float(np.max(all_phi))
    span_x = x_hi - x_lo
    pad_x = 0.03 * span_x + 1e-9
    pos = all_rmse[np.isfinite(all_rmse) & (all_rmse > 0)]
    if pos.size == 0:
        y_lo, y_hi = 1e-5, 1.0
    else:
        y_lo, y_hi = float(np.min(pos)), float(np.max(pos))
        y_hi = max(y_hi, y_lo * 1.001)
    # Log-scale margins (multiplicative)
    y_lo = max(y_lo / 1.8, 1e-20)
    y_hi = y_hi * 1.8
    return x_lo - pad_x, x_hi + pad_x, y_lo, y_hi


def plot_series(series, out_path):
    by_key = _series_by_key(series)
    x_lo, x_hi, y_lo, y_hi = _global_xy_limits(series)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5), sharex=True, sharey=True)
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            key, label, _flags, _color = CONFIGS[i * 2 + j]
            if i == 1:
                ax.set_xlabel(r"Porosity $\phi$", fontsize=15)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            if j == 0:
                ax.set_ylabel(r"RMSE (subcube-level)", fontsize=15)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            if key not in by_key:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes, fontsize=15)
            else:
                phi, rmse = by_key[key]
                ax.scatter(
                    phi, rmse, c="darkblue", s=15, alpha=0.65, edgecolors="none", linewidths=0)
            ax.set_yscale("log")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.text(
                0.5,
                0.95,
                label,
                transform=ax.transAxes,
                fontsize=15,
                va="top",
                ha="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="0.75",
                    linewidth=0.6),
                zorder=6)
    plt.tight_layout()
    out_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    plt.savefig(out_abs, dpi=300, bbox_inches="tight")
    plt.close()
    tqdm.write(f"Saved: {out_abs}")


def run_plot(cli):
    d_scan, d_dat = default_paths(cli.rock_type)
    if cli.scan_dir is None:
        cli.scan_dir = d_scan
    if cli.dat_file is None:
        cli.dat_file = d_dat
    cli.scan_dir = _canonical_path(cli.scan_dir)
    cli.dat_file = _canonical_path(cli.dat_file)
    cli.model_dir = _canonical_path(cli.model_dir)
    cli.figures_dir = _canonical_path(cli.figures_dir)
    cache_path = (_canonical_path(cli.cache_file) if cli.cache_file is not None else os.path.join(
        cli.figures_dir, f"phi_rmse_cache_{cli.rock_type}_{os.path.basename(cli.model_dir)}.pkl"))
    out_path = (_canonical_path(cli.out) if cli.out is not None else os.path.join(
        cli.figures_dir, f"phi_rmse_{cli.rock_type}.png"))

    series = None
    if not cli.recompute and os.path.isfile(cache_path):
        try:
            data = load_cache(cache_path)
            if cache_valid(data, cli):
                series = data["series"]
                tqdm.write(f"Loaded cache: {cache_path} (plotting only)")
        except Exception as e:
            tqdm.write(f"Cache unreadable ({e}), recomputing.")

    if series is None:
        series, meta, fps = compute_series(cli)
        save_cache(cache_path, series, meta, fps)
        tqdm.write(f"Wrote cache: {cache_path}")

    plot_series(series, out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RMSE vs porosity for four CNN models (size-75 subcubes)")
    ap.add_argument("--rock_type", type=str, default="B1")
    ap.add_argument("--model_dir", type=str, default="model")
    ap.add_argument("--scan_dir", type=str, default=None, help="Default: .../data_{rock_type}/skin_0/images")
    ap.add_argument("--dat_file", type=str, default=None, help="Default: .../data_{rock_type}/skin_0/stiffness.dat")
    ap.add_argument("--scan_shape", type=int, nargs=3, default=[75, 75, 75])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--figures_dir", type=str, default="figures")
    ap.add_argument("--out", type=str, default=None, help="Default: figures/phi_rmse_{rock_type}.png")
    ap.add_argument("--cache_file", type=str, default=None, help="Default: figures/phi_rmse_cache_{rock}_{model_dir}.pkl")
    ap.add_argument("--recompute", action="store_true", help="Ignore cache and re-run inference")
    ap.add_argument("--E1", type=float, default=2.903225806452e-04)
    ap.add_argument("--nu1", type=float, default=0.4516129032258)
    ap.add_argument("--E2", type=float, default=95.29411764705883)
    ap.add_argument("--nu2", type=float, default=0.058823529411764705)
    run_plot(ap.parse_args())
