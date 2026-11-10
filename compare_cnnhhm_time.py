"""Time CNN+HHM for one large-cube rock (subcubes 75^3). Run from multiscale/; checkpoints in --model_dir."""
import argparse, os, shutil, sys, time
import numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("data")
from order import read_rock_matrices, write_matrices_to_dat
from eval_cnn_hhm_moduli import homogenize_fullcube_stiffness, split_stiffness_file
from util_data_processing import RockDataset, load_scans_and_labels, write_stiffness_dat
from util_machine_learning import resolve_checkpoint_path, MLDRP, stiffness_symmetrize


def latex_sci_sec(x):
    if x is None:
        return r"\textemdash"
    mant, exp = f"{float(x):.2e}".lower().split("e")
    return f"${mant}\\times 10^{{{int(exp)}}}$"


def tex_escape(s):
    return s.replace("&", r"\&")


def tex_comment_safe(s):
    return s.replace("%", r"\%")


def autodetect_cnn_device():
    """Same rule as MLDRP: CUDA if available, else CPU."""
    if torch.cuda.is_available():
        idx = int(torch.cuda.current_device())
        name = torch.cuda.get_device_name(idx)
        return dict(
            kind="GPU",
            print_line=f"GPU — {name} [cuda:{idx}]",
            tex_cnn=f"CNN inference: GPU, {tex_comment_safe(tex_escape(name))}, \\texttt{{cuda:{idx}}}")
    return dict(kind="CPU", print_line="CPU", tex_cnn=r"CNN inference: \texttt{CPU}")


def latex_caption_cnn_hhm(hw):
    return r"Using \textit{CNN:} \texttt{" + hw["kind"] + r"} \quad \textit{HHM:} \texttt{CPU}"


def default_tex_label(L, cnn_kind):
    return f"tab:time_{cnn_kind.lower()}_r{L}"


class SubcubeHomogenizationTiming:
    @staticmethod
    def sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def checkpoint_path(args):
        if args.homogenization:
            name = f"cnn3d_rock_moduli_{args.rock_type}.pt"
        elif args.hs_bound:
            name = (f"cnn3d_rock_benchmark_hs_bound_{args.rock_type}.pt" if args.benchmark_model
                    else f"cnn3d_rock_hs_bound_{args.rock_type}.pt")
        else:
            name = f"cnn3d_rock_stiffness_{args.rock_type}.pt"
        return os.path.join(args.model_dir, name)

    @staticmethod
    def stiffness_dat_path(args):
        b = os.path.join(args.med_stiff_dir, "cnn_pred")
        if args.homogenization:
            return f"{b}_moduli.dat"
        if args.hs_bound:
            return (f"{b}_benchmark_hs_bound_stiffness.dat" if args.benchmark_model
                    else f"{b}_hs_bound_stiffness.dat")
        return f"{b}_stiffness.dat"

    @staticmethod
    def moduli_to_stiff36(all_mods):
        K, G = all_mods[:, 0], all_mods[:, 1]
        lam = K - 2.0 * G / 3.0
        c11, c12, c44 = lam + 2.0 * G, lam, G
        z = np.zeros(all_mods.shape[0])
        return np.column_stack([
            c11, c12, c12, z, z, z, c12, c11, c12, z, z, z, c12, c12, c11, z, z, z,
            z, z, z, c44, z, z, z, z, z, z, c44, z, z, z, z, z, z, c44])

    @classmethod
    def infer_subset(cls, ml, n_sub, out_path, X, fl, poro):
        X, fl, poro = X[:n_sub], fl[:n_sub], poro[:n_sub]
        y_dummy = np.zeros((n_sub, 36), dtype=np.float32)
        dl = DataLoader(RockDataset(X, poro, y_dummy), batch_size=ml.batch_size, shuffle=False)
        ml.model.eval()
        mode = ml.model.mode
        with torch.no_grad():
            if mode == "stiffness":
                preds = [ml.model(s.to(ml.device)).cpu().numpy() for s, _, _ in dl]
                arr = stiffness_symmetrize(np.concatenate(preds, axis=0))
            elif mode == "moduli":
                mods = [ml.model(s.to(ml.device)).cpu().numpy() for s, _, _ in dl]
                arr = cls.moduli_to_stiff36(np.vstack(mods))
            elif mode == "hs_bound":
                mods = [ml.model(s.to(ml.device), p.to(ml.device)).cpu().numpy()
                        for s, p, _ in dl]
                arr = cls.moduli_to_stiff36(np.vstack(mods))
            else:
                raise ValueError(f"Unknown model.mode={mode!r}")
        pt = [(float(p), float(p)) for p in poro]
        write_stiffness_dat(arr, out_path, filenames=fl, porosities=pt)

    @classmethod
    def seconds_one_rock(cls, ml, args, n_sub, chunk_size, X, fl, poro):
        """Wall time: CNN on n_sub subcubes, reorder/split I/O, then FFT-HHM solve (one R^L cube)."""
        os.makedirs(args.med_stiff_dir, exist_ok=True)
        path = cls.stiffness_dat_path(args)
        cls.sync_cuda()
        t0 = time.perf_counter()
        cls.infer_subset(ml, n_sub, path, X, fl, poro)
        cls.sync_cuda()
        t_cnn = time.perf_counter() - t0
        t1 = time.perf_counter()
        rock_matrices = read_rock_matrices(path, apply_increment=False)
        write_matrices_to_dat(rock_matrices, path)
        small = os.path.join(args.med_stiff_dir, "CNN_small")
        split_stiffness_file(input_path=path, output_dir=small, chunk_size=chunk_size,
                             target_size=args.scan_shape[0])
        t_post = time.perf_counter() - t1
        t2 = time.perf_counter()
        s75 = os.path.join(small, f"stiffness_{args.scan_shape[0]}_1.dat")
        homog_dir = os.path.join(args.med_stiff_dir, "CNN_75_homog")
        homogenize_fullcube_stiffness(Path(s75), rock_id=1, data_dir=homog_dir,
                                      mat_type=args.mat_type, avg_scheme=args.avg_scheme)
        cls.sync_cuda()
        t_hhm = time.perf_counter() - t2
        return t_cnn, t_post, t_hhm, t_cnn + t_post + t_hhm

    @staticmethod
    def ml_namespace(cli, med_sub=None):
        rt = cli.rock_type
        local_scan = os.path.join("size75", rt, "images")
        local_dat = os.path.join("size75", rt, "stiffness.dat")
        if cli.scan_dir:
            scan = cli.scan_dir
        elif os.path.isdir(local_scan):
            scan = local_scan
        else:
            scan = os.path.join("/path/to/digital_rock/machine_learning/bulk_shear_learning/size_75/data",
                                f"data_{rt}", "skin_0", "images")
        if cli.dat_file:
            dat = cli.dat_file
        elif os.path.isfile(local_dat):
            dat = local_dat
        else:
            dat = os.path.join("/path/to/digital_rock/machine_learning/bulk_shear_learning/size_75/data",
                               f"data_{rt}", "skin_0", "stiffness.dat")
        return argparse.Namespace(
            rock_type=rt, model_dir=cli.model_dir, scan_dir=scan, dat_file=dat,
            scan_shape=cli.scan_shape, batch_size=cli.batch_size, homogenization=False, hs_bound=False,
            benchmark_model=False, E1=cli.E1, nu1=cli.nu1, E2=cli.E2, nu2=cli.nu2, figures_dir=cli.figures_dir,
            stop_option=cli.stop_option, loss_avg_step=cli.loss_avg_step, early_stop=cli.early_stop,
            epochs=cli.epochs, mat_type=cli.mat_type, avg_scheme=cli.avg_scheme,
            med_stiff_dir=med_sub or cli.med_stiff_dir, model_name="cnn3d_rock", lr=1e-3, train_ratio=0.8, key=None)

    @staticmethod
    def write_tex(path, L, rows_tex, hw, label):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        cap = latex_caption_cnn_hhm(hw)
        lines = [
            r"% auto-generated by compare_cnnhhm_time.py",
            rf"% $t_{{\mathrm{{tot}}}}=t_{{\mathrm{{CNN}}}}+t_{{\mathrm{{prep}}}}+t_{{\mathrm{{HHM}}}}$ (s); {hw['tex_cnn']}",
            r"\begin{table}[htbp]",
            r"    \centering",
            f"    \\caption{{{cap}}}",
            f"    \\label{{{label}}}",
            r"    \begin{tabular}{lcccc}",
            r"    \hline",
            r"    Approach & $t_{\mathrm{CNN}}$ [s] & $t_{\mathrm{prep}}$ [s] & $t_{\mathrm{HHM}}$ [s] & $t_{\mathrm{tot}}$ [s] \\",
            r"    \hline",
        ]
        lines.extend("    " + r for r in rows_tex)
        lines.extend([r"    \hline", r"    \end{tabular}", r"\end{table}", ""])
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @classmethod
    def run(cls, cli):
        s = cli.scan_shape[0]
        chunk = int((cli.largecube_size / s) ** 3)
        if chunk * (s ** 3) != cli.largecube_size ** 3:
            raise SystemExit("largecube_size must be an integer multiple of scan_shape cube edge.")
        configs = [
            ("(1) Stiffness tensor", dict(homogenization=False, hs_bound=False, benchmark_model=False)),
            ("(2) Homogenized moduli", dict(homogenization=True, hs_bound=False, benchmark_model=False)),
            ("(3) HS-bound", dict(homogenization=False, hs_bound=True, benchmark_model=False)),
            ("(4) Benchmark 1 (prior HS-bound)", dict(homogenization=False, hs_bound=True, benchmark_model=True)),
        ]
        hw = autodetect_cnn_device()
        ns0 = cls.ml_namespace(cli)
        t_ld0 = time.perf_counter()
        X, _Y, fl, poro = load_scans_and_labels(ns0.scan_dir, ns0.dat_file, tuple(cli.scan_shape))
        t_load = time.perf_counter() - t_ld0
        tqdm.write(
            f"CNN+HHM timing — one R^{cli.largecube_size} rock = {chunk} subcubes "
            f"({s}³); batch_size={cli.batch_size}")
        tqdm.write(f"CNN inference hardware: {hw['print_line']}")
        tqdm.write("HHM (FFT homogenization): CPU")
        tqdm.write(f"One-time load_scans_and_labels (full list, inference uses first {chunk}): {t_load:.2f} s\n")
        med_root = os.path.abspath(cli.med_stiff_dir)
        if os.path.isdir(med_root) and not cli.keep_temp:
            shutil.rmtree(med_root, ignore_errors=True)
        row_names_tex = ("Model 1", "Model 2", "Model 3", "Benchmark 1")
        results = []
        tex_rows = []
        for run_idx, (label, flags) in enumerate(tqdm(configs, desc="Approaches"), start=1):
            rname = row_names_tex[run_idx - 1]
            args = cls.ml_namespace(cli, os.path.join(med_root, f"run_{run_idx}"))
            args.homogenization, args.hs_bound, args.benchmark_model = (
                flags["homogenization"], flags["hs_bound"], flags["benchmark_model"])
            ckpt = cls.checkpoint_path(args)
            if not os.path.isfile(ckpt):
                tqdm.write(f"{label}: SKIP (missing checkpoint: {ckpt})")
                results.append((label, None, None, None, None))
                tex_rows.append(
                    f"{rname} & {latex_sci_sec(None)} & {latex_sci_sec(None)} & "
                    f"{latex_sci_sec(None)} & {latex_sci_sec(None)} \\\\")
                continue
            ml = MLDRP(args)
            ml.init_model_from_args()
            ml.model.load_state_dict(torch.load(resolve_checkpoint_path(ckpt), map_location=ml.device))
            ml.model.to(ml.device).eval()
            if cli.warmup:
                os.makedirs(args.med_stiff_dir, exist_ok=True)
                wp = cls.stiffness_dat_path(args) + ".warmup"
                cls.infer_subset(ml, min(2, chunk), wp, X, fl, poro)
                if os.path.isfile(wp):
                    os.remove(wp)
                cls.sync_cuda()
            os.makedirs(args.med_stiff_dir, exist_ok=True)
            cnn, post, hhm, tot = cls.seconds_one_rock(ml, args, chunk, chunk, X, fl, poro)
            results.append((label, cnn, post, hhm, tot))
            tqdm.write(
                f"{label}: total {tot:.2f} s = CNN {cnn:.2f} + reorder/split {post:.2f} + HHM {hhm:.2f}")
            tex_rows.append(
                f"{rname} & {latex_sci_sec(cnn)} & {latex_sci_sec(post)} & "
                f"{latex_sci_sec(hhm)} & {latex_sci_sec(tot)} \\\\")
        tqdm.write("\n--- summary: $t_{\mathrm{tot}}$ = CNN + reorder/split + HHM (s) ---")
        for label, cnn, post, hhm, tot in results:
            if tot is None:
                tqdm.write(f"  {label}: N/A")
            else:
                tqdm.write(f"  {label}: {tot:.2f} s  (CNN {cnn:.2f} | split {post:.2f} | HHM {hhm:.2f})")
        tqdm.write(f"(CNN on {hw['kind']}; HHM on CPU)\n")
        tex_path = cli.tex_out or os.path.join(cli.figures_dir, f"cnn_hhm_R{cli.largecube_size}_time.tex")
        tab_label = cli.tex_label or default_tex_label(cli.largecube_size, hw["kind"])
        cap_tex = latex_caption_cnn_hhm(hw)
        cls.write_tex(tex_path, cli.largecube_size, tex_rows, hw, tab_label)
        tqdm.write(f"\nWrote LaTeX table: {tex_path}")
        tqdm.write("LaTeX caption (matches \\caption{...} in file):")
        tqdm.write(f"    {cap_tex}")
        tqdm.write(f"LaTeX label: \\label{{{tab_label}}}")
        if not cli.keep_temp and os.path.isdir(med_root):
            shutil.rmtree(med_root, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CNN+HHM wall time for one R^300 rock (64×75³ subcubes)")
    ap.add_argument("--rock_type", type=str, default="B1")
    ap.add_argument("--model_dir", type=str, default="model")
    ap.add_argument("--scan_dir", type=str, default=None, help="Override images dir")
    ap.add_argument("--dat_file", type=str, default=None, help="Override stiffness.dat")
    ap.add_argument("--largecube_size", type=int, default=300, choices=(300, 600))
    ap.add_argument("--scan_shape", type=int, nargs=3, default=[75, 75, 75])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--mat_type", type=str, default="isotropic")
    ap.add_argument("--avg_scheme", type=str, default="Hill")
    ap.add_argument("--med_stiff_dir", type=str, default="data_hhm_time_compare")
    ap.add_argument("--figures_dir", type=str, default="figures")
    ap.add_argument("--tex_out", type=str, default=None, help="LaTeX table path (default: figures/cnn_hhm_R{L}_time.tex)")
    ap.add_argument("--tex_cite", type=str, default="priorHHM2023", help="Bib key for 4th row Ref.~\\cite{...}")
    ap.add_argument("--tex_label", type=str, default=None, help=r"Table \label (default: tab:time_cpu_r{L} or tab:time_gpu_r{L})")
    ap.add_argument("--keep_temp", action="store_true")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--E1", type=float, default=2.903225806452e-04)
    ap.add_argument("--nu1", type=float, default=0.4516129032258)
    ap.add_argument("--E2", type=float, default=95.29411764705883)
    ap.add_argument("--nu2", type=float, default=0.058823529411764705)
    ap.add_argument("--stop_option", type=str, default="test")
    ap.add_argument("--loss_avg_step", type=int, default=15)
    ap.add_argument("--early_stop", action="store_true", default=False)
    ap.add_argument("--epochs", type=int, default=50)
    SubcubeHomogenizationTiming.run(ap.parse_args())
