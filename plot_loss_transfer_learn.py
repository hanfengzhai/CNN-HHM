import numpy as np, glob, os, re, argparse
import matplotlib.pyplot as plt, matplotlib.cm as cm
from tqdm import tqdm
from util_visualization import plotting_params
plt.rcParams.update(plotting_params)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot training and testing loss for transfer learning")
    p.add_argument("-d","--data_dir",default="data",help="Base data directory")
    p.add_argument("-o","--output",default="loss_transfer_learn.png",help="Output figure path")
    p.add_argument("-s","--skip",default="600",help="Substring to skip npy files")
    p.add_argument("--fig_dir", default="figures", help="Directory to save figures")
    p.add_argument("--loss_cache_dir", default="fig_B1",
                   help="Directory with loss_{stiffness,moduli,hs_bound,benchmark_hs_bound}.npy for summary plot")
    p.add_argument("--summary_out", default=None,
                   help="Output path for loss_summary_B1.png (default: fig_dir/loss_summary_B1.png)")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    pattern = re.compile(rf"{args.data_dir}/fine_tune_(?P<rock>[^/]+)/loss_(?P<rock2>[^_]+)_(?P<method>.+)_large\d+_full\d+\.npy")
    method_order = ["stiffness", "moduli", "hs_bound", "benchmark_hs_bound"]
    titles = {"stiffness": "Model 1", "moduli": "Model 2",
              "hs_bound": "Model 3",
              "benchmark_hs_bound": "Benchmark 1"}

    files = sorted(glob.glob(f"{args.data_dir}/fine_tune_*/loss_*.npy"))
    files_by_method = {m:[] for m in method_order}
    for fp in tqdm(files, desc="Collecting files"):
        path = fp.replace(os.sep,"/")
        if args.skip in path: continue
        m = pattern.match(path)
        if m:
            rock,rock2,method = m.group("rock"),m.group("rock2"),m.group("method")
            if rock==rock2 and method in method_order:
                files_by_method[method].append((rock,fp))

    fig, axes = plt.subplots(2, 4, figsize=(12,6), sharex=True, sharey="row")
    for col,method in enumerate(method_order):
        entries = files_by_method[method]
        n = len(entries)
        for i,(rock,fp) in enumerate(entries):
            tr,te = np.load(fp,allow_pickle=True)
            epochs = np.arange(tr.shape[0])
            c = cm.seismic(i/(n-1) if n>1 else 0.5)
            axes[0,col].semilogy(epochs,tr,label=rock,color=c)
            axes[1,col].semilogy(epochs,te,label=rock,color=c)
        axes[0,col].set_title(titles[method],fontsize=15)
        if col==0: axes[0,col].set_ylabel("Training Loss")
        axes[0,col].legend(loc="upper right")
        axes[1,col].set_xlabel("Epoch")
        if col==0: axes[1,col].set_ylabel("Testing Loss")
        axes[1,col].legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(args.fig_dir, exist_ok=True)
    plt.savefig(os.path.join(args.fig_dir, args.output), dpi=args.dpi)
    print(f"---> saved figure: {os.path.join(args.fig_dir, args.output)}")
    plt.close(fig)

    fig2, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    color_map = {"stiffness": "red", "moduli": "darkred",
                 "hs_bound": "darkblue", "benchmark_hs_bound": "blue"}
    style_map = {"stiffness": "-.", "moduli": "-",
                 "hs_bound": "-", "benchmark_hs_bound": "-."}
    for key in method_order:
        fp = os.path.join(args.loss_cache_dir, f"loss_{key}.npy")
        if not os.path.isfile(fp):
            print(f"---> skip missing cache: {fp}")
            continue
        label = titles[key]
        c = color_map[key]
        ls = style_map[key]
        arr = np.load(fp, allow_pickle=True)
        if arr.ndim == 2:
            tr, te = arr
            epochs = np.arange(tr.shape[0])
            ax_tr.semilogy(epochs, tr, label=label, color=c, linestyle=ls)
            ax_te.semilogy(epochs, te, label=label, color=c, linestyle=ls)
        else:
            epochs = np.arange(arr.shape[0])
            ax_tr.semilogy(epochs, arr, label=label, color=c, linestyle=ls)
            ax_te.semilogy(epochs, arr, label=label, color=c, linestyle=ls)
    ax_tr.set_xlabel('Epoch')
    ax_tr.set_ylabel('Training Loss')
    ax_tr.legend(loc='upper left')
    ax_te.set_xlabel('Epoch')
    ax_te.set_ylabel('Testing Loss')
    plt.tight_layout()
    summary_out = args.summary_out or os.path.join(args.fig_dir, 'loss_summary_B1.png')
    os.makedirs(os.path.dirname(os.path.abspath(summary_out)) or '.', exist_ok=True)
    plt.savefig(summary_out, dpi=args.dpi)
    print(f"---> saved figure: {summary_out} (cache: {args.loss_cache_dir})")