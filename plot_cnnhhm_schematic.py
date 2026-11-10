#!/usr/bin/env python3
import argparse, numpy as np, matplotlib.pyplot as plt, glob, random
from matplotlib.patches import ConnectionPatch
from util_visualization import plotting_params
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update(plotting_params)
random.seed(2025); np.random.seed(2025)

class RockVisualizer:
    def __init__(self, filename, dims, axis, slice_idx, 
                 fig_dir, subcube, cmap, rock_type, dpi=100):
        self.filename, self.dims, self.axis = filename, dims, axis
        self.slice_idx = slice_idx or dims[axis] // 2
        self.fig_dir, self.subcube, self.cmap = fig_dir, subcube, cmap
        self.rock_type = rock_type
        self._load_volume()
        self.dpi = dpi

    def _load_volume(self):
        nx, ny, nz = self.dims
        data = np.fromfile(self.filename, dtype=np.int8)
        try:
            self.vol = data.reshape((nx, ny, nz))
        except ValueError:
            raise ValueError(f"Cannot reshape data of size {data.size} into {self.dims}")

    def _get_slice(self):
        nx, ny, nz = self.dims
        idx = self.slice_idx
        if self.axis == 0:
            img = self.vol[idx, :, :]; h, w = ny, nz
        elif self.axis == 1:
            img = self.vol[:, idx, :]; h, w = nx, nz
        else:
            img = self.vol[:, :, idx]; h, w = nx, ny
        return img, (h, w)

    def plot_slice(self):
        img, (h, w) = self._get_slice()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap=self.cmap, origin='lower'); ax.axis('off')
        for x in range(0, w+1, self.subcube): ax.axvline(x-0.5, color='darkred', linewidth=1.2)
        for y in range(0, h+1, self.subcube): ax.axhline(y-0.5, color='darkred', linewidth=1.2)
        arr = dict(arrowstyle='<->', color='darkred', linewidth=1)
        ax.annotate('', xy=(0, -0.02), xytext=(1, -0.02),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=arr, clip_on=False)
        ax.text(0.5, -0.05, r'$N$', transform=ax.transAxes,
                ha='center', va='top', color='darkred', fontsize=30, clip_on=False)
        frac = self.subcube / h
        ax.annotate('', xy=(-0.02, 0), xytext=(-0.02, frac),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=arr, clip_on=False)
        ax.text(-0.05, frac/2, r'$n$', transform=ax.transAxes,
                ha='right', va='center', color='darkred', fontsize=30, clip_on=False)
        ax.text(-0.1, -0.15, '(a)', transform=ax.transAxes,
                ha='left', va='bottom', color='black', fontsize=30, clip_on=False)
        plt.tight_layout()
        out = f"{self.fig_dir}/{self.rock_type}_slice_{self.slice_idx}.png"
        fig.savefig(out, dpi=self.dpi)
        print(f"---> saved figure: {out}")
        return img, (h, w)

    def plot_subcubes(self, k=4):
        files = glob.glob(f'/path/to/digital_rock/rock_images_segmented/{self.rock_type}/size_300/skin_0/size_{self.subcube}/*.raw')
        picked = random.sample(files, k)
        subcubes, means = [], []
        for f in picked:
            data = np.fromfile(f, dtype=np.int8)
            cube = data.reshape((self.subcube, self.subcube, self.subcube))
            sl = cube[:, :, cube.shape[2] // 2]
            subcubes.append(sl); means.append(sl.mean())
        fig = plt.figure(figsize=(5, k * 1.8))
        fig.suptitle('Homogenization\nusing CNN / DNS', fontsize=30, fontweight='bold', y=1.0)
        gs = fig.add_gridspec(k, 2, wspace=0.3, hspace=0.3)
        for i, (cube, μ) in enumerate(zip(subcubes, means)):
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(cube, cmap='gray', origin='lower'); ax1.set_xticks([]); ax1.set_yticks([])
            for s in ax1.spines.values(): s.set_edgecolor('darkred'); s.set_linewidth(1.5)

            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow([[μ]], cmap='gray', vmin=0, vmax=1, origin='lower'); ax2.set_xticks([]); ax2.set_yticks([])
            for s in ax2.spines.values(): s.set_edgecolor('darkred'); s.set_linewidth(1.5)

            con = ConnectionPatch(xyA=(1, 0.5), coordsA=ax1.transAxes, xyB=(0, 0.5), coordsB=ax2.transAxes,
                arrowstyle='->,head_width=1,head_length=1', linestyle='--', lw=2, color='black', shrinkA=0, shrinkB=0)
            fig.add_artist(con)

            if i == 0:
                arr = dict(arrowstyle='<->', color='darkred', linewidth=2)
                ax1.annotate('', xy=(-0.1, 0), xytext=(-0.1, 1), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arr, clip_on=False)
                ax1.text(-0.2, 0.5, r'$n$', transform=ax1.transAxes, ha='right', va='center', color='darkred', fontsize=37, clip_on=False)
                ax2.text( 0.5, 0.5, r'$C_{ijkl}$', transform=ax2.transAxes, ha='center', va='center', color='darkred', fontsize=40, clip_on=False)

            if i == k - 1:
                ax1.text(-0.6, -0.55, '(b)', transform=ax1.transAxes, ha='left', va='bottom', color='black', fontsize=40, clip_on=False)

        plt.tight_layout()
        out = f"{self.fig_dir}/{self.rock_type}_subcubes.png"
        fig.savefig(out, dpi=self.dpi)
        print(f"---> saved figure: {out}")

    def plot_homogenized(self, img, size):
        h, w = size; sub = self.subcube
        hom = np.zeros_like(img, float)
        for y in range(0, h, sub):
            for x in range(0, w, sub): hom[y:y+sub, x:x+sub] = img[y:y+sub, x:x+sub].mean()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(hom, cmap=self.cmap, origin='lower', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for x in range(0, w+1, sub): ax.axvline(x-0.5, color='darkred', linewidth=1.2)
        for y in range(0, h+1, sub): ax.axhline(y-0.5, color='darkred', linewidth=1.2)
        for s in ax.spines.values(): s.set_edgecolor('darkred'); s.set_linewidth(1.5)
        arr = dict(arrowstyle='<->', color='darkred', linewidth=1)
        ax.annotate('', xy=(0, -0.02), xytext=(1, -0.02), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arr, clip_on=False)
        ax.text(0.5, -0.05, r'$N\ /\ n$', transform=ax.transAxes, ha='center', va='top', color='darkred', fontsize=30, clip_on=False)
        ax.text(-0.1, -0.15, '(c)', transform=ax.transAxes, ha='left', va='bottom', color='black', fontsize=30, clip_on=False)
        plt.tight_layout(); out = f"{self.fig_dir}/{self.rock_type}_homogenized.png"; fig.savefig(out, dpi=self.dpi)
        print(f"---> saved figure: {out}")
        return hom

    def plot_global_mean(self, hom):
        m = hom.mean()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow([[m]], cmap=self.cmap, vmin=0, vmax=1, origin='lower')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor('darkred'); s.set_linewidth(1.5)
        ax.text(0.5, 0.5, r'$K,\ G$', transform=ax.transAxes, ha='center', va='center', color='darkred', fontsize=30)
        ax.text(-0.15, -0.19, '(d)', transform=ax.transAxes, ha='left', va='bottom', color='black', fontsize=30, clip_on=False)
        plt.tight_layout(); out = f"{self.fig_dir}/{self.rock_type}_global_mean.png"; fig.savefig(out, dpi=self.dpi)
        print(f"---> saved figure: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualize rock scans and subcube homogenization steps")
    p.add_argument("-f", "--filename", default="/path/to/digital_rock/rock_images_segmented/B1/size_1200/Rock_1200.raw", help="Path to raw rock file")
    p.add_argument("-d", "--dims", nargs=3, type=int, default=(1200,1200,1200), metavar=("NX","NY","NZ"), help="Volume dimensions")
    p.add_argument("-a", "--axis", type=int, choices=[0,1,2], default=2, help="Slice axis")
    p.add_argument("-s", "--slice_idx", type=int, help="Slice index; default= center")
    p.add_argument("-r", "--rock_type", default="B1", help="Rock type")
    p.add_argument("-fig_dir", "--fig_dir", default="figures/", help="Directory for output figures")
    p.add_argument("--subcube", type=int, default=75, help="Subcube size")
    p.add_argument("--cmap", default="gray", help="Colormap")
    p.add_argument("--dpi", type=int, default=50, help="Figure DPI")
    args = p.parse_args()

    viz = RockVisualizer(args.filename, args.dims, args.axis, args.slice_idx, 
                         args.fig_dir, args.subcube, args.cmap, args.rock_type, args.dpi)
    img, size = viz.plot_slice()
    viz.plot_subcubes()
    hom = viz.plot_homogenized(img, size)
    viz.plot_global_mean(hom)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    files = [
        f"{args.fig_dir}/{args.rock_type}_slice_{viz.slice_idx}.png",
        f"{args.fig_dir}/{args.rock_type}_subcubes.png",
        f"{args.fig_dir}/{args.rock_type}_homogenized.png",
        f"{args.fig_dir}/{args.rock_type}_global_mean.png",]
    for ax, fname in zip(axes, files):
        img_m = plt.imread(fname)
        ax.imshow(img_m)
        ax.axis("off")

    start_xs = [0.25, 0.46, 0.74]
    end_xs   = [0.30, 0.51, 0.78]

    for sx, ex in zip(start_xs, end_xs):
        arrow = FancyArrowPatch(
            (sx, 0.5), (ex, 0.5),
            transform=fig.transFigure,
            arrowstyle='->',
            mutation_scale=20,
            lw=3, color='darkblue')
        fig.add_artist(arrow)
    plt.tight_layout()
    out = f"{args.fig_dir}/{args.rock_type}_overview.png"
    fig.savefig(out, dpi=args.dpi)
    print(f"---> saved figure: {out}")