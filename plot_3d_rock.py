"""Segmented .raw volume to PNG: PyVista (VTK/OpenGL) or matplotlib (no GL, for headless nodes)."""
import argparse, glob, math, os, sys
import numpy as np
from tqdm import tqdm


def _configure_headless_vtk():
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


class VolumeSurfaceExport:
    @staticmethod
    def resolve_raw_path(cli):
        if cli.raw_file:
            p = os.path.abspath(cli.raw_file)
            if not os.path.isfile(p):
                sys.exit(f"Not found: {p}")
            return p
        if not os.path.isdir(cli.rock_dir):
            sys.exit(f"rock_dir does not exist: {cli.rock_dir}\nSet --raw_file or --rock_dir.")
        pat = os.path.join(cli.rock_dir, cli.glob_pat)
        m = sorted(glob.glob(pat))
        if not m:
            sys.exit(f"No files matching {pat}")
        return m[0]

    @staticmethod
    def load_volume(path, shape, dtype_str, subsample):
        vol = np.fromfile(path, dtype=getattr(np, dtype_str))
        nx, ny, nz = shape
        need = nx * ny * nz
        if vol.size != need:
            sys.exit(f"Expected {need} voxels ({nx}x{ny}x{nz}), got {vol.size} from {path}")
        vol = vol.reshape((nx, ny, nz), order="C")
        if subsample > 1:
            vol = vol[::subsample, ::subsample, ::subsample]
        return vol.astype(np.float32)

    @staticmethod
    def _downsample_for_mc(vol, cap):
        mx = max(vol.shape)
        if mx <= cap:
            return vol, 1
        s = int(math.ceil(mx / cap))
        return vol[::s, ::s, ::s], s

    @staticmethod
    def export_matplotlib(vol, cli, out):
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            sys.exit("matplotlib backend needs scikit-image: pip install scikit-image")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        plotting_params = {"font.family": "serif", "font.serif": ["Libertinus Serif"],
                           "mathtext.fontset": "cm", "xtick.labelsize": 15, "ytick.labelsize": 15,
                           "axes.labelsize": 15, "legend.fontsize": 15}
        plt.rcParams.update(plotting_params)
        v, stride = VolumeSurfaceExport._downsample_for_mc(vol, cli.mc_cap)
        if stride > 1:
            tqdm.write(f"marching_cubes: stride {stride} -> shape {v.shape} (cap={cli.mc_cap})")
        mc = marching_cubes(v, level=0.5)
        verts, faces = mc[0], mc[1]
        fig = plt.figure(figsize=(cli.win_w / 100.0, cli.win_h / 100.0), dpi=100, facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        mesh = Poly3DCollection(verts[faces], facecolor=cli.color, edgecolor="none", linewidths=0)
        ax.add_collection3d(mesh)
        ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
        ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
        ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
        ax.set_xlabel(r"$x$", fontsize=15)
        ax.set_ylabel(r"$y$", fontsize=15)
        ax.set_zlabel(r"$z$", fontsize=15)
        ax.view_init(elev=22, azim=45)
        plt.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    @staticmethod
    def export_pyvista(vol, cli, out):
        _configure_headless_vtk()
        import pyvista as pv
        if cli.use_xvfb:
            try:
                pv.start_xvfb()
            except Exception:
                pass
        grid = pv.ImageData()
        grid.dimensions = np.array(vol.shape) + 1
        grid.spacing = (1.0, 1.0, 1.0)
        grid.origin = (0.0, 0.0, 0.0)
        grid.cell_data["values"] = vol.flatten(order="F")
        lo, hi = float(cli.solid_value) - 0.5, float(cli.solid_value) + 0.5
        surf = grid.threshold(value=[lo, hi], scalars="values").extract_surface()
        if surf.n_points == 0:
            sys.exit("Empty surface after threshold - check solid_value / dtype / data range.")
        surf = surf.smooth(n_iter=cli.smooth_iter, relaxation_factor=0.1)
        pv.set_plot_theme("document")
        plotter = pv.Plotter(off_screen=True, window_size=(cli.win_w, cli.win_h))
        plotter.set_background("white")
        plotter.add_mesh(surf, color=cli.color, smooth_shading=True, specular=0.25, ambient=0.18,
                         diffuse=0.85, show_edges=False)
        plotter.add_axes(line_width=2)
        plotter.camera_position = "iso"
        if cli.msaa:
            try:
                plotter.enable_anti_aliasing("msaa")
            except Exception:
                pass
        plotter.show(screenshot=out, auto_close=False)
        plotter.close()

    @classmethod
    def run(cls, cli):
        path = cls.resolve_raw_path(cli)
        nx, ny, nz = cli.shape
        if cli.subsample > 1:
            nx, ny, nz = nx // cli.subsample, ny // cli.subsample, nz // cli.subsample
        tqdm.write(f"Loading: {path}")
        tqdm.write(f"Shape (after subsample={cli.subsample}): {nx}x{ny}x{nz}")
        vol = cls.load_volume(path, cli.shape, cli.dtype, cli.subsample)
        out = os.path.abspath(cli.out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        if cli.backend == "matplotlib":
            cls.export_matplotlib(vol, cli, out)
        else:
            cls.export_pyvista(vol, cli, out)
        tqdm.write(f"Saved: {out} ({cli.win_w}x{cli.win_h} px)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3D surface render of one segmented rock .raw volume")
    ap.add_argument("--rock_dir", type=str, default=os.path.join("rock_data", "B1", "size_1200", "skin_0", "size_600"))
    ap.add_argument("--raw_file", type=str, default=None, help="Explicit .raw path (overrides glob)")
    ap.add_argument("--glob", type=str, default="Rock_600_*.raw", dest="glob_pat", help="First match in rock_dir")
    ap.add_argument("--shape", type=int, nargs=3, default=[600, 600, 600])
    ap.add_argument("--dtype", type=str, default="int8", choices=("int8", "uint8", "uint16"))
    ap.add_argument("--solid_value", type=float, default=1.0)
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--out", type=str, default=os.path.join("figures", "B1_R600_3d.png"))
    ap.add_argument("--win_w", type=int, default=2400)
    ap.add_argument("--win_h", type=int, default=2200)
    ap.add_argument("--smooth_iter", type=int, default=25)
    ap.add_argument("--color", type=str, default="tan")
    ap.add_argument("--backend", type=str, default="vtk", choices=("vtk", "matplotlib"),
                    help="vtk: PyVista (needs OpenGL/Mesa). matplotlib: CPU only, no GL")
    ap.add_argument("--mc_cap", type=int, default=128,
                    help="Max axis size before stride downsample (matplotlib backend only)")
    ap.add_argument("--no-xvfb", dest="use_xvfb", action="store_false", default=True,
                    help="Do not start Xvfb (vtk backend only)")
    ap.add_argument("--msaa", action="store_true", default=False, help="MSAA (vtk backend only)")
    VolumeSurfaceExport.run(ap.parse_args())
