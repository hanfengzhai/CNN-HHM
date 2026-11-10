#!/usr/bin/env python3
"""Build Google Drive upload zips for CNN-HHM (models, figures, size-75 scans).

Maintainer tool. Expects local paths via env vars or defaults under this repo.
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

# This repository root (standalone GitHub layout).
CNN = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CNN_HHM_DRIVE_OUT", CNN.parent / "drive_upload"))
SIZE75_ROOT = Path(
    os.environ.get(
        "CNN_HHM_SIZE75_ROOT",
        "/path/to/digital_rock/machine_learning/bulk_shear_learning/size_75/data",
    )
)
ROCKS = ("B1", "B2", "CG", "FB1", "FB2")
PRIV_ROCK_DATA = Path(
    os.environ.get(
        "CNN_HHM_ROCK_DATA",
        "/path/to/Digital_Rock/geomechanics/FFT_python/real_rock_parameters/multiscale/rock_data",
    )
)


def _add_tree(zf: zipfile.ZipFile, src: Path, arc_prefix: str, compress=True) -> int:
    """Add files under src into zip with arc_prefix/. Returns file count."""
    n = 0
    if not src.exists():
        raise FileNotFoundError(src)
    comp = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    if src.is_file():
        zf.write(src, arcname=f"{arc_prefix}/{src.name}", compress_type=comp)
        return 1
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        if path.name.endswith((".pyc",)) or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(src)
        zf.write(path, arcname=f"{arc_prefix}/{rel.as_posix()}", compress_type=comp)
        n += 1
        if n % 500 == 0:
            print(f"  … {n} files", flush=True)
    return n


def zip_models() -> Path:
    out = OUT / "CNN-HHM_models.zip"
    pts = sorted((CNN / "model").glob("cnn3d_rock_*.pt"))
    if not pts:
        raise SystemExit(f"No checkpoints in {CNN / 'model'}")
    print(f"Writing {out} ({len(pts)} checkpoints) …")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in pts:
            zf.write(p, arcname=f"CNN-HHM/model/{p.name}")
            print(f"  + {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
    print(f"Done {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def zip_figures() -> Path:
    out = OUT / "CNN-HHM_figures.zip"
    print(f"Writing {out} …")
    n = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("figures", "fig_B1", "fig_B2", "fig_CG", "fig_FB1", "fig_FB2"):
            src = CNN / name
            if src.is_dir():
                n += _add_tree(zf, src, f"CNN-HHM/{name}")
    print(f"Done {out}: {n} files, {out.stat().st_size / 1e6:.1f} MB")
    return out


def zip_size75_rock(rock: str) -> Path:
    """One rock: images/ + stiffness.dat (+ small .dat companions) under size75/<rock>/."""
    src = SIZE75_ROOT / f"data_{rock}" / "skin_0"
    if not src.is_dir():
        raise FileNotFoundError(src)
    out = OUT / f"CNN-HHM_size75_{rock}.zip"
    print(f"Writing {out} from {src} …")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        n = _add_tree(zf, src / "images", f"CNN-HHM/size75/{rock}/images")
        for fname in (
            "stiffness.dat",
            "porosity.dat",
            "bulk_mod.dat",
            "shear_mod.dat",
            "bulk_mod_HS_bound.dat",
            "shear_mod_HS_bound.dat",
            "bulk_HS_fraction.dat",
            "shear_HS_fraction.dat",
        ):
            f = src / fname
            if f.is_file():
                zf.write(f, arcname=f"CNN-HHM/size75/{rock}/{fname}")
                n += 1
    print(f"Done {out}: {n} entries, {out.stat().st_size / 1e9:.2f} GB")
    return out


def zip_ot_rock_data() -> Path | None:
    """Optional OT size-100 rocks used by super-resolution HHM demos."""
    if not PRIV_ROCK_DATA.is_dir():
        print(f"Skip OT rock_data (missing {PRIV_ROCK_DATA})")
        return None
    out = OUT / "CNN-HHM_rock_data_OT.zip"
    print(f"Writing {out} …")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        n = _add_tree(zf, PRIV_ROCK_DATA, "CNN-HHM/rock_data")
    print(f"Done {out}: {n} files, {out.stat().st_size / 1e6:.1f} MB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="+",
        choices=["models", "figures", "size75", "ot", "all"],
        default=["all"],
    )
    ap.add_argument("--rocks", nargs="+", default=list(ROCKS))
    args = ap.parse_args()
    only = set(args.only)
    if "all" in only:
        only = {"models", "figures", "size75", "ot"}
    OUT.mkdir(parents=True, exist_ok=True)
    if "models" in only:
        zip_models()
    if "figures" in only:
        zip_figures()
    if "ot" in only:
        zip_ot_rock_data()
    if "size75" in only:
        for rock in args.rocks:
            zip_size75_rock(rock)
    print("\nUpload these to Google Drive, set sharing to 'Anyone with the link',")
    print("then put each file id into download_data.py / README.md.")
    for p in sorted(OUT.glob("*.zip")):
        print(f"  {p.name:40s} {p.stat().st_size / 1e9:7.2f} GB")


if __name__ == "__main__":
    main()
