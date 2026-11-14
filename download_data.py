#!/usr/bin/env python3
"""Assemble and unpack CNN-HHM training assets shipped in anonymous_assets/.

Large files are stored as zip archives (or split parts) inside this repository
so reviewers do not need an external cloud account.

Usage (from this repository root):

  python download_data.py --assets models figures
  python download_data.py --assets size75 --rocks B1
  python download_data.py --list
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "anonymous_assets"

# Keys → archive stem under anonymous_assets/ (without .zip).
# Split archives use stem.zip.part00, stem.zip.part01, ...
ASSETS = {
    "models": "CNN-HHM_models",
    "figures": "CNN-HHM_figures",
    "ot": "CNN-HHM_rock_data_OT",
    "size75_B1": "CNN-HHM_size75_B1",
    "size75_B2": "CNN-HHM_size75_B2",
    "size75_CG": "CNN-HHM_size75_CG",
    "size75_FB1": "CNN-HHM_size75_FB1",
    "size75_FB2": "CNN-HHM_size75_FB2",
}


def resolve_keys(assets: list[str], rocks: list[str]) -> list[str]:
    keys: list[str] = []
    for a in assets:
        if a == "size75":
            keys.extend(f"size75_{r}" for r in rocks)
        elif a == "all":
            keys.extend(["models", "figures", "ot"])
            keys.extend(f"size75_{r}" for r in rocks)
        else:
            keys.append(a)
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _member_dest(name: str) -> Path | None:
    """Map zip member path into this repo. Zips may use a CNN-HHM/ prefix."""
    name = name.replace("\\", "/")
    if name.endswith("/"):
        return None
    if name.startswith("CNN-HHM/"):
        name = name[len("CNN-HHM/") :]
    if not name or name.startswith(".."):
        return None
    return ROOT / name


def _locate_archive(stem: str) -> tuple[Path, list[Path]]:
    """Return (assembled zip path, part files). parts empty if a single zip exists."""
    whole = ASSET_DIR / f"{stem}.zip"
    if whole.is_file():
        return whole, []
    parts = sorted(ASSET_DIR.glob(f"{stem}.zip.part*"))
    if not parts:
        raise SystemExit(
            f"Missing archive for {stem!r} under {ASSET_DIR}. "
            "Clone the full repository (including anonymous_assets/)."
        )
    return ASSET_DIR / f"{stem}.zip", parts


def _assemble(stem: str) -> Path:
    dest, parts = _locate_archive(stem)
    if not parts:
        return dest
    print(f"Assembling {dest.name} from {len(parts)} parts …")
    with open(dest, "wb") as out:
        for part in parts:
            with open(part, "rb") as src:
                shutil.copyfileobj(src, out)
    return dest


def unpack(key: str) -> None:
    if key not in ASSETS:
        raise SystemExit(f"Unknown asset key {key!r}. Try --list")
    stem = ASSETS[key]
    zip_path = _assemble(stem)
    print(f"Extracting {zip_path.name} into {ROOT} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            dest = _member_dest(info.filename)
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as out:
                out.write(src.read())
    # Remove reassembled zip if it was built from parts (keep original wholes).
    parts = sorted(ASSET_DIR.glob(f"{stem}.zip.part*"))
    if parts and zip_path.is_file():
        zip_path.unlink()
    print(f"Done: {key}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--assets",
        nargs="+",
        default=["models"],
        help="models | figures | ot | size75 | all | size75_B1 | …",
    )
    ap.add_argument(
        "--rocks",
        nargs="+",
        default=["B1", "B2", "CG", "FB1", "FB2"],
        help="Used when --assets includes size75",
    )
    ap.add_argument("--list", action="store_true", help="List asset keys")
    args = ap.parse_args()

    if args.list:
        for k, stem in ASSETS.items():
            whole = ASSET_DIR / f"{stem}.zip"
            npart = len(list(ASSET_DIR.glob(f"{stem}.zip.part*")))
            status = "zip" if whole.is_file() else (f"{npart} parts" if npart else "MISSING")
            print(f"{k:12s}  {stem:28s}  {status}")
        return

    if not ASSET_DIR.is_dir():
        raise SystemExit(f"Missing {ASSET_DIR}. Full clone required.")

    for key in resolve_keys(args.assets, args.rocks):
        unpack(key)


if __name__ == "__main__":
    main()
