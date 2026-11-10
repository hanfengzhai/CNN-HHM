#!/usr/bin/env python3
"""Download CNN-HHM assets from Google Drive with gdown.

Shared folder:
  https://drive.google.com/drive/folders/1SLbUaF1tLPfc1mj1LKjx6g-MR1320nHR?usp=sharing

Usage (from this repository root):

  pip install gdown
  python download_data.py --assets models figures
  python download_data.py --assets size75 --rocks B1
  python download_data.py --list
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    sys.exit("Install gdown first:  pip install gdown")

# Standalone GitHub repo root (== directory of this file).
ROOT = Path(__file__).resolve().parent

# Shared Drive folder (Anyone with the link).
GDRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/"
    "1SLbUaF1tLPfc1mj1LKjx6g-MR1320nHR?usp=sharing"
)

# Keys → (filename, Google Drive file id). Override with env GDRIVE_FILE_ID_<KEY>.
ASSETS = {
    "models": (
        "CNN-HHM_models.zip",
        os.environ.get("GDRIVE_FILE_ID_MODELS", "10YD6-i4runn48kJsKSoL-RkABUzKYw28"),
    ),
    "figures": (
        "CNN-HHM_figures.zip",
        os.environ.get("GDRIVE_FILE_ID_FIGURES", "1aoWmRcliKxbMF3YN5PAdgZ4AqHjuEsdv"),
    ),
    "ot": (
        "CNN-HHM_rock_data_OT.zip",
        os.environ.get("GDRIVE_FILE_ID_OT", "1CSBaHZsw5O2U49w9eJaIXoQsRetOb1N8"),
    ),
    "size75_B1": (
        "CNN-HHM_size75_B1.zip",
        os.environ.get("GDRIVE_FILE_ID_SIZE75_B1", "1ztAcBblkescrFlykEV8iMdtucySzAmFB"),
    ),
    "size75_B2": (
        "CNN-HHM_size75_B2.zip",
        os.environ.get("GDRIVE_FILE_ID_SIZE75_B2", "1wpgQyMTFB3TbV1PaJmTBpWnzH4cfqjm3"),
    ),
    "size75_CG": (
        "CNN-HHM_size75_CG.zip",
        os.environ.get("GDRIVE_FILE_ID_SIZE75_CG", "1XWq7uWiv4AhkBxlJGF6lmRtTTGowuktL"),
    ),
    "size75_FB1": (
        "CNN-HHM_size75_FB1.zip",
        os.environ.get("GDRIVE_FILE_ID_SIZE75_FB1", "11ad10GVNL966L1SXZkoIHO-ZNMpnZBmd"),
    ),
    "size75_FB2": (
        "CNN-HHM_size75_FB2.zip",
        os.environ.get("GDRIVE_FILE_ID_SIZE75_FB2", "1M9NRC1zvuagU4ojAuFNZcsWauQqNMtF6"),
    ),
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
    """Map zip member path into this repo. Zips use a CNN-HHM/ prefix."""
    name = name.replace("\\", "/")
    if name.endswith("/"):
        return None
    if name.startswith("CNN-HHM/"):
        name = name[len("CNN-HHM/") :]
    if not name or name.startswith(".."):
        return None
    return ROOT / name


def download_and_extract(key: str, quiet: bool = False) -> None:
    if key not in ASSETS:
        raise SystemExit(f"Unknown asset key {key!r}. Try --list")
    name, file_id = ASSETS[key]
    if file_id.startswith("PLACEHOLDER"):
        raise SystemExit(
            f"{key}: {name} is not on Drive yet (or id missing).\n"
            f"  Upload it to {GDRIVE_FOLDER}\n"
            f"  then put the file id in download_data.py ASSETS[{key!r}] "
            f"(or set env GDRIVE_FILE_ID_{key.upper()})."
        )
    url = f"https://drive.google.com/uc?id={file_id}"
    dest_zip = ROOT / name
    print(f"Downloading {name} …")
    gdown.download(url, str(dest_zip), quiet=quiet)
    if not dest_zip.is_file():
        raise SystemExit(f"Download failed: {dest_zip}")
    print(f"Extracting into {ROOT} …")
    with zipfile.ZipFile(dest_zip, "r") as zf:
        for info in zf.infolist():
            dest = _member_dest(info.filename)
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as out:
                out.write(src.read())
    dest_zip.unlink()
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
    ap.add_argument("--list", action="store_true", help="List asset keys and ids")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, (name, fid) in ASSETS.items():
            print(f"{k:12s}  {name:32s}  id={fid}")
        return

    for key in resolve_keys(args.assets, args.rocks):
        download_and_extract(key, quiet=args.quiet)


if __name__ == "__main__":
    main()
