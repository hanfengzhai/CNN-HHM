# CNN-HHM

Anonymous source release for the CNN–HHM workflow: a 3D CNN surrogate inside hierarchical homogenization (HHM) for fast elastic moduli of large digital rocks.

A large segmented scan is split into $75^3$ subcubes. The CNN predicts each subcube's elasticity. HHM then upscales those predictions to the full rock.

Three training targets share one convolutional backbone:

| Flag | Target |
| --- | --- |
| (default) | Model 1: full $6\times 6$ stiffness |
| `--homogenization` | Model 2: isotropic $(K,G)$ |
| `--hs_bound` | Model 3: Hashin–Shtrikman normalized factors |

The manuscript recommends Model 3 as the default for speed and HS consistency. A fourth option, `--benchmark_model` (with `--hs_bound`), reproduces Benchmark 1 from prior work.

## Setup

Python 3.10+, [PyTorch](https://pytorch.org/), NumPy, SciPy, matplotlib, tqdm. For the FFT coarse solve, also `atomman` (see `environment.yml`).

```bash
git clone <this-repository-url>
cd CNN-HHM
conda env create -f environment.yml
conda activate fft_solver
pip install torch
```

### Training data and model weights

Trained weights, paper figures, and $75^3$ scans ship in `anonymous_assets/` (zip archives; the models archive is split into `.partXX` files under 95 MB). Unpack after cloning:

```bash
# pretrained checkpoints → model/*.pt
python download_data.py --assets models

# optional: paper PNG trees (figures/, fig_*)
python download_data.py --assets figures

# optional: one rock of size-75 training scans (+ stiffness.dat)
python download_data.py --assets size75 --rocks B1

# optional: OT size-100 rocks used by some super-resolution demos
python download_data.py --assets ot
```

| Archive | Contents |
| --- | --- |
| `CNN-HHM_models` | `model/cnn3d_rock_*.pt` (Models 1–3 + Benchmark 1, all rocks) |
| `CNN-HHM_figures` | `figures/`, `fig_{B1,B2,CG,FB1,FB2}/` |
| `CNN-HHM_size75_<ROCK>` | `size75/<ROCK>/images` + `stiffness.dat` |
| `CNN-HHM_rock_data_OT` | `rock_data/{HR,LR}_dry_OT` |

After unpacking `size75`, train with:

```bash
python3 train_cnn.py \
  --scan_dir size75/B1/images \
  --dat_file size75/B1/stiffness.dat \
  --scan_shape 75 75 75 \
  --rock_type B1 --hs_bound --model_dir model
```

## Train

Point `--scan_dir` at a folder of `Rock_75_*.raw` subcubes and `--dat_file` at the matching DNS stiffness table.

```bash
python3 train_cnn.py \
  --scan_dir /path/to/images \
  --dat_file /path/to/stiffness.dat \
  --scan_shape 75 75 75 \
  --epochs 100 --batch_size 8 --lr 1e-3 \
  --train_ratio 0.8 --early_stop \
  --rock_type B1 --hs_bound
```

Rocks: `B1`, `B2`, `CG`, `FB1`, `FB2`. Run all three targets with `bash train_cnn.sh`.

## Evaluate CNN-HHM

```bash
python3 eval_cnn_hhm_moduli.py --largecube_size 300 --hs_bound --rock_type B1
```

`--largecube_size` is `300` or `600` (crops from the $900^3$ parent). `eval_cnn_hhm.sh` loops over rocks, sizes, and targets.

Coarse-scale FFT homogenization of a tabulated stiffness field:

```bash
python3 fft_homogenization.py -r B1 -m OT -S 800 -s 100 -p 0
```

## Layout

| Path | Role |
| --- | --- |
| `train_cnn.py` | Train or evaluate the 3D CNN |
| `eval_cnn_hhm_moduli.py` | CNN + HHM on large crops |
| `fft_homogenization.py` | Coarse FFT elasticity solve |
| `download_data.py` | Assemble split archives and unpack `anonymous_assets/` |
| `anonymous_assets/` | Zipped models, figures, and size-75 training data |
| `util_*.py` | Data, CNN, FFT, plotting |
| `data/`, `data_hhm/` | Stiffness and moduli tables |
| `model/` | Checkpoints (after `download_data.py`) |
| `size75/` | Training scans (after `download_data.py`) |
