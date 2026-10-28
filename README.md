# Vision Mamba (Vim) for Diabetic Retinopathy

A from-scratch PyTorch implementation of **Vision Mamba**, applied to 5-class
diabetic retinopathy (DR) grading of retinal fundus photographs.

> Zhu, Liao, Zhang, Wang, Liu, Wang. *Vision Mamba: Efficient Visual Representation
> Learning with Bidirectional State Space Model.* ICML 2024.
> [arXiv:2401.09417](https://arxiv.org/abs/2401.09417)

The model is trained on **EyePACS** (Kaggle 2015) and evaluated on a held-out,
patient-disjoint EyePACS test split and on the external **Messidor-2** dataset,
which is never used for training.

Every part of the model, including the selective scan, is written out explicitly.
No `mamba_ssm`, `causal_conv1d`, `timm` or `einops`, no existing Vim/Mamba code,
no pretrained weights, no ready-made dataset loaders and no ready-made metrics
(outside the tests). The code is heavily commented: each module states which
section, equation or line of Algorithm 1 of the paper it implements.

## Repository layout

```
configs/default.yaml      every hyperparameter, each with the paper's value beside it
vim/                      the model (one file per part of the paper)
  config.py               YAML -> dataclasses, derived values (J=64, M=65, E=384)
  patch_embed.py          Sec. 3.2, Eq. 4: image -> 64 patch tokens (x_p W)
  tokens.py               Sec. 3.2, Eq. 4: middle class token + position embeddings
  selective_scan.py       Sec. 3.1, Eqs. 1-2, Algorithm 1 lines 13-22
  vim_block.py            Sec. 3.3, Algorithm 1 (all 28 lines), bidirectional block
  vision_mamba.py         Sec. 3.2/3.4: the full classifier, parameter count
data/                     data pipeline (written from scratch)
  transforms.py           fundus crop/pad/resize/normalise + train augmentation
  eyepacs.py              EyePACS Dataset (trainLabels.csv)
  messidor2.py            Messidor-2 Dataset (EVALUATION ONLY)
  splits.py               patient-level, stratified, seeded train/val/test split
  synthetic.py            fake fundus images for tests and the smoke test
  __init__.py             build_dataloaders(), build_messidor_loader()
train/
  utils.py                set_seed, AverageMeter, checkpoints, CSV Logger
  engine.py               train_one_epoch(), validate()
  train.py                main training script (python -m train.train)
eval/
  metrics.py              accuracy, P/R/F1, confusion matrix, QWK (NumPy)
  evaluate.py             EyePACS test + Messidor-2 evaluation (python -m eval.evaluate)
  plots.py                report figures (python -m eval.plots)
tests/                    63 tests: shapes, hand-computed cases, gradients, data, metrics
report/main.tex           the LaTeX report; references.bib; figures/ (generated)
PROGRESS.md               what each build phase added and what its tests prove
```

## Install

Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

For a CPU-only laptop, install the smaller CPU build of PyTorch first:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Run the tests

```
python -m pytest tests -q          # about 30 s on a laptop CPU
python -m pytest tests -s          # the same, showing the printed shapes and values
```

## Smoke test (do this before using Kaggle GPU hours)

```
python -m train.train --smoke-test
```

Runs the whole pipeline (config, seeding, data, model, loss, optimiser, schedule,
logging, checkpointing) for 2 epochs on 100 images with the full 12-block model.
If `datasets/eyepacs/trainLabels.csv` is not present, it writes 100 synthetic fundus
images first, so it works on any machine (about a minute on a laptop CPU). Output
goes to `outputs/default_smoke/`, never to the real run's folder.

## Data

Download the datasets yourself (their licences forbid redistribution) and either
place them as below or point the config at them:

```
datasets/eyepacs/train/*.jpeg            datasets/eyepacs/trainLabels.csv
datasets/messidor2/images/*              datasets/messidor2/messidor_data.csv
```

`datasets/` is in `.gitignore`. The Messidor-2 CSV column names are config settings
(`data.messidor_id_column`, `data.messidor_label_column`).

## Train

```
python -m train.train --config configs/default.yaml
```

Any setting can be changed without editing the YAML, e.g.
`--override train.epochs=5 --override train.lr=5e-4`.

**On Kaggle** (GPU: T4 recommended; if you get a "no kernel image" CUDA error on a
P100, switch the accelerator to T4), attach an EyePACS dataset with extracted
`.jpeg` images and `trainLabels.csv`, then in a notebook cell:

```
!git clone <this repository> vim && cd vim && python -m train.train \
    --override data.eyepacs_image_dir=/kaggle/input/<dataset>/train \
    --override data.eyepacs_labels_csv=/kaggle/input/<dataset>/trainLabels.csv \
    --override train.output_dir=/kaggle/working/outputs/default
```

Time one epoch first (`--override train.epochs=1`) to check the full run fits in
the session limit. Each run writes `best.pt` (best validation QWK), `last.pt`,
`log.csv` and a copy of the config into the output folder.

## Evaluate and make the figures

```
python -m eval.evaluate --config configs/default.yaml     # EyePACS test + Messidor-2
python -m eval.plots    --config configs/default.yaml     # PNGs into report/figures/
```

`evaluate.py` prints a metrics table and writes `outputs/default/results.json` and
per-image prediction CSVs; Messidor-2 is skipped with a message if it is absent.
(On Kaggle, add the same `--override` paths as for training, plus
`--override eval.checkpoint=/kaggle/working/outputs/default/best.pt` and
`--override eval.results_json=/kaggle/working/outputs/default/results.json`.)

## Build the report

```
cd report
latexmk -pdf main.tex        # or: pdflatex main, bibtex main, pdflatex main (x2)
```

Or upload the `report/` folder to Overleaf. Missing figures are drawn as labelled
placeholder boxes, so the report compiles before training. Places that need numbers
from the real run are marked `% TODO: fill in after training`.

## Configuration: ours versus the paper (Vim-Ti)

| Setting | Paper | Ours | Why |
|---|---|---|---|
| Image size | 224 | 128 | compute on one GPU (approved) |
| Patch size / patches | 16 / 196 | 16 / 64 | follows from the image size |
| Embedding dim D | 192 | 192 | same |
| Expanded dim E | 384 | 384 | same (expand 2) |
| SSM state dim N | 16 | 16 | same |
| Depth L | 24 | 12 | slow Python-loop scan (approved) |
| Conv1d kernel | 4 | 4 | same |
| Class token | middle | middle (index 32) | paper's best (Table 5) |
| Classes | 1000 | 5 | DR grades 0-4 |
| Epochs | 300 | 30 | limited GPU hours |
| Batch size | 1024 | 32 | one 16 GB GPU |
| Optimiser / lr / wd | AdamW / 1e-3 / 0.05 | AdamW / 1e-3 / 0.05 | same |
| Schedule | cosine | 3-epoch warmup + cosine | short run |
| Augmentation | crop, flip, mixup, erasing | flips, rotation, mild jitter | keeps tiny lesions intact |
| EMA | yes | no | not needed for our goal |
| Parameters | ~7M | 6,857,093 | full E x E Delta projection, see below |

Our model has about as many parameters as Vim-Ti despite half the depth, because
we follow the printed Algorithm 1, which projects Delta with a full 384 x 384
matrix; the official code uses a low-rank factorisation. See
`vim/vision_mamba.py` (`count_parameters`) for the arithmetic.

## What is from scratch and what comes from PyTorch

**Written from scratch in this repository:** patch tokenisation into a sequence,
the class token and position embeddings, the discretisation (A_bar, B_bar), the
selective scan recurrence, the bidirectional block with separate weights per
direction, the S4D-real and Delta initialisations, the full model, the fundus
cropping and padding, the EyePACS and Messidor-2 datasets, the patient-level split,
the training and validation loops, the learning-rate schedule function, the class
weighting, and every metric (accuracy, precision, recall, F1, confusion matrix,
quadratic weighted kappa).

**Used from PyTorch / torchvision (generic building blocks only):** tensors and
autograd; `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, `nn.LayerNorm`, `nn.Parameter`;
the SiLU and softplus functions; cross-entropy loss; the AdamW optimiser and the
`LambdaLR` wrapper; mixed precision (`autocast`, `GradScaler`); `Dataset` and
`DataLoader`; torchvision image transforms (resize, flips, rotation, colour
jitter, normalise).

scikit-learn appears only in `tests/test_metrics.py`, to verify the from-scratch
metrics.

## Reproducibility

One seed (`seed: 42`) drives everything: Python, NumPy, PyTorch CPU and CUDA
generators, deterministic cuDNN, seeded DataLoader shuffling and worker seeding,
and the patient split. Each run saves its exact config next to its checkpoints,
and checkpoints store the config used to build them, so evaluation always
rebuilds the same model and the same split.

## Progress

- [x] Phase 0: `PatchEmbed` (B, 3, 128, 128) -> (B, 64, 192)
- [x] Phase 1: repo scaffolding (config, seeding, logging, checkpoints)
- [x] Phase 2: class token and position embeddings
- [x] Phase 3: selective scan
- [x] Phase 4: bidirectional Vim block
- [x] Phase 5: full model
- [x] Phase 6: data pipeline
- [x] Phase 7: training
- [x] Phase 8: evaluation and figures
- [x] Phase 9: report and reading guide
- [ ] Train on Kaggle, run evaluation and plots, fill in the report's TODOs

See `PROGRESS.md` for what each phase built and what its tests prove.
