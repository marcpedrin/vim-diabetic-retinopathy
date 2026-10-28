# Vision Mamba (Vim) for Diabetic Retinopathy

From-scratch implementation of Vision Mamba (arXiv 2401.09417) in plain PyTorch
for 5-class diabetic retinopathy classification.

Rules: only `torch`, `torchvision`, `numpy`, `pandas`, `pillow`, `matplotlib`,
`tqdm`, `pyyaml` (and `pytest`/`scikit-learn` in tests only). No `mamba_ssm`,
no `timm`, no pretrained weights.

## Structure

- `vim/` - model code (package)
- `data/` - datasets / loading
- `train/` - training scripts
- `eval/` - evaluation scripts
- `configs/` - experiment configs
- `tests/` - tests

## Progress

- [x] Phase 0: `PatchEmbed` (B, 3, 128, 128) -> (B, 64, 192)
- [x] Phase 1: repo scaffolding (config, seeding, logging, checkpoints)
- [x] Phase 2: class token and position embeddings
- [x] Phase 3: selective scan
- [x] Phase 4: bidirectional Vim block
- [ ] Phase 5: full model
- [ ] Phase 6: data pipeline
- [ ] Phase 7: training
- [ ] Phase 8: evaluation and figures
- [ ] Phase 9: report and reading guide

See `PROGRESS.md` for what each phase built and what its tests prove.

## Run tests

```
pip install -r requirements.txt
python -m pytest tests -s
```
