# Vision Mamba (Vim) for Diabetic Retinopathy

From-scratch implementation of Vision Mamba (arXiv 2401.09417) in plain PyTorch
for 5-class diabetic retinopathy classification.

Rules: only `torch`, `torchvision`, `numpy`. No `mamba_ssm`, no `timm`, no pretrained weights.

## Structure

- `vim/` - model code (package)
- `data/` - datasets / loading
- `train/` - training scripts
- `eval/` - evaluation scripts
- `configs/` - experiment configs
- `tests/` - tests

## Progress

- [x] Step 1: `PatchEmbed` (B, 3, 128, 128) -> (B, 64, 192)

## Run tests

```
pip install -r requirements.txt
python -m pytest tests -s
```
