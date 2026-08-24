# Decoder-aware group-variable quantization experiment

This experiment keeps the pretrained GlobalSplat encoder and Gaussian decoder
frozen and keeps the selected G9/T7, fixed G32/T32 rANS entropy model unchanged.
Only the scalar quantization step is changed. Channels are divided into 32
contiguous groups of 16 independently for geometry and texture.

The checked-in profile `config/codec/group_variable_sensitivity_v1.json` records
the held-out rendering PSNR sensitivity of every group. Tiers are distribution
derived per stream: critical is at least 20x median sensitivity, high is at
least 5x median, medium is at least median, and the remaining groups are low.

| Policy | Critical | High | Medium | Low |
|---|---:|---:|---:|---:|
| Conservative | 1.0x | 1.0x | 1.25x | 1.5x |
| Moderate | 1.0x | 1.25x | 1.5x | 2.0x |
| Aggressive | 0.75x | 1.0x | 1.75x | 2.5x |

For every policy the SLURM array evaluates geometry-only, texture-only, and
joint allocation. Task zero is the exact profile-free G9/T7 G32/T32 baseline.

```bash
bash slurm/submit.sh slurm/evaluation/eval_group_variable_quant_re10k.sbatch
```

After all ten array tasks finish:

```bash
python scripts/analysis/summarize_group_quant_rd.py \
  --input-root outputs/group_variable_quant_rd \
  --profile config/codec/group_variable_sensitivity_v1.json \
  --output outputs/group_variable_quant_rd/comparison.json
```

The comparison contains actual total and per-stream rANS sizes, bits/value,
PSNR, SSIM, LPIPS, latent MSE, total encode/decode time, baseline deltas,
expanded group multiplier tables, and nondominated joint operating points.
