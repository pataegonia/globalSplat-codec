# Codec RD experiment ledger

This file records compact, reproducible results. Raw per-scene outputs and SLURM logs remain in the referenced output directories; they are not duplicated here.

## Scope and invariant baseline

Objective: improve bitstream rate versus rendered quality for frozen GlobalSplat scene latents. Unless noted otherwise, experiments use the RE10K 32K checkpoint, context 12, geometry/texture tensors `[1,4096,512]`, and full rendering evaluation.

The selected baseline is G9/T7 G32/T32: geometry 9-bit, texture 7-bit, symmetric scalar quantization with per-scene/per-channel max scales; separate fixed 32-group CDF16 entropy models; CompressAI rANS. Mean over 5,601 held-out scenes is 3,331,072.3 bytes, PSNR 29.39492, SSIM 0.901298, LPIPS 0.122892.

## E0: Baseline codec and entropy grouping

The fixed CDF profile was calibrated on 1,000 training scenes. Integer decode was exact for all 5,601 evaluated scenes.

| Variant | Mean bytes/scene | Saving vs original |
|---|---:|---:|
| Adaptive K1, JSON scales | 3,447,609 | 0% |
| Adaptive K1, binary scales | 3,430,128 | 0.507% |
| Fixed K1 | 3,431,744 | 0.460% |
| Fixed G32/T16 | 3,332,292 | 3.345% |
| Fixed G32/T32 (selected) | 3,331,072 | 3.380% |

Selected file breakdown: texture rANS 1,397,047 bytes, geometry rANS 1,929,271, scale side information 4,096, remaining container metadata about 658. See `docs/latent_codec_final.md` and `slurm/evaluation/eval_latent_codec_final_re10k.sbatch`.

## E1: Channel-wise rendering sensitivity

Each 16-channel group (32 per stream) was independently coarsened while the frozen decoder rendered calibration and held-out scenes. The ranking generalized: calibration/test PSNR-rank Spearman was 0.792 for geometry and 0.894 for texture. Maximum-to-median sensitivity was about 73.5x for geometry and 69.5x for texture; the four most sensitive groups accounted for about 77.5% and 76.5% of positive impact, respectively. Latent-MSE versus PSNR-sensitivity Pearson correlation was weak (0.316 geometry, 0.420 texture).

Notable held-out groups: geometry channels 80-95 and 496-511 were especially sensitive; texture groups included channels 0-15, 240-255, and 176-191. Conclusion: group-dependent bit allocation is justified, but must use rendered distortion rather than latent MSE.

Scripts: `scripts/analysis/analyze_channel_sensitivity.py`, `scripts/analysis/codec_diagnostic_utils.py`. SLURM: `slurm/analysis/codec_diagnostics_re10k.sbatch`.

## E2: Linear transform-coding potential

PCA/KLT statistics were fit on 200 calibration scenes and evaluated on 200 held-out scenes. Covariance is concentrated (geometry: first 8 PCs explain 96.1%; texture: first 32 explain 88.2%), but the current per-channel maximum-scale quantizer makes transformed entropy worse.

| Transform | Estimated combined rate change vs identity |
|---|---:|
| Block PCA 16 | +2.24% |
| Block PCA 32 | +3.30% |
| Block PCA 64 | +4.21% |
| Full PCA | +4.29% |

Full PCA reduced quantization MSE only modestly (texture 3.45%, geometry 7.81%). Decision: reject vanilla PCA/KLT under the current quantizer/entropy model; covariance concentration alone is insufficient evidence of RD gain.

Script: `scripts/analysis/analyze_transform_coding.py`.

## E3: Geometry-to-texture linear prediction

OLS/ridge predictors were fit on 200 calibration scenes and tested on 200 held-out scenes, including reconstructed geometry as decoder-available input. OLS reduced texture variance by about 40%, but actual rANS texture payload increased 0.163%; ridge was worse. Decision: reject lightweight linear predictive coding and do not use residual variance as a rate proxy.

Script: `scripts/analysis/analyze_geo_texture_prediction.py`.

## E4: Manual decoder-aware group-variable quantization

For group `g`, `Delta_g = Delta_base_g * m_g`. Multipliers are fixed across scenes, so no per-scene table is transmitted. The entropy coder is unchanged. Groups contain 16 contiguous channels.

| Policy/ablation | Rate reduction | PSNR drop | SSIM drop | LPIPS increase |
|---|---:|---:|---:|---:|
| Conservative geometry | 2.197% | 0.00953 | - | - |
| Conservative texture | 2.456% | 0.00341 | - | - |
| Conservative joint | 4.659% | 0.01293 | 0.000226 | 0.000252 |
| Moderate geometry | 3.585% | 0.03590 | - | - |
| Moderate texture | 3.989% | 0.01078 | - | - |
| Moderate joint | 7.580% | 0.04652 | 0.000792 | 0.000796 |

Texture delivered better marginal rate savings at lower distortion. The original Aggressive policy is invalid: a 0.75x tier reduced step size without expanding the symbol range, causing clipping and a 3.61193 dB joint PSNR loss. All subsequent optimized profiles enforce `m >= 1.0`.

Files: `config/codec/group_variable_sensitivity_v1.json`, `config/group_quantization/group_variable_g9_t7.yaml`, `docs/group_variable_quantization.md`, `slurm/evaluation/eval_group_variable_quant_re10k.sbatch`, `scripts/analysis/summarize_group_quant_rd.py`.

## E5: Principled group-level RD allocation

On calibration scenes, every geometry and texture group was independently measured at multipliers 1, 1.25, 1.5, 1.75, 2, 2.5, and 3. The discrete optimizer used actual rANS payload changes and downstream PSNR/SSIM/LPIPS changes. Final tables were frozen and evaluated on the strictly held-out 5,601-scene set.

| Policy | Rate reduction | PSNR drop | SSIM drop | LPIPS increase | Pareto |
|---|---:|---:|---:|---:|:---:|
| Baseline | 0% | 0 | 0 | 0 | yes |
| Conservative joint | 4.659% | 0.01293 | 0.000226 | 0.000252 | no |
| Target 5% | 4.947% | 0.01168 | 0.000190 | 0.000237 | yes |
| Target 7.5% | 7.412% | 0.03072 | 0.000509 | 0.000689 | yes |
| Moderate joint | 7.580% | 0.04652 | 0.000792 | 0.000796 | yes |
| Target 10% | 9.859% | 0.08396 | 0.001407 | 0.001978 | yes |
| Target 12.5% | 12.339% | 0.42867 | 0.007268 | 0.010142 | yes mathematically; poor quality |
| Target 15% | 12.725% | 1.03195 | 0.018652 | 0.020871 | saturated; reject |

The calibration additive model predicted the following counts of groups left at 1.0x (geometry/texture): Target 5, 18/11; Target 7.5, 11/6; Target 10, 5/2; Target 12.5 and 15, 0/0. All profiles had a maximum multiplier of 3.0. The 15% target was infeasible even in the additive model. Held-out distortion confirms a sharp RD knee after roughly 10% saving.

Commands and artifacts:

```bash
bash slurm/submit.sh slurm/analysis/measure_group_quant_rd_re10k.sbatch
bash slurm/submit.sh slurm/evaluation/eval_optimized_group_quant_re10k.sbatch
python scripts/analysis/summarize_optimized_group_quant_rd.py \
  --input-root outputs/optimized_group_quant/heldout \
  --optimized-profile outputs/optimized_group_quant/optimized_profile.json \
  --output outputs/optimized_group_quant/heldout/comparison.json
```

The cluster holds `calibration_operating_points.json`, `optimized_profile.json`, held-out `comparison.json`, and per-policy `codec_summary.json` / `scores_all_avg.json`. The exact multiplier tables must be copied locally for durable reproducibility.

## E6: Codec-input token/raster decorrelation

Evaluated 200 RE10K context-12 scenes. Adjacent-token correlation was essentially the shuffled baseline across lags 1-256. Channel correlation remains non-negligible, so this rejects local prediction, not every possible transform.

| Stream | Raw entropy (bits/value) | 1-D delta | Change | Linear 2-D predictor | Change |
|---|---:|---:|---:|---:|---:|
| Texture | 5.396 | 5.875 | 8.9% worse | 6.403 | 18.6% worse |
| Geometry | 7.662 | 8.047 | 5.0% worse | 8.572 | 11.9% worse |

Artifacts: `ctx12/REPORT.md`, `ctx12/summary.json`, `ctx12/decorrelation_summary.{png,pdf}` and companion CSV/heatmap files. Scripts: `scripts/analysis/analyze_codec_decorrelation.py`, `scripts/analysis/plot_codec_decorrelation_report.py`. SLURM: `slurm/analysis/codec_decorrelation_re10k.sbatch`.

## E7: Final narrow allocation sweep

The unresolved RD knee was evaluated at nominal 8.0%, 8.5%, 9.0%, and
9.5% targets. The existing 50-scene `calibration_operating_points.json` was
reused exactly; latents were not recollected. All four calibration-derived
profiles were frozen before evaluation, all multipliers were at least 1.0,
and every policy was evaluated once on the same 5,601-scene held-out RE10K
context-12 split.

| Policy | Bytes/scene | Rate reduction | PSNR | PSNR drop | SSIM | SSIM drop | LPIPS | LPIPS increase | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Target 8% | 3,068,035.5 | 7.896% | 29.36017 | 0.03474 | 0.900722 | 0.000576 | 0.123687 | 0.000795 | yes |
| Target 8.5% | 3,051,753.9 | 8.385% | 29.35187 | 0.04305 | 0.900583 | 0.000715 | 0.123896 | 0.001004 | yes |
| Target 9% | 3,035,267.9 | 8.880% | 29.34185 | 0.05306 | 0.900417 | 0.000881 | 0.124175 | 0.001283 | yes |
| Target 9.5% | 3,018,838.5 | 9.373% | 29.32801 | 0.06691 | 0.900179 | 0.001119 | 0.124474 | 0.001581 | yes |

Target 8.5% is the maximum measured rate reduction satisfying PSNR loss
`<= 0.05 dB`: 8.385239% actual saving with 0.0430477 dB PSNR loss,
0.00071524 SSIM loss, and 0.00100398 LPIPS increase. Target 9% is the first
measured point above the constraint at 0.0530643 dB loss.

At the selected point, geometry/texture payloads are 1,815,148.3 and
1,231,676.6 bytes/scene; geometry/texture latent MSEs are
`1.735788e-05` and `1.104206e-04`; encode/decode times are 0.61980 and
0.60255 seconds/scene. It saves 0.973 percentage points more than optimized
Target 7.5%, at an additional 0.01233 dB PSNR loss. Relative to manual
Moderate, it saves 0.805 percentage points and improves PSNR by 0.00347 dB
and SSIM by 0.0000766, but LPIPS is worse by 0.0002079; therefore neither
strictly dominates the other across all four RD metrics.

Artifacts: `heldout/comparison.json`,
`slurm/evaluation/optimized_group_quant_narrow_8_9p5.json`,
`slurm/evaluation/eval_optimized_group_quant_narrow_re10k.sbatch`, and
`scripts/analysis/summarize_narrow_group_quant_rd.py`.

## Scalar-stage final decision

Use Target 5% for near-lossless operation and Target 8.5% when maximizing
compression under the 0.05 dB PSNR-loss constraint. Target 7.5% remains a
lower-distortion intermediate point, and Target 10% is only appropriate when
a 0.08396 dB loss is acceptable. Do not use sub-1.0 multipliers or the
high-rate-loss tail.

This is the final tuning result for decoder-aware scalar/group-variable
quantization. Do not continue tuning this quantizer. The next research stage
is a lightweight learned analysis/synthesis transform at the codec boundary,
with the pretrained GlobalSplat model frozen.

## E8: Learned-transform Stage A implementation (pending cluster smoke result)

The first learned-transform experiment is implemented but has not yet produced
a training result. Separate geometry/texture analysis and synthesis transforms
use identity-initialized token-wise residual MLPs (`512 -> 256 -> 512`). The
four transforms contain 1,055,744 parameters. A training-only per-channel
factorized logistic rate model adds 2,048 parameters, for 1,057,792 trainable
codec parameters total. No pretrained GlobalSplat parameters are trainable.

The repository audit found no existing differentiable entropy-rate estimator.
The new estimator must show positive Pearson and Spearman correlation with
actual rANS payload on a training/calibration subset before the smoke test can
update parameters. Identity-initialized `R0` and rendered `D0` determine
`lambda0=R0/D0`; the exact 0.25x/0.5x/1x/2x/4x values are saved at runtime.

The training path uses G9/T7 per-scene/per-channel max quantization with all
group multipliers fixed at 1.0, STE rounding for training reconstruction, hard
rounding for validation, and `R + lambda * 255^2 * RGB_MSE`. The slot encoder
runs without gradients; the frozen Gaussian decoder/renderer remain in the
autograd graph. Calibration and validation reserve disjoint training scenes;
the held-out test split is not used for training or model selection.

Stage A is a 300-step, one-GPU smoke job. Later SLURM stages are gated as
2,000-step five-lambda pilot, 10,000-step best-three screening, and up to
40,000-step best-two training. Each final checkpoint gets a newly calibrated
K=32 transformed-symbol entropy profile from 1,000 training scenes before the
same 5,601-scene actual-rANS evaluation.

Files: `globalsplat/compression/learned_transform.py`,
`scripts/learned_codec/`, `slurm/learned_codec/`,
`tests/test_learned_transform_codec.py`, and
`docs/learned_latent_transform.md`.

## Current decision and next measurement

Submit only the Stage A 300-step central-lambda smoke job. It must record
finite `R0`, `D0`, the exact lambda sweep, positive rate/rANS Pearson and
Spearman correlations, transform gradients, no pretrained gradients, finite
optimization, hard validation quantization, and successful checkpoint
round-trip. Do not submit the five-lambda pilot until every smoke gate passes.
