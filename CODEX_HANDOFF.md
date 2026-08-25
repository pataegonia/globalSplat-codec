# Codex handoff: GlobalSplat latent codec RD research

## 1. Research objective

Improve codec rate-distortion (RD): reduce the scene-latent bitstream while preserving downstream rendered PSNR/SSIM/LPIPS, or improve rendering quality at a fixed rate. Peak GPU memory is not the research objective. The pretrained GlobalSplat network remains frozen.

## 2. Current codec architecture

For each scene, the dual-stream encoder produces geometry and texture latents (RE10K-32K: each `[1, 4096, 512]`). Each stream follows:

`latent -> symmetric scalar quantization -> integer symbols -> fixed grouped CDF16 + CompressAI rANS -> bitstream -> rANS decode -> inverse quantization -> Gaussian decoder`

Quantization uses a per-scene, per-channel maximum-derived scale. Geometry and texture are coded independently. The selected entropy profiles contain 32 groups of 16 channels per stream. Per-scene side information is two arrays of 512 little-endian FP32 scales (4,096 bytes total) plus container metadata; fixed CDFs, channel mappings, and group multiplier profiles are shared and not charged per scene.

## 3. Important repository/file locations

- Codec: `globalsplat/compression/arithmetic_codec.py`, `globalsplat/compression/__init__.py`
- Codec integration/evaluation: `globalsplat/model/globalsplat.py`, `globalsplat/model/model_wrapper.py`
- Model config: `config/model/globalsplat.yaml`
- Baseline description: `docs/latent_codec_final.md`
- Manual group allocation: `config/codec/group_variable_sensitivity_v1.json`, `config/group_quantization/group_variable_g9_t7.yaml`, `docs/group_variable_quantization.md`
- Optimized allocation: `scripts/analysis/measure_group_quant_operating_points.py`, `scripts/analysis/optimize_group_quant_allocation.py`, `scripts/analysis/summarize_optimized_group_quant_rd.py`, `docs/optimized_group_quantization.md`
- Tests: `tests/test_arithmetic_codec.py`
- Full experiment ledger: [experiments.md](experiments.md)

## 4. Baseline codec configuration

The exact baseline is G9/T7, G32/T32: 9-bit geometry, 7-bit texture, per-scene/per-channel maximum scale, raw FP32 scale serialization (`GSAC0002`), fixed train-calibrated 32-group CDF16 for each stream, and CompressAI rANS. The fixed profile was calibrated on 1,000 RE10K training scenes and is 83,968 bytes once per deployment.

Held-out 5,601-scene baseline: 3,331,072.3 bytes/scene; texture payload 1,397,047; geometry payload 1,929,271; scales 4,096; other container data about 658. Rendering: PSNR 29.39492, SSIM 0.901298, LPIPS 0.122892.

## 5. Experiments already completed

1. Baseline entropy grouping and binary scale serialization.
2. Channel/group rendering-sensitivity analysis on calibration and held-out scenes.
3. PCA/KLT transform-coding potential analysis.
4. Linear/ridge geometry-to-texture predictive coding analysis.
5. Manual Conservative/Moderate/Aggressive group-variable quantization, including geometry-only, texture-only, and joint ablations.
6. Calibration-measured discrete RD allocation at nominal 5%, 7.5%, 10%, 12.5%, and 15% rate targets, followed by held-out rendering evaluation.
7. Token-local/raster predictive-decorrelation diagnostic on 200 RE10K context-12 scenes.
8. Final narrow allocation sweep at nominal 8.0%, 8.5%, 9.0%, and 9.5% targets on the full 5,601-scene held-out split.
9. Lightweight learned-transform Stage A implementation and local synthetic validation; the cluster smoke result is pending.

See [experiments.md](experiments.md) for methods, tables, and artifact references.

## 6. Key numerical results

| Policy | Actual rate reduction | PSNR drop | SSIM drop | LPIPS increase |
|---|---:|---:|---:|---:|
| Target 5% | 4.947% | 0.01168 dB | 0.000190 | 0.000237 |
| Target 7.5% | 7.412% | 0.03072 dB | 0.000509 | 0.000689 |
| Target 8% | 7.896% | 0.03474 dB | 0.000576 | 0.000795 |
| Target 8.5% | 8.385% | 0.04305 dB | 0.000715 | 0.001004 |
| Manual Moderate joint | 7.580% | 0.04652 dB | 0.000792 | 0.000796 |
| Target 9% | 8.880% | 0.05306 dB | 0.000881 | 0.001283 |
| Target 9.5% | 9.373% | 0.06691 dB | 0.001119 | 0.001581 |
| Target 10% | 9.859% | 0.08396 dB | 0.001407 | 0.001978 |
| Target 12.5% | 12.339% | 0.42867 dB | 0.007268 | 0.010142 |
| Target 15% (infeasible model target) | 12.725% | 1.03195 dB | 0.018652 | 0.020871 |

## 7. Methods rejected and why

- Manual Aggressive allocation: invalid. It used multipliers below 1.0 while retaining the original symbol range, causing clipping and a 3.61 dB PSNR collapse.
- Vanilla PCA/KLT with the existing maximum-scale quantizer: despite covariance concentration, coded-rate estimates worsened by 2.24% to 4.29%; do not implement without jointly redesigning quantization/entropy coding.
- Linear/ridge geometry-to-texture prediction: OLS reduced variance by about 40% but increased actual rANS texture payload by 0.163%; variance was not a reliable rate proxy.
- Conventional token delta and artificial raster predictors: token correlations matched shuffled baselines; entropy increased 5.0% for geometry and 8.9% for texture with 1-D delta, and 11.9%/18.6% with linear Paeth-like prediction.
- Latent variance/MSE-only allocation: weak correlation with rendered sensitivity; downstream rendering metrics must drive allocation.

## 8. Current best operating points

- Near-lossless choice: optimized Target 5% (4.947% saving, 0.01168 dB PSNR loss); it dominates manual Conservative joint.
- Maximum-saving choice under the 0.05 dB PSNR-loss constraint: optimized Target 8.5% (8.385239% saving, 0.0430477 dB loss, 0.00071524 SSIM loss, 0.00100398 LPIPS increase).
- Lower-distortion balanced choice: optimized Target 7.5% (7.412% saving, 0.03072 dB loss).
- Higher-compression choice: optimized Target 10% (9.859% saving, 0.08396 dB loss) if that quality loss is acceptable.
- Avoid Target 12.5% and the saturated Target 15% for quality-sensitive use.

## 9. Current conclusions

Decoder-aware group-variable scalar quantization is validated and requires no neural retraining or per-scene multiplier side information. Texture generally offers cheaper marginal rate savings than geometry, while a small set of geometry groups requires protection. The exact measured maximum under the 0.05 dB PSNR-loss constraint is Target 8.5% at 8.385239% actual saving. Target 9% is already above the constraint at 0.0530643 dB loss. This closes scalar/group-variable quantizer tuning.

## 10. Exact next research stage

Do not continue tuning the scalar/group-variable quantizer. The learned-transform protocol is implemented. Submit only `slurm/learned_codec/train_transform_smoke_re10k.sbatch` next. Do not submit the lambda pilot unless the smoke output records positive estimated-rate/actual-rANS Pearson and Spearman correlations, finite forward/backward and rate/distortion terms, transform gradients, no pretrained gradients, hard-quantized validation, and checkpoint save/load success. The frozen Target 8.5% point is the primary `<= 0.05 dB` comparison baseline.

## 11. Important implementation constraints

- Keep every pretrained GlobalSplat parameter frozen. The Gaussian decoder and renderer must remain in the training autograd graph.
- Train only the four `512 -> 256 -> 512` residual transforms and the training-only entropy parameters.
- Keep the transformed representation at 512 channels and use G9/T7 per-scene/per-channel maximum scaling with all multipliers equal to 1.0.
- Do not reuse the original-basis Target 8.5% multiplier table in transformed space.
- Use rendered `255^2 * RGB_MSE` as distortion and estimated total bits per latent scalar as rate during training.
- Never use the 5,601-scene test set for lambda, checkpoint, entropy-model, or architecture selection.
- Recalibrate a new fixed K=32 profile on transformed training/calibration symbols for every final checkpoint, then freeze it before actual-rANS test evaluation.
- Do not add hyperpriors, autoregressive context, cross-stream prediction/attention, token attention, dimension reduction, PCA/KLT, or new GlobalSplat training in this first learned experiment.

## 12. Relevant experiment/log/config files

- Calibration SLURM: `slurm/analysis/measure_group_quant_rd_re10k.sbatch`
- Held-out SLURM: `slurm/evaluation/eval_optimized_group_quant_re10k.sbatch`
- Manual held-out SLURM: `slurm/evaluation/eval_group_variable_quant_re10k.sbatch`
- Expected cluster artifacts: `outputs/optimized_group_quant/calibration_operating_points.json`, `outputs/optimized_group_quant/optimized_profile.json`, `outputs/optimized_group_quant/heldout/comparison.json`, and per-policy `codec_summary.json` / `scores_all_avg.json`
- Decorrelation artifacts downloaded locally: `ctx12/REPORT.md`, `ctx12/summary.json`, plots/CSVs; logs `slurm-gs-codec-decorrelation-404703.{out,err}`
- Optimized held-out job logs reported on the cluster: `logs/slurm/slurm-gs-opt-group-rd-403979_*.{out,err}`
- Final narrow profile: `slurm/evaluation/optimized_group_quant_narrow_8_9p5.json`
- Final narrow held-out results and comparison: `heldout/`, `heldout/comparison.json`
- Final narrow evaluation/summarization: `slurm/evaluation/eval_optimized_group_quant_narrow_re10k.sbatch`, `scripts/analysis/summarize_narrow_group_quant_rd.py`
- Learned transform module/tests: `globalsplat/compression/learned_transform.py`, `tests/test_learned_transform_codec.py`
- Learned training/calibration/evaluation: `scripts/learned_codec/`
- Learned staged jobs: `slurm/learned_codec/`
- Learned protocol: `docs/learned_latent_transform.md`

## 13. Unresolved issues

- The prior optimized artifacts and full held-out outputs are preserved locally under `optimized_group_quant/`; the final narrow outputs are under `heldout/`.
- The optimizer's additive calibration model is approximate; held-out results, especially beyond 10%, diverge materially from predicted distortion.
- Generalization beyond RE10K context 12 (other context counts/datasets/checkpoints) has not been established.
- The learned-transform identity `R0`, `D0`, numeric lambda sweep, and rate/rANS correlations are not known until Stage A runs on the cluster.
- No learned-transform checkpoint has passed the smoke gates yet; Stage B and later jobs must remain unsubmitted until it does.
