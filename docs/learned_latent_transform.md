# Lightweight learned latent transform

## Scope

This is the first post-scalar-codec experiment. The pretrained GlobalSplat
encoder, Gaussian decoder, and renderer are frozen. Only four token-wise
residual MLPs and a training-only factorized entropy model are optimized.

For each stream, analysis and synthesis use:

```text
output = input + Linear(256, 512)(GELU(Linear(512, 256)(LayerNorm(input))))
```

The final Linear layer is zero-initialized, so every transform begins as an
exact identity. Geometry and texture use separate analysis and synthesis
networks. The representation remains 512-dimensional.

The transform has 1,055,744 parameters. Two per-channel logistic training-rate
models add 2,048 parameters, for 1,057,792 trainable codec parameters total.
There is no hyperprior, autoregressive model, cross-stream prediction,
attention, PCA, or group-variable quantization.

## Quantization and rate

Training and inference retain geometry 9-bit / texture 7-bit symmetric scalar
quantization with per-scene/per-channel maximum scales. All group multipliers
are 1.0. The rendered-distortion path uses straight-through rounding during
training and hard rounding during validation.

No differentiable entropy model existed in the repository. Training therefore
uses independent per-channel logistic probability masses. Before any updates,
their moments are initialized from training/calibration symbols. The job must
measure positive Pearson and Spearman correlation between estimated rate and
actual fixed-profile rANS payload on the calibration subset or it aborts.

The initial loss scales are measured at identity initialization:

```text
D0 = mean(255^2 * RGB_MSE)
R0 = estimated total bits / total latent scalars
lambda0 = R0 / D0
```

The exact measured lambda sweep is written to `initial_scale.json`.

Stage A first compares the production scalar codec and the identity-initialized
learned path on the same validation scenes using hard round/clamp quantization
and actual fixed-profile rANS. It records rendered metrics, latent MSE, actual
bytes, reconstructed-latent differences, bitstream equality, and decoded-symbol
equality in `identity_equivalence.json`. Training stops before step 1 if this
check fails.

`initial_scale.json` also records mean estimated and actual rANS payload bpv,
per-scene values, and Pearson/Spearman correlations. Both correlations must be
at least 0.2 for the smoke job to continue. The five measured lambda values are
printed for review but are not submitted automatically.

Final checkpoints do not deploy the logistic model. Each final transform gets
a new fixed K=32 profile calibrated from 1,000 transformed training scenes.
That profile is frozen before the 5,601-scene actual-rANS held-out evaluation.

## Freezing and gradient safeguards

The slot encoder runs under `no_grad`. Every pretrained model parameter has
`requires_grad=False`. The Gaussian decoder and renderer forward passes remain
inside autograd during training so rendered RGB gradients reach the codec.
An identity-point gradient probe verifies that all codec tensors receive finite
gradient objects, every trainable codec module has a finite nonzero gradient,
and no pretrained tensor receives a gradient. Because exact zero initialization
can initially block upstream residual-branch gradients, a second probe after
optimizer step 1 requires every trainable codec tensor to have a finite,
nonzero gradient. Stage A saves `smoke_checkpoint.pt`, reloads it into a fresh
codec instance, and compares estimated rate, PSNR, and reconstructed latents.

Validation reports `quantization_mode=hard_round_clamp` and never uses STE.
Malformed dataset examples are recorded with identifier, reason, and active
phase in `malformed_examples.json`; a systematic skip rate aborts the gate.

## Staged SLURM workflow

Only submit Stage A initially:

```bash
bash slurm/submit.sh slurm/learned_codec/train_transform_smoke_re10k.sbatch
```

The smoke job runs 300 optimizer steps with batch size 1 and four-scene gradient
accumulation. It validates at 100-step intervals and writes to the separate
`smoke_stage_a_v2_lambda_1` output directory. Do not submit Stage B unless
`smoke_gate.json` reports PASS for identity equivalence, rate/rANS correlation,
finite codec gradients, zero pretrained gradients, hard validation
quantization, checkpoint round-trip, malformed-data handling, and optimization
sanity. Even a fully passing run still requires manual approval before Stage B.

If all implementation gates pass but the central lambda lowers rate while
losing more than 0.05 dB PSNR, run a separate 300-step smoke at `2*lambda0`
before considering `4*lambda0`; do not launch the full sweep automatically.

Stage B runs the frozen five-lambda sweep for 2,000 steps:

```bash
bash slurm/submit.sh slurm/learned_codec/train_transform_pilot_re10k.sbatch
```

After validation-only selection of exactly three multipliers:

```bash
sbatch --export=ALL,SCREEN_MULTIPLIERS='0.5 1 2' \
  slurm/learned_codec/train_transform_screen_re10k.sbatch
```

Replace the example values with the measured best three. After selecting the
best two at 10,000 steps:

```bash
sbatch --export=ALL,FULL_MULTIPLIERS='0.5 1' \
  slurm/learned_codec/train_transform_full_re10k.sbatch
```

Recalibrate one profile per final checkpoint, then evaluate. Use an `afterok`
dependency or wait for calibration to finish before evaluation:

```bash
sbatch --export=ALL,FINAL_MULTIPLIERS='0.5 1' \
  slurm/learned_codec/calibrate_transform_cdf_re10k.sbatch

sbatch --export=ALL,FINAL_MULTIPLIERS='0.5 1' \
  slurm/learned_codec/eval_transform_re10k.sbatch
```

Summarize actual held-out results against all required scalar baselines:

```bash
python scripts/learned_codec/summarize_transform_rd.py \
  --learned-root outputs/learned_transform/heldout \
  --scalar-comparison optimized_group_quant/heldout/comparison.json \
  --narrow-comparison heldout/comparison.json \
  --output outputs/learned_transform/heldout/comparison.json
```

The primary success criterion is actual rate reduction greater than 8.385239%
with held-out PSNR loss at most 0.05 dB.
