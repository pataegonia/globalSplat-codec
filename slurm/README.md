# SLURM jobs

- `analysis/`: latent-analysis jobs.
- `benchmark/`: profiling jobs.
- `evaluation/`: evaluation and codec sweeps.
- `analysis/cdf_phase1_re10k.sbatch`: collect train-calibrated G9/T7 channel
  histograms and compare scene-adaptive global, fixed global, and fixed
  per-channel CDF rates without rendering.
- `analysis/grouped_cdf_re10k.sbatch`: CPU-only K=1/4/8/16/32/512 fixed grouped
  CDF analysis from saved Phase-1 histograms.
- `analysis/fixed_grouped_codec_re10k.sbatch`: four-way GPU array on
  `ariel-v10` measuring actual rANS payload/container/timing and exact symbol
  recovery for adaptive K1, fixed K1, fixed G32/T16, and fixed G32/T32.
- `evaluation/eval_latent_codec_final_re10k.sbatch`: full rendered evaluation
  of the selected G9/T7 + fixed G32/T32 codec on `ariel-v10`, including
  synchronized per-module time and CUDA peak-memory profiling.
- `benchmark/compare_codec_memory_io_re10k.sbatch`: A/B comparison of the
  pre-optimization file-I/O/index-rebuild path and the new in-memory/cached
  path, reporting latency, exact bitstream equality, and bitrate equality.
- `benchmark/profile_synthetic_resolution_memory.sbatch`: dataset-free
  warmed-up, sequential square+video inference memory sweep with per-stage CUDA
  peaks, sampled process RSS, the selected G9/T7 fixed-grouped codec, and
  isolated OOM handling. Set `SWEEP=square` or `SWEEP=video` to run one set.
- `training/`: model training jobs.
- `learned_codec/`: staged one-GPU learned latent-transform jobs. Submit only
  `train_transform_smoke_re10k.sbatch` initially; pilot/screen/full/calibration/
  evaluation are gated by the preceding validation results.

Submit jobs from the repository root so relative output paths resolve correctly:

```bash
bash slurm/submit.sh slurm/evaluation/eval_latent_codec_channel_re10k.sbatch

bash slurm/submit.sh slurm/analysis/cdf_phase1_re10k.sbatch

bash slurm/submit.sh slurm/analysis/grouped_cdf_re10k.sbatch

bash slurm/submit.sh slurm/analysis/fixed_grouped_codec_re10k.sbatch

bash slurm/submit.sh slurm/evaluation/eval_latent_codec_final_re10k.sbatch

bash slurm/submit.sh slurm/benchmark/compare_codec_memory_io_re10k.sbatch

bash slurm/submit.sh slurm/benchmark/profile_synthetic_resolution_memory.sbatch

bash slurm/submit.sh slurm/learned_codec/train_transform_smoke_re10k.sbatch
```

Additional `sbatch` options go after the job path:

```bash
bash slurm/submit.sh slurm/evaluation/eval_latent_codec_channel_re10k.sbatch --array=0-2
```

SLURM stdout and stderr are written under `logs/slurm/`; model and metric
artifacts remain under `outputs/`.
