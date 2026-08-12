# SLURM jobs

- `analysis/`: latent-analysis jobs.
- `benchmark/`: profiling jobs.
- `evaluation/`: evaluation and codec sweeps.
- `training/`: model training jobs.

Submit jobs from the repository root so relative output paths resolve correctly:

```bash
bash slurm/submit.sh slurm/evaluation/eval_latent_codec_channel_re10k.sbatch
```

Additional `sbatch` options go after the job path:

```bash
bash slurm/submit.sh slurm/evaluation/eval_latent_codec_channel_re10k.sbatch --array=0-2
```

SLURM stdout and stderr are written under `logs/slurm/`; model and metric
artifacts remain under `outputs/`.
