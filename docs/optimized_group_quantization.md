# Optimized decoder-aware group quantization

This experiment replaces manual multiplier tiers with calibration-measured
group RD curves and discrete allocation. It does not train or modify any neural
weights and does not change the fixed G32/T32 rANS entropy model.

1. Measure calibration operating points and build target allocations:

```bash
bash slurm/submit.sh slurm/analysis/measure_group_quant_rd_re10k.sbatch
```

The job evaluates multipliers 1, 1.25, 1.5, 1.75, 2, 2.5, and 3 for every
16-channel group. All multipliers are at least one. It writes compact group RD
curves and an explicit optimized profile for 5%, 7.5%, 10%, 12.5%, and 15%
target reductions.

2. Evaluate baseline, manual references, and optimized policies on held-out
RE10K test scenes:

```bash
bash slurm/submit.sh slurm/evaluation/eval_optimized_group_quant_re10k.sbatch
```

3. Generate the Pareto table and RD plot:

```bash
python scripts/analysis/summarize_optimized_group_quant_rd.py \
  --input-root outputs/optimized_group_quant/heldout \
  --optimized-profile outputs/optimized_group_quant/optimized_profile.json \
  --output outputs/optimized_group_quant/heldout/comparison.json
```

The companion plot is written as `comparison.png`.
