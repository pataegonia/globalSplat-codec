# Utility scripts

- `analysis/`: latent statistics and visualization utilities.
  - `analyze_scene_latent_correlation.py`: scene-latent correlations.
  - `plot_latent_correlation.py`: correlation histograms.
  - `analyze_cdf_phase1.py`: renderer-free train/test histogram collection,
    adaptive/fixed zero-order entropy comparison, and byte-accurate GSAC header
    breakdown. This does not train network weights.
  - `analyze_grouped_cdf.py`: CPU-only agglomerative entropy grouping for
    K=1/4/8/16/32/512 using saved Phase-1 histograms.
  - `eval_fixed_grouped_codec.py`: actual CompressAI rANS evaluation for the
    adaptive K1, fixed K1, fixed G32/T16, and fixed G32/T32 modes. It verifies
    exact decoded integer symbols and skips the unchanged renderer.
- `benchmark/`: dataset and pipeline profiling utilities.
  - `profile_synthetic_resolution_memory.py`: dataset-free, isolated-process
    CUDA peak-memory sweep over synthetic image resolutions. Reports synthetic
    input, tokenizer, slot encoder, optional latent codec, Gaussian decoder,
    gsplat renderer, and end-to-end inference peaks as JSON and CSV. CPU RSS is
    sampled as well because the arithmetic codec runs through NumPy/CompressAI.
- `evaluation/`: local evaluation and ablation entry points.
- `learned_codec/`: frozen-GlobalSplat learned-transform training, transformed
  K=32 entropy calibration, actual-rANS held-out evaluation, and final scalar
  baseline comparison. See `docs/learned_latent_transform.md` before running.

Run these commands from the repository root. The Python entry points resolve
the repository and Hydra config directories from their own locations.
