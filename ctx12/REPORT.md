# Codec-input latent decorrelation

## Setting

- Dataset: RE10K, context 12, 200 scenes
- Codec inputs per scene: `[1, 4096, 512]` for texture and geometry
- Quantization: texture 7-bit, geometry 9-bit, `channel` scale
- Artificial rasterizations: 64x64, 32x128

## Main results

| Stream | Adjacent token | Shuffle | Channel-pair | Raw entropy | 1D delta | 1D change | 2D linear | 2D change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Texture | 0.0605 | 0.0615 | 0.1127 | 5.396 | 5.875 | -8.9% | 6.403 | -18.6% |
| Geometry | 0.0825 | 0.0791 | 0.1004 | 7.662 | 8.047 | -5.0% | 8.572 | -11.9% |

Adjacent-token dependence is indistinguishable from the random-permutation
baseline across lags 1–256. Conventional 1D and artificially rasterized 2D
predictors increase, rather than reduce, quantized-symbol entropy. This supports
the claim that the codec inputs contain little exploitable token-local redundancy.

Channel correlation is not negligible, so the evidence should not be described
as proving that all transform coding is ineffective. The defensible conclusion
is specifically about sequential, local, and raster-based predictive coding.

## Paper-ready wording

> The codec-input latents exhibit little dependence along the token dimension:
> adjacent-token correlations are comparable to random permutations across all
> tested lags. Moreover, conventional 1D and artificial 2D predictors increase
> empirical quantized-symbol entropy by 5–19%, indicating limited exploitable
> locality for predictive or raster-based compression.

## Files

- `decorrelation_summary.png` / `.pdf`: lag correlation and entropy comparison
- `channel_correlation_heatmaps.png` / `.pdf`: full 512×512 matrices
- `token_lag_correlation.csv`: lag statistics
- `entropy_comparison.csv`: raw and predictor entropy
- `channel_correlation_summary.csv`: channel-level summary
