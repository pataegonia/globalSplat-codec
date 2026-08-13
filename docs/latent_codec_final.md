# Final scene-latent codec setting

The selected codec is inserted after the dual-stream Mixer MLP and before the
Gaussian decoders. It operates independently on the geometry and texture scene
latents, each shaped `[1, 4096, 512]` for the RE10K 32K model.

## Final configuration

| Component | Setting |
|---|---|
| Geometry quantizer | 9-bit symmetric scalar quantization |
| Texture quantizer | 7-bit symmetric scalar quantization |
| Scale | Per-scene, per-channel maximum; 512 FP32 values per stream |
| Scale serialization | Raw little-endian FP32 (`GSAC0002`) |
| Geometry entropy model | Train-fixed 32-group CDF16 |
| Texture entropy model | Train-fixed 32-group CDF16 |
| Channel grouping | Agglomerative minimum entropy-merge-cost hierarchy |
| CDF smoothing | Additive alpha = 0.5 |
| Entropy coder | CompressAI rANS |
| Shared profile | 83,968 raw bytes, stored once and not charged per scene |
| Additional neural training | None |

The fixed profile is calibrated on 1,000 RE10K training scenes. The deployment
bitstream contains the two rANS payloads, two 512-element FP32 scale arrays, and
container metadata. Fixed CDF tables and channel-to-group mappings are shared
codec state and are not copied into each scene file.

## Actual 5,601-scene result

| Mode | Mean bytes/scene | Saving vs original | Exact symbols |
|---|---:|---:|---:|
| Original adaptive K1 with JSON scales | 3,447,609 | baseline | — |
| Adaptive K1 with binary scales | 3,430,128 | 0.507% | 5,601 / 5,601 |
| Fixed K1 | 3,431,744 | 0.460% | 5,601 / 5,601 |
| Fixed Geo32 / Tex16 | 3,332,292 | 3.345% | 5,601 / 5,601 |
| **Fixed Geo32 / Tex32 (selected)** | **3,331,072** | **3.380%** | **5,601 / 5,601** |

For the selected setting, the mean file breakdown is:

| Field | Bytes/scene |
|---|---:|
| Texture rANS payload | 1,397,047 |
| Geometry rANS payload | 1,929,271 |
| Binary scale side information | 4,096 |
| Container and other metadata | 658 |
| **Total** | **3,331,072** |

The shared Geo32/Tex32 profile is 8,192 bytes larger than Geo32/Tex16 but saves
1,219 bytes per scene. It is therefore smaller in aggregate from seven scenes
onward. CDF selection never changes reconstruction: all 5,601 decoded integer
symbol tensors matched the encoded tensors exactly.

## Full evaluation

Set `GLOBALSPLAT_CDF_PROFILE` to the Phase-2 profile bundle and compose the
`codec=final_g9_t7_g32_t32` preset. The Slurm entry point does both automatically:

```bash
bash slurm/submit.sh slurm/evaluation/eval_latent_codec_final_re10k.sbatch
```

This final job enables synchronized module profiling. After five warm-up scenes
it reports mean/P50/P95/min/max time, CUDA peak allocated/reserved memory, and
process RSS/high-water memory for:

1. view tokenizer (RGB patches, Plucker rays, and camera conditioning),
2. dual-stream slot encoder,
3. complete latent codec round-trip,
4. Gaussian decoder,
5. gsplat rasterizer, and
6. PSNR, SSIM, and LPIPS evaluation modules.

Codec JSON additionally separates rANS encode/decode, geometry/texture stream
times, CDF construction, and group-dispatch time. The relevant output files are:

| File | Contents |
|---|---|
| `scores_all_avg.json` | PSNR, SSIM, LPIPS, and end-to-end benchmark |
| `codec_summary.json` | rate breakdown and detailed codec timing |
| `module_profile_summary.json` | aggregated module timing and memory |
| `module_profile_per_scene.json` | raw per-scene module measurements |
| `peak_memory.json` | largest absolute CUDA allocator peak |
