#!/usr/bin/env python3
"""Build and evaluate fixed grouped CDF profiles from Phase-1 histograms."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from globalsplat.compression.cdf_analysis import (
    agglomerative_entropy_mappings,
    evaluate_fixed_group_mapping,
    grouped_counts,
)


STREAMS = ("texture", "geometry")
DEFAULT_GROUPS = (1, 4, 8, 16, 32, 512)
ASYMMETRIC_CONFIGS = ((8, 4), (16, 4), (16, 8), (32, 4), (32, 8), (32, 16))


def v2_scale_descriptor_bytes(scale_count: int = 512) -> int:
    """JSON bytes added by V2's two binary-scale descriptors.

    The scale payload itself is accounted separately as raw FP32.  This is only
    the compact-JSON metadata needed to locate and type those arrays.
    """
    scale_bytes = scale_count * np.dtype("<f4").itemsize
    base = {"x": 0}
    described = {
        "x": 0,
        "scale_dtype": "<f4",
        "scale_count": scale_count,
        "scale_bytes": scale_bytes,
    }
    compact = lambda value: json.dumps(value, separators=(",", ":")).encode()
    return 2 * (len(compact(described)) - len(compact(base)))


def rate_dict(bits: float, values: int, scenes: int) -> dict[str, float]:
    return {
        "total_bits": float(bits),
        "bits_per_value": float(bits / values),
        "bytes_per_scene": float(bits / (8 * scenes)),
    }


def add_recovery(rows: dict[int, dict], rate_name: str) -> None:
    first = rows[min(rows)][rate_name]["total_bits"]
    ceiling = rows[max(rows)][rate_name]["total_bits"]
    maximum_gain = first - ceiling
    for row in rows.values():
        gain = first - row[rate_name]["total_bits"]
        row[f"{rate_name}_gain_vs_k1_bytes_per_scene"] = gain / (8 * row["scenes"])
        row[f"{rate_name}_recovery_percent"] = (
            100.0 * gain / maximum_gain if maximum_gain > 0 else 0.0
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histograms", required=True, type=Path)
    parser.add_argument("--phase1-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--groups", type=int, nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cdf-precision", type=int, default=16)
    parser.add_argument("--skip-cdf-quantization", action="store_true")
    parser.add_argument(
        "--baseline-total-bytes",
        type=float,
        default=3447609.138189609,
        help="Measured scene-adaptive G9/T7 total used only for total-size deltas",
    )
    parser.add_argument("--binary-scale-bytes", type=int, default=4096)
    parser.add_argument("--container-bytes", type=float, default=None)
    args = parser.parse_args()

    hist_path = args.histograms.expanduser().resolve()
    phase1_path = args.phase1_report.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not hist_path.is_file():
        raise SystemExit(f"histograms not found: {hist_path}")
    if not phase1_path.is_file():
        raise SystemExit(f"Phase-1 report not found: {phase1_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    scenes = int(phase1["evaluation_scenes"])
    v1_container_bytes = float(phase1["header_breakdown"]["mean_container_bytes"])
    descriptor_bytes = v2_scale_descriptor_bytes()
    container_bytes = (
        float(args.container_bytes)
        if args.container_bytes is not None
        else v1_container_bytes + descriptor_bytes
    )
    groups = tuple(sorted(set(args.groups)))
    if groups[0] != 1 or groups[-1] != 512:
        raise SystemExit("groups must include K=1 and K=512 for gain recovery")
    cdf_precision = None if args.skip_cdf_quantization else args.cdf_precision
    histograms = np.load(hist_path)

    stream_reports: dict[str, dict] = {}
    profile_arrays: dict[str, np.ndarray] = {}
    for stream in STREAMS:
        calibration = histograms[f"calibration_{stream}_counts"]
        evaluation = histograms[f"evaluation_{stream}_counts"]
        print(f"building {stream} hierarchy for K={groups}", flush=True)
        mappings = agglomerative_entropy_mappings(calibration, groups)
        values = int(evaluation.sum())
        rows: dict[int, dict] = {}
        for k in groups:
            result = evaluate_fixed_group_mapping(
                calibration,
                evaluation,
                mappings[k],
                alpha=args.alpha,
                cdf_precision=cdf_precision,
            )
            row = {
                "groups": k,
                "scenes": scenes,
                "empirical": rate_dict(result["empirical_bits"], values, scenes),
                "fixed_ideal": rate_dict(result["fixed_ideal_bits"], values, scenes),
                "mismatch_penalty_bytes_per_scene": float(
                    result["mismatch_penalty_bits"] / (8 * scenes)
                ),
                "profile_bytes_raw": int(
                    mappings[k].astype(np.uint16).nbytes
                    + k * (calibration.shape[1] + 1) * np.dtype(np.int32).itemsize
                ),
            }
            if result["fixed_cdf_quantized_bits"] is not None:
                row["fixed_cdf_quantized"] = rate_dict(
                    result["fixed_cdf_quantized_bits"], values, scenes
                )
                row["cdf_quantization_penalty_bytes_per_scene"] = float(
                    (result["fixed_cdf_quantized_bits"] - result["fixed_ideal_bits"])
                    / (8 * scenes)
                )
            rows[k] = row
            profile_arrays[f"{stream}_mapping_k{k}"] = mappings[k].astype(np.uint16)
            profile_arrays[f"{stream}_calibration_counts_k{k}"] = grouped_counts(
                calibration, mappings[k]
            )
        add_recovery(rows, "fixed_ideal")
        if all("fixed_cdf_quantized" in row for row in rows.values()):
            add_recovery(rows, "fixed_cdf_quantized")
        stream_reports[stream] = {str(k): rows[k] for k in groups}

    combined_rows: dict[int, dict] = {}
    values = sum(
        int(histograms[f"evaluation_{stream}_counts"].sum()) for stream in STREAMS
    )
    for k in groups:
        tex = stream_reports["texture"][str(k)]
        geo = stream_reports["geometry"][str(k)]
        row = {
            "groups_per_stream": k,
            "scenes": scenes,
            "fixed_ideal": rate_dict(
                tex["fixed_ideal"]["total_bits"] + geo["fixed_ideal"]["total_bits"],
                values,
                scenes,
            ),
            "profile_bytes_raw": tex["profile_bytes_raw"] + geo["profile_bytes_raw"],
        }
        if "fixed_cdf_quantized" in tex and "fixed_cdf_quantized" in geo:
            row["fixed_cdf_quantized"] = rate_dict(
                tex["fixed_cdf_quantized"]["total_bits"]
                + geo["fixed_cdf_quantized"]["total_bits"],
                values,
                scenes,
            )
        payload_key = "fixed_cdf_quantized" if "fixed_cdf_quantized" in row else "fixed_ideal"
        estimated_total = (
            row[payload_key]["bytes_per_scene"]
            + args.binary_scale_bytes
            + container_bytes
        )
        row["estimated_total_with_binary_scale_bytes_per_scene"] = estimated_total
        row["saving_vs_measured_baseline_bytes_per_scene"] = (
            args.baseline_total_bytes - estimated_total
        )
        row["saving_vs_measured_baseline_percent"] = (
            100.0 * (args.baseline_total_bytes - estimated_total)
            / args.baseline_total_bytes
        )
        combined_rows[k] = row
    add_recovery(combined_rows, "fixed_ideal")
    if all("fixed_cdf_quantized" in row for row in combined_rows.values()):
        add_recovery(combined_rows, "fixed_cdf_quantized")

    asymmetric = []
    for geo_k, tex_k in ASYMMETRIC_CONFIGS:
        if geo_k not in groups or tex_k not in groups:
            continue
        geo = stream_reports["geometry"][str(geo_k)]
        tex = stream_reports["texture"][str(tex_k)]
        rate_name = (
            "fixed_cdf_quantized"
            if "fixed_cdf_quantized" in geo and "fixed_cdf_quantized" in tex
            else "fixed_ideal"
        )
        payload = geo[rate_name]["bytes_per_scene"] + tex[rate_name]["bytes_per_scene"]
        estimated_total = payload + args.binary_scale_bytes + container_bytes
        asymmetric.append(
            {
                "geometry_groups": geo_k,
                "texture_groups": tex_k,
                "rate": rate_name,
                "payload_bytes_per_scene": payload,
                "estimated_total_with_binary_scale_bytes_per_scene": estimated_total,
                "saving_vs_measured_baseline_bytes_per_scene": (
                    args.baseline_total_bytes - estimated_total
                ),
                "saving_vs_measured_baseline_percent": (
                    100.0 * (args.baseline_total_bytes - estimated_total)
                    / args.baseline_total_bytes
                ),
                "profile_bytes_raw": (
                    geo["profile_bytes_raw"] + tex["profile_bytes_raw"]
                ),
            }
        )
    asymmetric.sort(
        key=lambda row: row["estimated_total_with_binary_scale_bytes_per_scene"]
    )

    report = {
        "config": {
            "histograms": str(hist_path),
            "phase1_report": str(phase1_path),
            "groups": list(groups),
            "alpha": args.alpha,
            "cdf_precision": cdf_precision,
            "evaluation_scenes": scenes,
            "baseline_total_bytes_per_scene": args.baseline_total_bytes,
            "binary_scale_bytes_per_scene": args.binary_scale_bytes,
            "estimated_container_bytes_per_scene": container_bytes,
            "v1_container_bytes_from_phase1": v1_container_bytes,
            "v2_binary_scale_descriptor_bytes": descriptor_bytes,
            "recovery_definition": "eta_K=(R_K1-R_K)/(R_K1-R_K512)",
            "profile_bytes_are_one_time_not_per_scene": True,
        },
        "streams": stream_reports,
        "combined_symmetric": {str(k): combined_rows[k] for k in groups},
        "asymmetric_candidates": asymmetric,
    }
    (output_dir / "grouped_cdf_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "grouped_cdf_profiles.npz", **profile_arrays)
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote grouped CDF analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
