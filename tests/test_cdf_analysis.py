import numpy as np
import pytest
import torch

from scripts.analysis.analyze_grouped_cdf import v2_scale_descriptor_bytes

from globalsplat.compression.cdf_analysis import (
    agglomerative_entropy_mappings,
    channel_histogram,
    cross_entropy_bits,
    empirical_entropy_bits,
    evaluate_fixed_group_mapping,
    grouped_counts,
    out_of_calibration_support,
    quantized_channel_histogram,
    smoothed_pmf,
)


def test_channel_histogram_counts_each_channel_independently():
    symbols = np.array(
        [
            [[0, 1, 2], [1, 1, 0]],
            [[0, 2, 2], [3, 1, 0]],
        ],
        dtype=np.int64,
    )
    counts = channel_histogram(symbols, alphabet=4)

    assert counts.shape == (3, 4)
    np.testing.assert_array_equal(counts[0], [2, 1, 0, 1])
    np.testing.assert_array_equal(counts[1], [0, 3, 1, 0])
    np.testing.assert_array_equal(counts[2], [2, 0, 2, 0])
    np.testing.assert_array_equal(counts.sum(axis=1), [4, 4, 4])


def test_quantized_histogram_uses_one_scene_channel_scales():
    latent = torch.tensor(
        [[[0.0, -2.0], [0.5, 0.0], [1.0, 2.0], [-1.0, 1.0]]]
    )
    counts, scales = quantized_channel_histogram(latent, bits=4)

    assert counts.shape == (2, 15)
    np.testing.assert_array_equal(counts.sum(axis=1), [4, 4])
    assert scales.shape == (2,)
    with pytest.raises(ValueError, match="exactly one scene"):
        quantized_channel_histogram(latent.expand(2, -1, -1), bits=4)


def test_entropy_and_fixed_cross_entropy_are_well_defined():
    counts = np.array([[2, 2], [4, 0]], dtype=np.int64)
    assert empirical_entropy_bits(counts) == pytest.approx(4.0)

    pmf = smoothed_pmf(np.array([[2, 0], [0, 2]]), alpha=0.5)
    assert np.all(pmf > 0)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0)
    value = cross_entropy_bits(counts, pmf)
    assert np.isfinite(value)


def test_out_of_calibration_support_counts_bins_and_values():
    calibration = np.array([[1, 0, 2], [0, 3, 0]])
    evaluation = np.array([[0, 4, 1], [2, 0, 5]])

    assert out_of_calibration_support(calibration, evaluation) == {
        "bins": 3,
        "values": 11,
    }


def test_agglomerative_entropy_grouping_merges_similar_distributions():
    counts = np.array(
        [[100, 1], [90, 10], [1, 100], [10, 90]], dtype=np.int64
    )
    mappings = agglomerative_entropy_mappings(counts, target_groups=(1, 2, 4))

    assert np.array_equal(mappings[4], np.arange(4))
    assert mappings[2][0] == mappings[2][1]
    assert mappings[2][2] == mappings[2][3]
    assert mappings[2][0] != mappings[2][2]
    assert np.all(mappings[1] == 0)
    pooled = grouped_counts(counts, mappings[2])
    np.testing.assert_array_equal(pooled.sum(axis=0), counts.sum(axis=0))


def test_more_fixed_groups_recover_entropy_without_changing_counts():
    counts = np.array(
        [[100, 1], [90, 10], [1, 100], [10, 90]], dtype=np.int64
    )
    mappings = agglomerative_entropy_mappings(counts, target_groups=(1, 2, 4))
    reports = {
        k: evaluate_fixed_group_mapping(
            counts, counts, mapping, alpha=0.5, cdf_precision=None
        )
        for k, mapping in mappings.items()
    }

    assert reports[4]["fixed_ideal_bits"] < reports[2]["fixed_ideal_bits"]
    assert reports[2]["fixed_ideal_bits"] < reports[1]["fixed_ideal_bits"]


def test_v2_scale_descriptor_size_is_separate_from_binary_scale_payload():
    assert v2_scale_descriptor_bytes(512) == 114
