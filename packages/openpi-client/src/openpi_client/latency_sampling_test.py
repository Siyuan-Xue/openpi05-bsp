import pytest

from openpi_client import latency_sampling


def _key(*, namespace="formal", request_ordinal=0):
    return latency_sampling.LatencySampleKeyV1(
        namespace=namespace,
        seed=42,
        suite="libero_spatial",
        task_id=0,
        trial_index=0,
        request_ordinal=request_ordinal,
    )


def test_sha256_box_muller_has_frozen_cross_mode_formal_samples():
    sampler = latency_sampling.NormalLatencySamplerV1()

    assert sampler.sample_target_ns(_key(request_ordinal=0)) == 313_006_623
    assert sampler.sample_target_ns(_key(request_ordinal=1)) == 313_452_945
    assert (
        sampler.sample_target_ns(
            latency_sampling.LatencySampleKeyV1(
                namespace="formal",
                seed=42,
                suite="libero_goal",
                task_id=7,
                trial_index=49,
                request_ordinal=12,
            )
        )
        == 348_100_647
    )


def test_mode_is_not_part_of_paired_sample_identity():
    sampler = latency_sampling.NormalLatencySamplerV1()
    baseline_async = sampler.sample_target_ns(_key(request_ordinal=3))
    baseline_rtc = sampler.sample_target_ns(_key(request_ordinal=3))
    bsp_spline_async = sampler.sample_target_ns(_key(request_ordinal=3))

    assert baseline_async == baseline_rtc == bsp_spline_async


def test_calibration_namespace_does_not_consume_formal_sequence():
    sampler = latency_sampling.NormalLatencySamplerV1()

    assert sampler.sample_target_ns(_key(namespace="calibration")) == 295_988_777
    assert sampler.sample_target_ns(_key(namespace="formal")) == 313_006_623


def test_negative_normal_draw_is_deterministically_resampled():
    sampler = latency_sampling.NormalLatencySamplerV1(mean_ns=0, stddev_ns=100)
    key = latency_sampling.LatencySampleKeyV1(
        namespace="negative-test",
        seed=42,
        suite="suite",
        task_id=0,
        trial_index=0,
        request_ordinal=0,
    )

    assert sampler.sample_target_ns(key) == 76


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("namespace", ""),
        ("seed", True),
        ("task_id", -1),
        ("trial_index", 1.5),
        ("request_ordinal", "0"),
    ],
)
def test_sample_key_rejects_noncanonical_identity(field, value):
    values = {
        "namespace": "formal",
        "seed": 42,
        "suite": "libero_spatial",
        "task_id": 0,
        "trial_index": 0,
        "request_ordinal": 0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        latency_sampling.LatencySampleKeyV1(**values)
