"""Standard-library-only contracts for optimizer-step planning."""

import pytest

from openpi.training import train_planning as train_planning_module
from openpi.training.train_planning import add_trees
from openpi.training.train_planning import average_tree_sum
from openpi.training.train_planning import optimizer_step_numbers
from openpi.training.train_planning import plan_gradient_accumulation
from openpi.training.train_planning import should_save_checkpoint


def _tree_map(function, *trees):
    first = trees[0]
    if isinstance(first, dict):
        return {key: _tree_map(function, *(tree[key] for tree in trees)) for key in first}
    if isinstance(first, tuple):
        return tuple(_tree_map(function, *(tree[index] for tree in trees)) for index in range(len(first)))
    return function(*trees)


def test_none_and_full_micro_batch_preserve_single_batch_behavior():
    without_accumulation = plan_gradient_accumulation(
        batch_size=256,
        micro_batch_size=None,
        process_count=1,
        device_count=1,
    )
    explicit_full_batch = plan_gradient_accumulation(
        batch_size=256,
        micro_batch_size=256,
        process_count=1,
        device_count=1,
    )

    assert without_accumulation == explicit_full_batch
    assert without_accumulation.accumulation_steps == 1
    assert without_accumulation.local_micro_batch_size == 256


def test_global_micro_batch_is_partitioned_per_process():
    plan = plan_gradient_accumulation(
        batch_size=256,
        micro_batch_size=32,
        process_count=4,
        device_count=8,
    )

    assert plan.micro_batch_size == 32
    assert plan.local_micro_batch_size == 8
    assert plan.accumulation_steps == 8


def test_single_device_h20_plan_uses_256_micro_batches():
    plan = plan_gradient_accumulation(
        batch_size=256,
        micro_batch_size=1,
        process_count=1,
        device_count=1,
    )

    assert plan.accumulation_steps == 256
    assert tuple(plan.accumulation_indices) == tuple(range(256))


@pytest.mark.parametrize(
    ("batch_size", "micro_batch_size", "process_count", "device_count", "expected_match"),
    [
        (0, None, 1, 1, "batch_size must be a positive integer"),
        (256, 0, 1, 1, "micro_batch_size must be a positive integer"),
        (256, 512, 1, 1, "cannot exceed effective batch size"),
        (255, 1, 1, 2, "Batch size 255 must be divisible by device count 2"),
        (256, 3, 1, 1, "must be divisible by micro-batch size"),
        (256, 4, 2, 8, "Micro-batch size 4 must be divisible by device count 8"),
        (256, 8, 3, 8, "Device count 8 must be divisible by process count 3"),
        (256, 8, 0, 8, "process_count must be a positive integer"),
    ],
)
def test_invalid_batch_and_topology_combinations_are_rejected(
    batch_size,
    micro_batch_size,
    process_count,
    device_count,
    expected_match,
):
    with pytest.raises(ValueError, match=expected_match):
        plan_gradient_accumulation(
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            process_count=process_count,
            device_count=device_count,
        )


def test_gradient_trees_are_summed_and_averaged_leafwise():
    first = {"encoder": (2.0, 6.0), "head": {"kernel": 10.0}}
    second = {"encoder": (4.0, 2.0), "head": {"kernel": 2.0}}

    total = add_trees(first, second, tree_map=_tree_map)
    averaged = average_tree_sum(total, 2, tree_map=_tree_map)

    assert total == {"encoder": (6.0, 8.0), "head": {"kernel": 12.0}}
    assert averaged == {"encoder": (3.0, 4.0), "head": {"kernel": 6.0}}


def test_tree_average_rejects_a_nonpositive_micro_batch_count():
    with pytest.raises(ValueError, match="count must be a positive integer"):
        average_tree_sum({"weight": 1.0}, 0, tree_map=_tree_map)


def test_one_optimizer_step_has_many_micro_batches_but_one_completed_step():
    plan = plan_gradient_accumulation(
        batch_size=8,
        micro_batch_size=2,
        process_count=1,
        device_count=1,
    )
    start_step = 17
    completed_step = tuple(optimizer_step_numbers(start_step, start_step + 1))

    assert tuple(plan.accumulation_indices) == (0, 1, 2, 3)
    assert completed_step == (18,)


def test_checkpoint_predicate_uses_updated_optimizer_step_labels():
    saved = [
        step
        for step in optimizer_step_numbers(0, 30_000)
        if should_save_checkpoint(step, num_train_steps=30_000, save_interval=1_000)
    ]

    assert saved[:2] == [1_000, 2_000]
    assert saved[-3:] == [28_000, 29_000, 30_000]
    assert 10_000 in saved
    assert 20_000 in saved
    assert 30_000 in saved
    assert 0 not in saved


def test_final_step_is_saved_when_not_aligned_to_interval():
    assert not should_save_checkpoint(2_500, num_train_steps=2_501, save_interval=1_000)
    assert should_save_checkpoint(2_501, num_train_steps=2_501, save_interval=1_000)


def test_resume_begins_after_the_completed_checkpoint_boundary():
    resumed_steps = optimizer_step_numbers(10_000, 30_000)

    assert resumed_steps.start == 10_001
    assert resumed_steps.stop == 30_001
    assert len(resumed_steps) == 20_000


def test_exact_phase_one_short10k_milestones_are_preserved():
    assert hasattr(train_planning_module, "should_keep_checkpoint"), (
        "train planning must expose the exact checkpoint preservation predicate"
    )
    permanent = (0, 1_000, 2_000, 5_000, 10_000)

    kept = [
        step
        for step in (0, 1_000, 2_000, 3_000, 5_000, 9_000, 10_000)
        if train_planning_module.should_keep_checkpoint(
            step,
            permanent_steps=permanent,
            keep_period=10_000,
        )
    ]

    assert kept == [0, 1_000, 2_000, 5_000, 10_000]


@pytest.mark.parametrize(
    ("case", "expected_match"),
    [
        ({"step": -1, "permanent_steps": (), "keep_period": None}, "step must be a nonnegative integer"),
        ({"step": False, "permanent_steps": (), "keep_period": None}, "step must be a nonnegative integer"),
        (
            {"step": 0, "permanent_steps": (-1,), "keep_period": None},
            "permanent_steps must be unique nonnegative integers",
        ),
        (
            {"step": 0, "permanent_steps": (False,), "keep_period": None},
            "permanent_steps must be unique nonnegative integers",
        ),
        (
            {"step": 0, "permanent_steps": (0, 0), "keep_period": None},
            "permanent_steps must be unique nonnegative integers",
        ),
        (
            {"step": 0, "permanent_steps": (5_000, 0), "keep_period": None},
            "permanent_steps must be unique nonnegative integers",
        ),
        ({"step": 0, "permanent_steps": (), "keep_period": 0}, "keep_period must be a positive integer"),
    ],
)
def test_checkpoint_preservation_rejects_malformed_steps_and_periods(case, expected_match):
    assert hasattr(train_planning_module, "should_keep_checkpoint"), (
        "train planning must expose the exact checkpoint preservation predicate"
    )

    with pytest.raises(ValueError, match=expected_match):
        train_planning_module.should_keep_checkpoint(**case)
