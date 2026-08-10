"""Pure planning helpers shared by the JAX training loop and its tests."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import TypeVar

Tree = TypeVar("Tree")


@dataclasses.dataclass(frozen=True)
class GradientAccumulationPlan:
    """Validated global and per-process batch geometry for one optimizer step."""

    batch_size: int
    micro_batch_size: int
    local_micro_batch_size: int
    accumulation_steps: int

    @property
    def accumulation_indices(self) -> range:
        """Indices folded into the RNG for the consecutive micro-batches."""
        return range(self.accumulation_steps)


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def plan_gradient_accumulation(
    *,
    batch_size: int,
    micro_batch_size: int | None,
    process_count: int,
    device_count: int,
) -> GradientAccumulationPlan:
    """Validate and return the global JAX micro-batch plan.

    ``batch_size`` and ``micro_batch_size`` are both global. The existing JAX
    loader partitions either one across processes, so every process receives
    ``micro_batch_size / process_count`` examples for each accumulation index.
    ``None`` retains the historical one-loader-batch-per-update behavior.
    """
    _require_positive_integer("batch_size", batch_size)
    _require_positive_integer("process_count", process_count)
    _require_positive_integer("device_count", device_count)
    if device_count % process_count != 0:
        raise ValueError(f"Device count {device_count} must be divisible by process count {process_count}.")

    resolved_micro_batch_size = batch_size if micro_batch_size is None else micro_batch_size
    _require_positive_integer("micro_batch_size", resolved_micro_batch_size)
    if resolved_micro_batch_size > batch_size:
        raise ValueError(
            f"Micro-batch size {resolved_micro_batch_size} cannot exceed effective batch size {batch_size}."
        )
    if batch_size % device_count != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by device count {device_count}.")
    if batch_size % process_count != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by process count {process_count}.")
    if batch_size % resolved_micro_batch_size != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by micro-batch size {resolved_micro_batch_size}.")
    if resolved_micro_batch_size % device_count != 0:
        raise ValueError(
            f"Micro-batch size {resolved_micro_batch_size} must be divisible by device count {device_count}."
        )
    if resolved_micro_batch_size % process_count != 0:
        raise ValueError(
            f"Micro-batch size {resolved_micro_batch_size} must be divisible by process count {process_count}."
        )

    return GradientAccumulationPlan(
        batch_size=batch_size,
        micro_batch_size=resolved_micro_batch_size,
        local_micro_batch_size=resolved_micro_batch_size // process_count,
        accumulation_steps=batch_size // resolved_micro_batch_size,
    )


def add_trees(left: Tree, right: Tree, *, tree_map: Callable[..., Tree]) -> Tree:
    """Add matching gradient-tree leaves without depending on JAX at import time."""
    return tree_map(lambda left_leaf, right_leaf: left_leaf + right_leaf, left, right)


def average_tree_sum(tree_sum: Tree, count: int, *, tree_map: Callable[..., Tree]) -> Tree:
    """Divide every leaf of a summed gradient tree by its micro-batch count."""
    _require_positive_integer("count", count)
    return tree_map(lambda leaf: leaf / count, tree_sum)


def optimizer_step_numbers(start_step: int, num_train_steps: int) -> range:
    """Return updated optimizer-step numbers, including the final requested step."""
    if isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0:
        raise ValueError(f"start_step must be a nonnegative integer, got {start_step!r}")
    if isinstance(num_train_steps, bool) or not isinstance(num_train_steps, int) or num_train_steps < 0:
        raise ValueError(f"num_train_steps must be a nonnegative integer, got {num_train_steps!r}")
    if start_step > num_train_steps:
        raise ValueError(f"Restored optimizer step {start_step} exceeds requested training steps {num_train_steps}.")
    return range(start_step + 1, num_train_steps + 1)


def should_save_checkpoint(completed_step: int, *, num_train_steps: int, save_interval: int) -> bool:
    """Return whether an updated optimizer step is an interval or final checkpoint."""
    _require_positive_integer("completed_step", completed_step)
    _require_positive_integer("num_train_steps", num_train_steps)
    _require_positive_integer("save_interval", save_interval)
    if completed_step > num_train_steps:
        raise ValueError(
            f"Completed optimizer step {completed_step} exceeds requested training steps {num_train_steps}."
        )
    return completed_step % save_interval == 0 or completed_step == num_train_steps


def should_keep_checkpoint(
    step: int,
    *,
    permanent_steps: tuple[int, ...],
    keep_period: int | None,
) -> bool:
    """Return whether Orbax must preserve a checkpoint beyond ``max_to_keep``."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a nonnegative integer, got {step!r}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in permanent_steps):
        raise ValueError("permanent_steps must be unique nonnegative integers in ascending order")
    if tuple(sorted(permanent_steps)) != permanent_steps or len(set(permanent_steps)) != len(permanent_steps):
        raise ValueError("permanent_steps must be unique nonnegative integers in ascending order")
    if keep_period is not None:
        _require_positive_integer("keep_period", keep_period)
    return step in permanent_steps or (keep_period is not None and step % keep_period == 0)
