"""Capture-QC functions vendored from public-dataset-adapter, written by Sambhav Gupta at Pantheon
(commit 99d0a9e), and released here under this repository's Apache-2.0 license with his credit.

These functions are copied so that this pipeline runs the same checks without importing that repository's
Lance, LanceDB, Hugging Face, OpenCV or GPU dependencies. Only numpy and scipy are needed.
checks/capture_qc.py is the only caller.

Source files and functions, in the order they appear below:

  adapters/common/filtering.py
    FilterPolicy (thresholds kept at the upstream defaults; our calibrated overrides live in
      checks/capture_qc.py), action_intensity_features, _correlation_record,
      visual_action_correlation_checks, _strict_timestamps, _gap_metrics, _event_spans,
      _robust_z_max, _jump_return_indices, _rotation_jump_return_indices, _interleaved_hold_indices,
      _derivative_metrics, _motion_metrics, _gripper_integral_checks, gripper_sensor_bug_checks,
      _inactive_gripper_checks, _largest_action_video_checks, _unexplained_visual_change_checks
  adapters/common/processing.py
    ARM_SLICES, ARM_OFFSETS, SMOOTHING_* constants, median_period_ns, delta_actions, _valid_runs,
      _smooth_euclidean_path, _smooth_rotation_path, smooth_canonical_trajectory

Every modification made to the upstream text:

  1. Imports. Removed pyarrow, subprocess, tempfile, os and the adapters.common.common imports
     (ACTION_WIDTH, AdapterError, canonical_json, lance_table_write_lock, open_database) and the
     adapters.common.processing / gpu_video imports. ACTION_WIDTH (14) and ARM_OFFSETS are defined
     here with their upstream values, and EpisodeDrop is a local ValueError subclass instead of
     processing.EpisodeDrop(AdapterError).
  2. Nothing that writes, migrates or regrades rows is included (quality_from_decision,
     add_gripper_sensor_bug_quality, evaluate_episode, apply_localized_validity,
     regrade_quality_record and the native/raw-row layer stay upstream). The caller composes the
     checks the way evaluate_episode does; checks/capture_qc.py names the upstream line of each.
  3. Left out because nothing here calls them: from filtering.py UMI_SPEED_SOURCES,
     RETIRED_QUARANTINE_REASONS, dataset_filter_policy (its one override is for a check capture_qc.py
     never shows), inspect_processed_video and its ffmpeg decode (capture_qc.py decodes each camera
     itself and computes the same frame statistics with its calibrated thresholds), _gap_evidence,
     _indices_in_runs_at_least and _pose_value_reasons; from processing.py ROTATION_SLICES. Every
     function that is included is copied whole.
  4. _largest_action_video_checks: a camera whose per-interval value at the checked index is not
     finite is skipped for that index (upstream skips a camera only when the index is past its
     end), and the camera's median and percentile are taken over its finite values. We mark an
     anchor interval NaN when a camera paired by real time recorded no new frame in it, so that an
     interval without a camera observation is never read as "no visual motion".
  5. _unexplained_visual_change_checks: the per-camera mean and 95th percentile in the metrics are
     taken over finite values (np.nanmean / np.nanquantile) for the same reason. The flag rule is
     unchanged (a NaN never passes the >= test).

Requires Python 3.10 or newer (upstream uses dataclass slots and zip(strict=...)).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from scipy.sparse import eye as sparse_eye
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

SOURCE = "public-dataset-adapter@99d0a9e"
ACTION_WIDTH = 14
ARM_OFFSETS = (0, 7)


class EpisodeDrop(ValueError):
    """Local stand-in for processing.EpisodeDrop (modification 1)."""


# ---------------------------------------------------------------- adapters/common/filtering.py


@dataclasses.dataclass(frozen=True, slots=True)
class FilterPolicy:
    """Explicit starter thresholds; none silently alter stored values."""

    maximum_umi_translation_speed_m_s: float = 2.0
    maximum_umi_rotation_speed_rad_s: float = 8.0
    jump_relative_robust_z: float = 10.0
    minimum_jump_translation_m: float = 0.02
    minimum_jump_rotation_rad: float = 0.15
    maximum_jump_interval_ratio: float = 1.5
    maximum_static_fraction: float = 0.95
    static_translation_m: float = 1e-4
    static_rotation_rad: float = 1e-3
    static_gripper_delta: float = 1e-4
    gripper_sensor_bug_minimum_delta: float = 0.5
    round_trip_translation_tolerance_m: float = 2e-4
    round_trip_rotation_tolerance_rad: float = 2e-4
    maximum_extreme_exposure_fraction: float = 0.25
    maximum_low_contrast_fraction: float = 0.25
    maximum_duplicate_pair_fraction: float = 0.20
    maximum_frozen_run_s: float = 1.0
    minimum_smoothness_translation_m: float = 0.001
    minimum_smoothness_rotation_rad: float = 0.02
    maximum_interleaved_hold_ratio: float = 0.10
    minimum_interleaved_hold_events: int = 3
    largest_action_visual_percentile: float = 0.15
    largest_action_visual_median_ratio: float = 0.35
    largest_action_visual_absolute_difference: float = 1.0
    gripper_integral_tolerance: float = 0.01
    minimum_episode_duration_s: float = 5.0
    minimum_unexplained_visual_difference: float = 8.0
    maximum_unexplained_translation_m: float = 5e-4
    maximum_unexplained_rotation_rad: float = 5e-3
    maximum_unexplained_gripper_delta: float = 5e-3
    minimum_visual_action_r_squared: float = 0.04
    minimum_visual_action_pairs: int = 30
    minimum_visual_action_active_pairs: int = 8


def action_intensity_features(row: Mapping[str, Any]) -> dict[str, list[float]]:
    """Return per-arm and total pose-action intensity, excluding grippers.

    Translation and rotation each receive unit total mass within an episode,
    making meter and radian contributions equal before they are summed.
    """

    robot = cast(Mapping[str, Any], row["robot"])
    actions = np.asarray(robot.get("actions_global") or [], dtype=np.float64)
    valid = np.asarray(robot.get("action_valid_global") or [], dtype=np.bool_)
    if actions.ndim != 2 or actions.shape[1:] != (ACTION_WIDTH,) or valid.shape != actions.shape:
        return {f"{name}_action_intensity": [] for name in ("left", "right", "total")}
    arms: list[np.ndarray] = []
    for offset in ARM_OFFSETS:
        translation = np.where(
            valid[:, offset : offset + 3], np.abs(actions[:, offset : offset + 3]), 0.0
        ).sum(axis=1)
        rotation = np.where(
            valid[:, offset + 3 : offset + 6], np.abs(actions[:, offset + 3 : offset + 6]), 0.0
        ).sum(axis=1)
        translation /= max(float(translation.sum()), 1e-12)
        rotation /= max(float(rotation.sum()), 1e-12)
        arms.append(translation + rotation)
    return {
        "left_action_intensity": arms[0].astype(np.float32).tolist(),
        "right_action_intensity": arms[1].astype(np.float32).tolist(),
        "total_action_intensity": (arms[0] + arms[1]).astype(np.float32).tolist(),
    }


def _correlation_record(
    action: Sequence[float], pixel: Sequence[float], policy: FilterPolicy
) -> dict[str, Any]:
    action_array = np.asarray(action, dtype=np.float64)
    pixel_array = np.asarray(pixel, dtype=np.float64)
    usable = min(len(action_array), max(0, len(pixel_array) - 1))
    if usable < policy.minimum_visual_action_pairs:
        return {"status": "inconclusive", "reason": "too_few_pairs", "pairs": usable}
    x = action_array[:usable]
    y = pixel_array[1 : usable + 1]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < policy.minimum_visual_action_pairs:
        return {"status": "inconclusive", "reason": "too_few_finite_pairs", "pairs": len(x)}
    active_threshold = float(np.quantile(x, 0.75))
    active = int(np.count_nonzero(x > max(active_threshold, 1e-12)))
    if (
        active < policy.minimum_visual_action_active_pairs
        or np.std(x) <= 1e-12
        or np.std(y) <= 1e-6
    ):
        return {
            "status": "inconclusive",
            "reason": "insufficient_motion_variance",
            "pairs": len(x),
            "active_pairs": active,
        }
    # Winsorization makes the episode statistic insensitive to one cut, flash,
    # human occlusion, or encoder glitch while retaining zero-lag alignment.
    x = np.minimum(x, np.quantile(x, 0.99))
    y = np.minimum(y, np.quantile(y, 0.99))
    # Clipping can remove the only varying samples (e.g. a single camera cut).
    # Check the actual correlation inputs, not just the pre-clipping signals.
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-6:
        return {
            "status": "inconclusive",
            "reason": "insufficient_variance_after_clipping",
            "pairs": len(x),
            "active_pairs": active,
        }
    correlation = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(correlation):
        return {
            "status": "inconclusive",
            "reason": "nonfinite_correlation",
            "pairs": len(x),
            "active_pairs": active,
        }
    r_squared = correlation * correlation
    mismatch = correlation <= 0.0 or r_squared < policy.minimum_visual_action_r_squared
    return {
        "status": "mismatch" if mismatch else "matched",
        "pairs": len(x),
        "active_pairs": active,
        "correlation": correlation,
        "r_squared": r_squared,
        "threshold_r_squared": policy.minimum_visual_action_r_squared,
        "alignment": "action[i] versus pixel_change[i+1], zero lag",
    }


def visual_action_correlation_checks(
    row: Mapping[str, Any], policy: FilterPolicy
) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {}
    reasons: list[str] = []
    mapping = (
        (
            "left",
            "left_action_intensity",
            "left_pixel_change_amount",
            "left_gripper_pixel_action_corr_mismatch",
        ),
        (
            "right",
            "right_action_intensity",
            "right_pixel_change_amount",
            "right_gripper_pixel_action_corr_mismatch",
        ),
        (
            "overhead",
            "total_action_intensity",
            "overhead_pixel_change_amount",
            "overhead_pixel_action_corr_mismatch",
        ),
    )
    for name, action_column, pixel_column, reason in mapping:
        action = row.get(action_column)
        pixel = row.get(pixel_column)
        if action is None or pixel is None:
            checks[name] = {"status": "unavailable"}
            continue
        record = _correlation_record(action, pixel, policy)
        checks[name] = record
        if record["status"] == "mismatch":
            reasons.append(reason)
    return checks, reasons


def _strict_timestamps(values: Sequence[int], label: str, reasons: list[str]) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.int64)
    if len(timestamps) < 2:
        reasons.append(f"{label}_too_short")
        return np.empty(0, dtype=np.float64)
    differences = np.diff(timestamps).astype(np.float64) / 1e9
    if np.any(differences <= 0):
        reasons.append(f"{label}_non_monotonic_or_duplicate")
    return differences


def _gap_metrics(differences: np.ndarray) -> dict[str, Any]:
    if not len(differences):
        return {
            "median_dt_s": None,
            "maximum_dt_s": None,
            "gap_count": 0,
            "gap_interval_indices": [],
            "gap_intervals": [],
        }
    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    threshold = max(1.5 * median, median + 5.0 * mad)
    indices = np.flatnonzero(differences > threshold).astype(int).tolist()
    return {
        "median_dt_s": median,
        "maximum_dt_s": float(np.max(differences)),
        "gap_threshold_s": threshold,
        "gap_count": len(indices),
        "gap_interval_indices": indices,
        "gap_intervals": [
            {
                "interval_index": index,
                "actual_dt_s": float(differences[index]),
                "expected_dt_s": median,
                "threshold_dt_s": threshold,
                "excess_dt_s": float(differences[index] - median),
                "expected_interval_ratio": float(differences[index] / max(median, 1e-12)),
            }
            for index in indices
        ],
    }


def _event_spans(
    reason: str,
    indices: Sequence[int],
    *,
    arm: str | None = None,
    camera: str | None = None,
    detail: str | None = None,
) -> list[dict[str, Any]]:
    """Compress action/frame indices into inclusive frame spans for the UI."""

    ordered = sorted(set(int(index) for index in indices if int(index) >= 0))
    if not ordered:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index > previous + 1:
            runs.append((start, previous + 1))
            start = index
        previous = index
    runs.append((start, previous + 1))
    return [
        {
            "reason": reason,
            "start_frame": start,
            "end_frame": end,
            **({"arm": arm} if arm is not None else {}),
            **({"camera": camera} if camera is not None else {}),
            **({"detail": detail} if detail is not None else {}),
        }
        for start, end in runs
    ]


def _robust_z_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    if mad < 1e-12:
        return 0.0 if np.allclose(finite, median) else 1e12
    return float(np.max(np.abs(finite - median) / (1.4826 * mad)))


def _jump_return_indices(
    increments: np.ndarray,
    timestamps_ns: np.ndarray,
    interval_valid: np.ndarray,
    *,
    relative_robust_z: float,
    minimum_magnitude: float,
    maximum_interval_ratio: float,
) -> list[int]:
    if len(increments) < 2:
        return []
    magnitudes = np.linalg.norm(increments, axis=1)
    valid_magnitudes = magnitudes[interval_valid]
    if not len(valid_magnitudes):
        return []
    median = np.median(valid_magnitudes)
    mad = np.median(np.abs(valid_magnitudes - median))
    threshold = max(
        minimum_magnitude,
        median + relative_robust_z * 1.4826 * max(mad, 1e-12),
    )
    spacing = np.diff(timestamps_ns).astype(np.float64)
    median_spacing = np.median(spacing)
    regular = spacing <= maximum_interval_ratio * median_spacing
    left = increments[:-1]
    right = increments[1:]
    large = (
        (magnitudes[:-1] > threshold)
        & (magnitudes[1:] > threshold)
        & interval_valid[:-1]
        & interval_valid[1:]
        & regular[:-1]
        & regular[1:]
    )
    reversal = np.einsum("ij,ij->i", left, right) < 0
    events = (
        large
        & reversal
        & (
            np.linalg.norm(left + right, axis=1)
            < 0.25 * (np.linalg.norm(left, axis=1) + np.linalg.norm(right, axis=1))
        )
    )
    return np.flatnonzero(events).astype(int).tolist()


def _rotation_jump_return_indices(
    increments: np.ndarray,
    timestamps_ns: np.ndarray,
    interval_valid: np.ndarray,
    *,
    relative_robust_z: float,
    minimum_magnitude: float,
    maximum_interval_ratio: float,
) -> list[int]:
    """Count adjacent, unusually large rotations that nearly cancel."""

    if len(increments) < 2:
        return []
    magnitudes = np.linalg.norm(increments, axis=1)
    valid_magnitudes = magnitudes[interval_valid]
    if not len(valid_magnitudes):
        return []
    median = np.median(valid_magnitudes)
    mad = np.median(np.abs(valid_magnitudes - median))
    threshold = max(
        minimum_magnitude,
        median + relative_robust_z * 1.4826 * max(mad, 1e-12),
    )
    spacing = np.diff(timestamps_ns).astype(np.float64)
    median_spacing = np.median(spacing)
    regular = spacing <= maximum_interval_ratio * median_spacing
    large = (
        (magnitudes[:-1] > threshold)
        & (magnitudes[1:] > threshold)
        & interval_valid[:-1]
        & interval_valid[1:]
        & regular[:-1]
        & regular[1:]
    )
    if not large.any():
        return []
    first = Rotation.from_rotvec(increments[:-1])
    second = Rotation.from_rotvec(increments[1:])
    # SAFETY: scipy stubs lose the concrete Rotation return type for composition.
    composed = cast(Rotation, first * second)
    residual = composed.magnitude()
    events = large & (residual < 0.25 * (magnitudes[:-1] + magnitudes[1:]))
    return np.flatnonzero(events).astype(int).tolist()


def _interleaved_hold_indices(
    increments: np.ndarray,
    interval_valid: np.ndarray,
    *,
    minimum_neighbor_magnitude: float,
    maximum_hold_ratio: float,
) -> list[int]:
    """Find high -> near-zero -> high pulse trains with coherent direction.

    This targets lower-rate telemetry copied into a faster frame table. A
    single stop/start is not enough to quarantine an episode; policy applies a
    separate minimum event count across both arms and pose channels.
    """

    if len(increments) < 3:
        return []
    magnitudes = np.linalg.norm(increments, axis=1)
    before = increments[:-2]
    after = increments[2:]
    before_magnitude = magnitudes[:-2]
    middle_magnitude = magnitudes[1:-1]
    after_magnitude = magnitudes[2:]
    denominator = np.maximum(before_magnitude * after_magnitude, 1e-12)
    direction_cosine = np.einsum("ij,ij->i", before, after) / denominator
    coherent = direction_cosine > 0.5
    neighboring_motion = np.minimum(before_magnitude, after_magnitude)
    events = (
        interval_valid[:-2]
        & interval_valid[1:-1]
        & interval_valid[2:]
        & coherent
        & (neighboring_motion >= minimum_neighbor_magnitude)
        & (middle_magnitude <= maximum_hold_ratio * neighboring_motion)
    )
    return (np.flatnonzero(events) + 1).astype(int).tolist()


def _derivative_metrics(
    values: np.ndarray,
    timestamps_ns: np.ndarray,
    interval_valid: np.ndarray,
) -> dict[str, float | None]:
    """Return rate, acceleration, and jerk norms using actual sample spacing."""

    if not len(values):
        return {
            "maximum_rate": None,
            "rate_robust_z": 0.0,
            "maximum_acceleration": None,
            "acceleration_robust_z": 0.0,
            "maximum_jerk": None,
            "jerk_robust_z": 0.0,
        }
    dt_s = np.diff(timestamps_ns).astype(np.float64) / 1e9
    if len(dt_s) != len(values) or np.any(dt_s <= 0):
        return {
            "maximum_rate": None,
            "rate_robust_z": 0.0,
            "maximum_acceleration": None,
            "acceleration_robust_z": 0.0,
            "maximum_jerk": None,
            "jerk_robust_z": 0.0,
        }
    rate_vectors = values / dt_s[:, None]
    rate = np.linalg.norm(rate_vectors, axis=1)[interval_valid]
    if len(rate_vectors) > 1:
        rate_time = (timestamps_ns[:-1] + timestamps_ns[1:]) / 2.0
        acceleration_dt = np.diff(rate_time) / 1e9
        acceleration_vectors = np.diff(rate_vectors, axis=0) / acceleration_dt[:, None]
        acceleration_valid = interval_valid[:-1] & interval_valid[1:]
        acceleration = np.linalg.norm(acceleration_vectors, axis=1)[acceleration_valid]
    else:
        acceleration_vectors = np.empty((0, values.shape[1]), dtype=np.float64)
        acceleration = np.empty(0, dtype=np.float64)
        rate_time = np.empty(0, dtype=np.float64)
        acceleration_valid = np.empty(0, dtype=np.bool_)
    if len(acceleration_vectors) > 1:
        acceleration_time = (rate_time[:-1] + rate_time[1:]) / 2.0
        jerk_dt = np.diff(acceleration_time) / 1e9
        jerk_valid = acceleration_valid[:-1] & acceleration_valid[1:]
        jerk = np.linalg.norm(np.diff(acceleration_vectors, axis=0) / jerk_dt[:, None], axis=1)[
            jerk_valid
        ]
    else:
        jerk = np.empty(0, dtype=np.float64)
    return {
        "maximum_rate": float(np.max(rate)) if len(rate) else None,
        "rate_robust_z": _robust_z_max(rate),
        "maximum_acceleration": float(np.max(acceleration)) if len(acceleration) else None,
        "acceleration_robust_z": _robust_z_max(acceleration),
        "maximum_jerk": float(np.max(jerk)) if len(jerk) else None,
        "jerk_robust_z": _robust_z_max(jerk),
    }


def _motion_metrics(
    actions_global: np.ndarray,
    action_valid: np.ndarray,
    state_timestamps_ns: Sequence[int],
    *,
    static_translation_m: float,
    static_rotation_rad: float,
    static_gripper_delta: float,
    jump_relative_robust_z: float,
    minimum_jump_translation_m: float,
    minimum_jump_rotation_rad: float,
    maximum_jump_interval_ratio: float,
    minimum_smoothness_translation_m: float,
    minimum_smoothness_rotation_rad: float,
    maximum_interleaved_hold_ratio: float,
) -> dict[str, Any]:
    """Measure fixed-world motion without differencing changing local frames."""

    timestamps = np.asarray(state_timestamps_ns, dtype=np.int64)
    arms: dict[str, Any] = {}
    all_translation_rates: list[np.ndarray] = []
    all_rotation_rates: list[np.ndarray] = []
    active = np.zeros(len(actions_global), dtype=np.bool_)
    for arm, offset in zip(("left", "right"), ARM_OFFSETS, strict=True):
        translation_valid = action_valid[:, offset : offset + 3].all(axis=1)
        rotation_valid = action_valid[:, offset + 3 : offset + 6].all(axis=1)
        gripper_valid = action_valid[:, offset + 6]
        translation = actions_global[:, offset : offset + 3]
        rotation = actions_global[:, offset + 3 : offset + 6]
        gripper = np.abs(actions_global[:, offset + 6])
        dt_s = np.diff(timestamps).astype(np.float64) / 1e9
        translation_rate = np.linalg.norm(translation, axis=1) / np.maximum(dt_s, 1e-12)
        rotation_rate = np.linalg.norm(rotation, axis=1) / np.maximum(dt_s, 1e-12)
        all_translation_rates.append(translation_rate[translation_valid])
        all_rotation_rates.append(rotation_rate[rotation_valid])
        active |= (
            (translation_valid & (np.linalg.norm(translation, axis=1) > static_translation_m))
            | (rotation_valid & (np.linalg.norm(rotation, axis=1) > static_rotation_rad))
            | (gripper_valid & (gripper > static_gripper_delta))
        )
        translation_jump_indices = _jump_return_indices(
            translation,
            timestamps,
            translation_valid,
            relative_robust_z=jump_relative_robust_z,
            minimum_magnitude=minimum_jump_translation_m,
            maximum_interval_ratio=maximum_jump_interval_ratio,
        )
        rotation_jump_indices = _rotation_jump_return_indices(
            rotation,
            timestamps,
            rotation_valid,
            relative_robust_z=jump_relative_robust_z,
            minimum_magnitude=minimum_jump_rotation_rad,
            maximum_interval_ratio=maximum_jump_interval_ratio,
        )
        translation_hold_indices = _interleaved_hold_indices(
            translation,
            translation_valid,
            minimum_neighbor_magnitude=minimum_smoothness_translation_m,
            maximum_hold_ratio=maximum_interleaved_hold_ratio,
        )
        rotation_hold_indices = _interleaved_hold_indices(
            rotation,
            rotation_valid,
            minimum_neighbor_magnitude=minimum_smoothness_rotation_rad,
            maximum_hold_ratio=maximum_interleaved_hold_ratio,
        )
        arms[arm] = {
            "translation": _derivative_metrics(translation, timestamps, translation_valid),
            "rotation": _derivative_metrics(rotation, timestamps, rotation_valid),
            "translation_jump_return_count": len(translation_jump_indices),
            "translation_jump_return_indices": translation_jump_indices,
            "rotation_jump_return_count": len(rotation_jump_indices),
            "rotation_jump_return_indices": rotation_jump_indices,
            "translation_interleaved_hold_count": len(translation_hold_indices),
            "translation_interleaved_hold_indices": translation_hold_indices,
            "rotation_interleaved_hold_count": len(rotation_hold_indices),
            "rotation_interleaved_hold_indices": rotation_hold_indices,
            "largest_translation_index": (
                int(np.argmax(np.where(translation_valid, translation_rate, -np.inf)))
                if translation_valid.any()
                else None
            ),
            "largest_rotation_index": (
                int(np.argmax(np.where(rotation_valid, rotation_rate, -np.inf)))
                if rotation_valid.any()
                else None
            ),
        }
    translation_rates = (
        np.concatenate(all_translation_rates)
        if any(len(values) for values in all_translation_rates)
        else np.empty(0)
    )
    rotation_rates = (
        np.concatenate(all_rotation_rates)
        if any(len(values) for values in all_rotation_rates)
        else np.empty(0)
    )
    return {
        "arms": arms,
        "maximum_translation_speed_m_s": (
            float(np.max(translation_rates)) if len(translation_rates) else None
        ),
        "maximum_rotation_speed_rad_s": (
            float(np.max(rotation_rates)) if len(rotation_rates) else None
        ),
        "maximum_translation_speed_robust_z": _robust_z_max(translation_rates),
        "maximum_rotation_speed_robust_z": _robust_z_max(rotation_rates),
        "static_fraction": float(1.0 - active.mean()) if len(active) else 1.0,
    }


def _gripper_integral_checks(
    robot: Mapping[str, Any],
    states: np.ndarray,
    state_valid: np.ndarray,
    policy: FilterPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay original/smoothed gripper deltas over every contiguous valid run."""

    metrics: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    for variant, key in (
        ("original", "actions_local"),
        ("smoothed", "smoothed_actions_local"),
    ):
        payload = robot.get(key)
        if payload is None:
            continue
        actions = np.asarray(payload, dtype=np.float64)
        valid = np.asarray(robot["action_valid_local"], dtype=np.bool_)
        for arm, offset in zip(("left", "right"), ARM_OFFSETS, strict=True):
            gripper_index = offset + 6
            running: float | None = None
            values: list[float] = []
            run_starts: list[dict[str, float | int]] = []
            bad: list[int] = []
            for index in range(len(actions)):
                interval_valid = bool(valid[index, gripper_index])
                start_valid = bool(state_valid[index, gripper_index])
                if not interval_valid or not start_valid:
                    running = None
                    continue
                if running is None:
                    running = float(states[index, gripper_index])
                    run_starts.append(
                        {
                            "action_index": index,
                            "initial_open_fraction": running,
                        }
                    )
                running += float(actions[index, gripper_index])
                values.append(running)
                if running < -policy.gripper_integral_tolerance or running > (
                    1.0 + policy.gripper_integral_tolerance
                ):
                    bad.append(index)
            metrics[f"{variant}_{arm}"] = {
                "minimum_replayed_open_fraction": min(values) if values else None,
                "maximum_replayed_open_fraction": max(values) if values else None,
                "out_of_range_count": len(bad),
                "out_of_range_indices": bad,
                "valid_run_starts": run_starts,
            }
            evidence.extend(
                _event_spans(
                    "gripper_action_integral_out_of_range",
                    bad,
                    arm=arm,
                    detail=variant,
                )
            )
    return metrics, evidence


def gripper_sensor_bug_checks(
    robot: Mapping[str, Any],
    policy: FilterPolicy | None = None,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Inspect unsmoothed canonical deltas; flag both offending action intervals.

    Pair i,i+1 shares state frame i+1. Evidence records that pivot explicitly.
    Magnitudes are in the stored gripper unit (not silently rescaled by FPS).
    Invalid/nonfinite/absent channels never count as observations.
    """
    threshold = (policy or FilterPolicy()).gripper_sensor_bug_minimum_delta
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("gripper sensor threshold must be positive and finite")
    actions = np.asarray(robot.get("actions_local"), dtype=np.float64)
    valid = np.asarray(robot.get("action_valid_local"), dtype=np.bool_)
    metrics: dict[str, Any] = {
        "threshold": threshold,
        "unit": robot.get("gripper_unit"),
        "signal": "unsmoothed_actions_local",
        "version": 1,
    }
    reasons, events = [], []
    if actions.ndim != 2 or actions.shape[1:] != (14,) or valid.shape != actions.shape:
        metrics["status"] = "unavailable"
        return metrics, reasons, events
    for arm, column in (("left", 6), ("right", 13)):
        x = actions[:, column]
        usable = valid[:, column] & np.isfinite(x)
        pair = (
            usable[:-1]
            & usable[1:]
            & (np.abs(x[:-1]) >= threshold)
            & (np.abs(x[1:]) >= threshold)
            & (np.signbit(x[:-1]) != np.signbit(x[1:]))
        )
        starts = np.flatnonzero(pair)
        metrics[arm] = {"pair_start_indices": starts.tolist(), "count": len(starts)}
        reason = f"{arm}_gripper_sensor_bug"
        if len(starts):
            reasons.append(reason)
        for i in starts:
            events.append(
                {
                    "reason": reason,
                    "arm": arm,
                    "start_frame": int(i),
                    "end_frame": int(i + 2),
                    "pivot_frame": int(i + 1),
                    "first_delta": float(x[i]),
                    "second_delta": float(x[i + 1]),
                    "unit": robot.get("gripper_unit"),
                }
            )
    return metrics, reasons, events


def _inactive_gripper_checks(
    robot: Mapping[str, Any],
    policy: FilterPolicy,
) -> tuple[dict[str, Any], list[str]]:
    """Flag a recorded arm whose gripper aperture never changes.

    Arms without any valid gripper samples are absent, not inactive. This keeps
    one-arm UMI episodes from acquiring a false right-gripper quarantine.
    """

    actions = np.asarray(robot["actions_local"], dtype=np.float64)
    valid = np.asarray(robot["action_valid_local"], dtype=np.bool_)
    metrics: dict[str, Any] = {}
    reasons: list[str] = []
    for arm, offset in zip(("left", "right"), ARM_OFFSETS, strict=True):
        gripper_index = offset + 6
        arm_valid = valid[:, gripper_index]
        values = np.abs(actions[arm_valid, gripper_index])
        active_count = int(np.count_nonzero(values > policy.static_gripper_delta))
        metrics[arm] = {
            "status": "recorded" if len(values) else "absent",
            "valid_action_count": int(len(values)),
            "active_action_count": active_count,
            "maximum_absolute_delta": float(values.max()) if len(values) else None,
        }
        if len(values) and active_count == 0:
            reasons.append(f"{arm}_gripper_never_acts")
    return metrics, reasons


def _largest_action_video_checks(
    actions_global: np.ndarray,
    action_valid: np.ndarray,
    camera_pair_differences: Mapping[str, Sequence[float]],
    policy: FilterPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare four maximum pose deltas with their adjacent video frames.

    This is deliberately conservative. A check is eligible only when the
    action is physically large and a robust within-trajectory outlier. It is
    unsupported only when *every* available camera pair is both in the lowest
    15% of that camera's motion and below absolute/relative pixel thresholds.
    """

    checks: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    for arm, offset in zip(("left", "right"), ARM_OFFSETS, strict=True):
        for signal, signal_slice, absolute_floor in (
            ("translation", slice(offset, offset + 3), policy.minimum_jump_translation_m),
            ("rotation", slice(offset + 3, offset + 6), policy.minimum_jump_rotation_rad),
        ):
            valid = action_valid[:, signal_slice].all(axis=1)
            magnitudes = np.linalg.norm(actions_global[:, signal_slice], axis=1)
            if not valid.any():
                checks[f"{arm}_{signal}"] = {"status": "unavailable"}
                continue
            valid_magnitudes = magnitudes[valid]
            median = float(np.median(valid_magnitudes))
            mad = float(np.median(np.abs(valid_magnitudes - median)))
            robust_threshold = median + 10.0 * 1.4826 * max(mad, 1e-12)
            index = int(np.argmax(np.where(valid, magnitudes, -np.inf)))
            magnitude = float(magnitudes[index])
            eligible = magnitude >= max(absolute_floor, robust_threshold)
            camera_evidence: dict[str, Any] = {}
            unsupported_by_every_camera = True
            usable_cameras = 0
            for camera_name, values in camera_pair_differences.items():
                differences = np.asarray(values, dtype=np.float64)
                if index >= len(differences) or not len(differences):
                    continue
                # Modification 4: no camera observation in this interval -> camera not usable here.
                if not np.isfinite(differences[index]):
                    continue
                differences = differences[np.isfinite(differences)]
                usable_cameras += 1
                difference = float(values[index])
                median_difference = float(np.median(differences))
                percentile = float(np.mean(differences <= difference))
                median_ratio = difference / max(median_difference, 1e-6)
                camera_supports_motion = not (
                    percentile <= policy.largest_action_visual_percentile
                    and median_ratio <= policy.largest_action_visual_median_ratio
                    and difference <= policy.largest_action_visual_absolute_difference
                )
                unsupported_by_every_camera &= not camera_supports_motion
                camera_evidence[camera_name] = {
                    "mean_absolute_luma_difference": difference,
                    "within_camera_percentile": percentile,
                    "median_ratio": median_ratio,
                    "supports_motion": camera_supports_motion,
                }
            unsupported = eligible and usable_cameras > 0 and unsupported_by_every_camera
            key = f"{arm}_{signal}"
            checks[key] = {
                "status": "unsupported" if unsupported else "supported_or_inconclusive",
                "action_index": index,
                "magnitude_m" if signal == "translation" else "magnitude_rad": magnitude,
                "robust_outlier_threshold": robust_threshold,
                "absolute_floor": absolute_floor,
                "eligible": eligible,
                "cameras": camera_evidence,
            }
            if unsupported:
                evidence.extend(
                    _event_spans(
                        "largest_action_not_in_video",
                        [index],
                        arm=arm,
                        detail=signal,
                    )
                )
    return checks, evidence


def _unexplained_visual_change_checks(
    actions_global: np.ndarray,
    action_valid: np.ndarray,
    camera_pair_differences: Mapping[str, Sequence[float]],
    policy: FilterPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Locate large image changes whose canonical action is effectively a hold.

    The image statistic is the mean absolute luma difference on the existing
    64x64 QC decode.  An interval is eligible only when both recorded arms are
    below conservative translation, rotation, and gripper thresholds.  This
    intentionally detects exogenous scene motion and action/video mismatch;
    it does not claim which object moved.
    """

    count = len(actions_global)
    tiny = np.ones(count, dtype=np.bool_)
    for _arm, offset in zip(("left", "right"), ARM_OFFSETS, strict=True):
        translation_valid = action_valid[:, offset : offset + 3].all(axis=1)
        rotation_valid = action_valid[:, offset + 3 : offset + 6].all(axis=1)
        gripper_valid = action_valid[:, offset + 6]
        translation = np.linalg.norm(actions_global[:, offset : offset + 3], axis=1)
        rotation = np.linalg.norm(actions_global[:, offset + 3 : offset + 6], axis=1)
        gripper = np.abs(actions_global[:, offset + 6])
        arm_present = translation_valid | rotation_valid | gripper_valid
        arm_tiny = ~translation_valid | (translation <= policy.maximum_unexplained_translation_m)
        arm_tiny &= ~rotation_valid | (rotation <= policy.maximum_unexplained_rotation_rad)
        arm_tiny &= ~gripper_valid | (gripper <= policy.maximum_unexplained_gripper_delta)
        tiny &= ~arm_present | arm_tiny
    checks: dict[str, Any] = {
        "thresholds": {
            "mean_absolute_luma_difference": policy.minimum_unexplained_visual_difference,
            "translation_m": policy.maximum_unexplained_translation_m,
            "rotation_rad": policy.maximum_unexplained_rotation_rad,
            "gripper_delta": policy.maximum_unexplained_gripper_delta,
        },
        "cameras": {},
    }
    evidence: list[dict[str, Any]] = []
    for camera_name, values in camera_pair_differences.items():
        differences = np.asarray(values, dtype=np.float64)
        usable = min(count, len(differences))
        indices = np.flatnonzero(
            tiny[:usable] & (differences[:usable] >= policy.minimum_unexplained_visual_difference)
        ).astype(int)
        checks["cameras"][camera_name] = {
            "pair_count": int(usable),
            "flagged_count": int(len(indices)),
            "flagged_indices": indices.tolist(),
            # Modification 5: finite values only.
            "mean_pair_difference": float(np.nanmean(differences[:usable])) if usable else None,
            "p95_pair_difference": (
                float(np.nanquantile(differences[:usable], 0.95)) if usable else None
            ),
        }
        evidence.extend(
            _event_spans(
                "visual_change_unexplained_by_action",
                indices.tolist(),
                camera=camera_name,
                detail="large luma change while both canonical arms hold",
            )
        )
    return checks, evidence


# ---------------------------------------------------------------- adapters/common/processing.py

ARM_SLICES = ((slice(0, 3), slice(3, 6)), (slice(7, 10), slice(10, 13)))
ARM_OFFSETS = (0, 7)
SMOOTHING_REFERENCE_FPS = 30.0
SMOOTHING_POSITION_STRENGTH = 6.0
SMOOTHING_ROTATION_STRENGTH = 12.0
SMOOTHING_GRIPPER_STRENGTH = 6.0
SMOOTHING_MAX_POSITION_DEVIATION_M = 0.02
SMOOTHING_MAX_ROTATION_DEVIATION_RAD = 0.15
SMOOTHING_MAX_GRIPPER_DEVIATION = 0.15
SMOOTHING_VERSION = "se3-trajectory-smoothing-v3"


def median_period_ns(timestamps_ns: Sequence[int] | np.ndarray) -> float:
    values = np.asarray(timestamps_ns, dtype=np.int64)
    differences = np.diff(values)
    positive = differences[differences > 0]
    if not len(positive):
        raise EpisodeDrop("timestamp stream has no positive frame interval")
    return float(np.median(positive))


def delta_actions(
    states: np.ndarray, state_valid: np.ndarray, timestamps_ns: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    local_actions = np.diff(states, axis=0)
    global_actions = local_actions.copy()
    valid = state_valid[:-1] & state_valid[1:]
    for translation_slice, rotation_slice in ARM_SLICES:
        previous = Rotation.from_rotvec(states[:-1, rotation_slice])
        following = Rotation.from_rotvec(states[1:, rotation_slice])
        world_translation = states[1:, translation_slice] - states[:-1, translation_slice]
        local_actions[:, translation_slice] = previous.inv().apply(world_translation)
        # SAFETY: scipy's multiplication stubs erase the concrete Rotation result type.
        local_relative = cast(Rotation, previous.inv() * following)
        local_actions[:, rotation_slice] = local_relative.as_rotvec()
        global_actions[:, translation_slice] = world_translation
        # SAFETY: scipy's multiplication stubs erase the concrete Rotation result type.
        global_relative = cast(Rotation, following * previous.inv())
        global_actions[:, rotation_slice] = global_relative.as_rotvec()
    local_actions[~valid] = 0.0
    global_actions[~valid] = 0.0
    timestamps = [
        (int(left) + int(right)) // 2
        for left, right in zip(timestamps_ns, timestamps_ns[1:], strict=False)
    ]
    return (
        local_actions.astype(np.float32),
        global_actions.astype(np.float32),
        valid,
        timestamps,
    )


def _valid_runs(mask: np.ndarray, *, minimum_length: int = 4) -> list[slice]:
    """Return contiguous true runs long enough for endpoint-constrained smoothing."""

    runs: list[slice] = []
    start = 0
    while start < len(mask):
        while start < len(mask) and not mask[start]:
            start += 1
        stop = start
        while stop < len(mask) and mask[stop]:
            stop += 1
        if stop - start >= minimum_length:
            runs.append(slice(start, stop))
        start = max(stop, start + 1)
    return runs


def _smooth_euclidean_path(values: np.ndarray, strength: float) -> np.ndarray:
    """Second-difference Tikhonov smoothing with both endpoints fixed exactly."""

    source = np.asarray(values, dtype=np.float64)
    if len(source) < 4 or strength <= 0.0:
        return source.copy()
    difference = lil_matrix((len(source) - 2, len(source)), dtype=np.float64)
    for row in range(len(source) - 2):
        difference[row, row : row + 3] = (1.0, -2.0, 1.0)
    system = (sparse_eye(len(source), format="csc") + strength * difference.T @ difference).tolil()
    right_hand_side = source.copy()
    for endpoint in (0, len(source) - 1):
        system.rows[endpoint] = [endpoint]
        system.data[endpoint] = [1.0]
        right_hand_side[endpoint] = source[endpoint]
    result = np.asarray(spsolve(system.tocsc(), right_hand_side), dtype=np.float64)
    result[0] = source[0]
    result[-1] = source[-1]
    return result


def _smooth_rotation_path(
    rotations: Rotation,
    strength: float,
    *,
    iterations: int = 64,
    relaxation: float = 0.6,
) -> Rotation:
    """Regularize one SO(3) path with fixed endpoints and geodesic fidelity.

    Each Jacobi step moves an interior orientation toward the geodesic midpoint
    of its neighbors while pulling it back toward the published orientation.
    All updates are applied in the current tangent space, so no quaternion or
    Euler subtraction is used.
    """

    if len(rotations) < 4 or strength <= 0.0:
        return Rotation.from_quat(rotations.as_quat())
    original = Rotation.from_quat(rotations.as_quat())
    current = Rotation.from_quat(rotations.as_quat())
    for _ in range(iterations):
        previous = current[:-2]
        center = current[1:-1]
        following = current[2:]
        # SAFETY: scipy's multiplication stubs erase the concrete Rotation result type.
        neighbor_delta = cast(Rotation, previous.inv() * following).as_rotvec()
        # SAFETY: scipy's multiplication stubs erase the concrete Rotation result type.
        midpoint = cast(Rotation, previous * Rotation.from_rotvec(0.5 * neighbor_delta))
        # SAFETY: scipy's multiplication stubs erase the concrete Rotation result type.
        fidelity = cast(Rotation, center.inv() * original[1:-1]).as_rotvec()
        # SAFETY: scipy's multiplication stubs erase the concrete Rotation result type.
        smoothness = cast(Rotation, center.inv() * midpoint).as_rotvec()
        correction = relaxation * (fidelity + strength * smoothness) / (1.0 + strength)
        # SAFETY: scipy's multiplication stubs erase the concrete Rotation result type.
        updated_center = cast(Rotation, center * Rotation.from_rotvec(correction))
        quaternions = current.as_quat()
        quaternions[1:-1] = updated_center.as_quat()
        quaternions[0] = original[0].as_quat()
        quaternions[-1] = original[-1].as_quat()
        current = Rotation.from_quat(quaternions)
    return current


def smooth_canonical_trajectory(
    states: np.ndarray,
    state_valid: np.ndarray,
    timestamps_ns: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a lightly regularized SE(3)+gripper trajectory and diagnostics.

    The optimization is performed on states, then actions are re-derived. This
    guarantees that replay is internally consistent and that every valid run
    retains its exact first/final pose and gripper aperture. The smoothing
    strength is normalized to physical time relative to 30 Hz.
    """

    source = np.asarray(states, dtype=np.float64)
    valid = np.asarray(state_valid, dtype=np.bool_)
    if source.shape != valid.shape or source.ndim != 2 or source.shape[1] != ACTION_WIDTH:
        raise EpisodeDrop(f"cannot smooth malformed states {source.shape}/{valid.shape}")
    period_ns = median_period_ns(timestamps_ns)
    rate_hz = 1e9 / period_ns
    cadence_scale = float(np.clip((rate_hz / SMOOTHING_REFERENCE_FPS) ** 4, 0.05, 32.0))
    output = source.copy()
    run_counts = {"pose": 0, "gripper": 0}
    for offset in ARM_OFFSETS:
        pose_valid = valid[:, offset : offset + 6].all(axis=1)
        for run in _valid_runs(pose_valid):
            original_position = source[run, offset : offset + 3]
            smoothed_position = _smooth_euclidean_path(
                source[run, offset : offset + 3],
                SMOOTHING_POSITION_STRENGTH * cadence_scale,
            )
            position_correction = smoothed_position - original_position
            maximum_position_correction = float(np.max(np.linalg.norm(position_correction, axis=1)))
            if maximum_position_correction > SMOOTHING_MAX_POSITION_DEVIATION_M:
                position_correction *= (
                    SMOOTHING_MAX_POSITION_DEVIATION_M / maximum_position_correction
                )
            output[run, offset : offset + 3] = original_position + position_correction
            original_rotation = Rotation.from_rotvec(source[run, offset + 3 : offset + 6])
            smoothed_rotation = _smooth_rotation_path(
                original_rotation,
                SMOOTHING_ROTATION_STRENGTH * cadence_scale,
            )
            # SAFETY: scipy's multiplication stubs erase the concrete Rotation result type.
            rotation_correction = cast(
                Rotation, original_rotation.inv() * smoothed_rotation
            ).as_rotvec()
            maximum_rotation_correction = float(np.max(np.linalg.norm(rotation_correction, axis=1)))
            if maximum_rotation_correction > SMOOTHING_MAX_ROTATION_DEVIATION_RAD:
                rotation_correction *= (
                    SMOOTHING_MAX_ROTATION_DEVIATION_RAD / maximum_rotation_correction
                )
            # SAFETY: scipy's multiplication stubs erase the concrete Rotation result type.
            output[run, offset + 3 : offset + 6] = cast(
                Rotation, original_rotation * Rotation.from_rotvec(rotation_correction)
            ).as_rotvec()
            run_counts["pose"] += 1
        gripper_valid = valid[:, offset + 6]
        for run in _valid_runs(gripper_valid):
            gripper = _smooth_euclidean_path(
                source[run, offset + 6],
                SMOOTHING_GRIPPER_STRENGTH * cadence_scale,
            )
            gripper_correction = gripper - source[run, offset + 6]
            maximum_gripper_correction = float(np.max(np.abs(gripper_correction)))
            if maximum_gripper_correction > SMOOTHING_MAX_GRIPPER_DEVIATION:
                gripper_correction *= SMOOTHING_MAX_GRIPPER_DEVIATION / maximum_gripper_correction
            gripper = source[run, offset + 6] + gripper_correction
            output[run, offset + 6] = np.clip(gripper, 0.0, 1.0)
            output[run.start, offset + 6] = source[run.start, offset + 6]
            output[run.stop - 1, offset + 6] = source[run.stop - 1, offset + 6]
            run_counts["gripper"] += 1
    output[~valid] = 0.0
    position_errors: list[np.ndarray] = []
    rotation_errors: list[np.ndarray] = []
    gripper_errors: list[np.ndarray] = []
    for offset in ARM_OFFSETS:
        position_mask = valid[:, offset : offset + 3].all(axis=1)
        if position_mask.any():
            position_errors.append(
                np.linalg.norm(
                    output[position_mask, offset : offset + 3]
                    - source[position_mask, offset : offset + 3],
                    axis=1,
                )
            )
        rotation_mask = valid[:, offset + 3 : offset + 6].all(axis=1)
        if rotation_mask.any():
            # SAFETY: scipy's multiplication stubs erase the concrete Rotation result type.
            rotation_errors.append(
                cast(
                    Rotation,
                    Rotation.from_rotvec(source[rotation_mask, offset + 3 : offset + 6]).inv()
                    * Rotation.from_rotvec(output[rotation_mask, offset + 3 : offset + 6]),
                ).magnitude()
            )
        gripper_mask = valid[:, offset + 6]
        if gripper_mask.any():
            gripper_errors.append(
                np.abs(output[gripper_mask, offset + 6] - source[gripper_mask, offset + 6])
            )

    def error_summary(parts: list[np.ndarray]) -> dict[str, float]:
        values = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
        return {
            "rms": float(np.sqrt(np.mean(values * values))) if len(values) else 0.0,
            "maximum": float(np.max(values)) if len(values) else 0.0,
        }

    return output.astype(np.float32), {
        "version": SMOOTHING_VERSION,
        "method": "endpoint-constrained discrete SE(3) trajectory regularization",
        "rate_hz": rate_hz,
        "cadence_scale": cadence_scale,
        "strength_at_30_hz": {
            "position": SMOOTHING_POSITION_STRENGTH,
            "rotation": SMOOTHING_ROTATION_STRENGTH,
            "gripper": SMOOTHING_GRIPPER_STRENGTH,
        },
        "valid_runs": run_counts,
        "position_deviation_m": error_summary(position_errors),
        "rotation_geodesic_deviation_rad": error_summary(rotation_errors),
        "gripper_deviation": error_summary(gripper_errors),
        "endpoint_policy": "first and final sample of every contiguous valid run fixed exactly",
        "maximum_allowed_deviation": {
            "position_m": SMOOTHING_MAX_POSITION_DEVIATION_M,
            "rotation_rad": SMOOTHING_MAX_ROTATION_DEVIATION_RAD,
            "gripper_open_fraction": SMOOTHING_MAX_GRIPPER_DEVIATION,
        },
    }
