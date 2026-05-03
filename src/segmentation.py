from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


DEFAULT_SEGMENT_COLUMNS = [
    "segment_id",
    "fish_id",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "duration",
    "n_frames",
    "mean_v",
    "mean_d_win",
    "mean_E_move",
    "mean_S_pose",
    "mean_p",
    "mean_c",
    "mean_dv",
]


def segment_behavior(
    feature_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    seg_feature_df = build_segmentation_features(feature_df, config)
    changepoints, change_score_df = detect_changepoints(seg_feature_df, config)
    candidate_df = build_candidate_segments(feature_df, changepoints, config)
    candidate_df = filter_short_segments(candidate_df, config)
    save_candidate_segments(candidate_df, config, output_dir)
    save_change_scores(change_score_df, config, output_dir)
    return candidate_df


def build_segmentation_features(feature_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    seg_config = config.get("segmentation", {})
    requested_features = seg_config.get("signal_features", ["v", "d_win", "E_move", "inv_S_pose"])

    seg_df = feature_df[["frame", "fish_id", "timestamp"]].copy()
    for feature in requested_features:
        if feature == "inv_S_pose":
            seg_df[feature] = 1.0 - feature_df["S_pose"]
        else:
            seg_df[feature] = feature_df[feature]

    if seg_config.get("normalize_features", True):
        seg_df = _normalize_segmentation_features(seg_df, requested_features, seg_config.get("normalization", "zscore"))

    return seg_df


def detect_changepoints(
    seg_feature_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, list[int]], pd.DataFrame]:
    seg_config = config.get("segmentation", {})
    requested_features = seg_config.get("signal_features", ["v", "d_win", "E_move", "inv_S_pose"])
    min_segment_length = int(seg_config.get("min_segment_length_frames", 15))
    penalty = float(seg_config.get("penalty", 5.0))

    results: dict[str, list[int]] = {}
    score_rows: list[dict[str, Any]] = []
    for fish_id, group in seg_feature_df.groupby("fish_id", sort=False):
        matrix = group[requested_features].to_numpy(dtype=float)
        scores = _compute_change_scores(matrix)
        score_threshold = _compute_score_threshold(scores, penalty)
        boundary_positions = _select_boundaries(
            scores,
            min_segment_length=min_segment_length,
            score_threshold=score_threshold,
            total_length=len(group),
        )
        boundary_set = set(boundary_positions)
        results[str(fish_id)] = boundary_positions

        for row_idx, (_, row) in enumerate(group.iterrows()):
            score_rows.append(
                {
                    "fish_id": str(fish_id),
                    "frame": int(row["frame"]),
                    "timestamp": float(row["timestamp"]),
                    "change_score": float(scores[row_idx]),
                    "score_threshold": float(score_threshold) if np.isfinite(score_threshold) else np.nan,
                    "is_boundary": row_idx in boundary_set,
                }
            )

    return results, pd.DataFrame(score_rows)


def build_candidate_segments(
    feature_df: pd.DataFrame,
    changepoints: dict[str, list[int]],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    segment_id = 0

    for fish_id, group in feature_df.groupby("fish_id", sort=False):
        boundaries = changepoints.get(str(fish_id), [])
        cut_points = [0] + boundaries + [len(group)]
        for start_idx, end_idx in zip(cut_points[:-1], cut_points[1:]):
            segment = group.iloc[start_idx:end_idx]
            if segment.empty:
                continue
            segment_id += 1
            rows.append(
                {
                    "segment_id": segment_id,
                    "fish_id": str(fish_id),
                    "start_frame": int(segment["frame"].iloc[0]),
                    "end_frame": int(segment["frame"].iloc[-1]),
                    "start_time": float(segment["timestamp"].iloc[0]),
                    "end_time": float(segment["timestamp"].iloc[-1]),
                    "duration": float(segment["timestamp"].iloc[-1] - segment["timestamp"].iloc[0]),
                    "n_frames": int(len(segment)),
                    "mean_v": float(segment["v"].mean()),
                    "mean_d_win": float(segment["d_win"].mean()),
                    "mean_E_move": float(segment["E_move"].mean()),
                    "mean_S_pose": float(segment["S_pose"].mean()),
                    "mean_p": float(segment["p"].mean()),
                    "mean_c": float(segment["c"].mean()),
                    "mean_dv": float(segment["dv"].mean()),
                }
            )

    if not rows:
        return pd.DataFrame(columns=DEFAULT_SEGMENT_COLUMNS)
    return pd.DataFrame(rows)


def filter_short_segments(candidate_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    min_duration = float(config.get("segmentation", {}).get("candidate_min_duration", 0.5))
    if candidate_df.empty:
        return candidate_df.copy()
    return candidate_df[candidate_df["duration"] >= min_duration].reset_index(drop=True)


def save_candidate_segments(
    candidate_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> Path:
    base_dir = Path(output_dir) if output_dir is not None else Path(config.get("paths", {}).get("output_dir", "outputs"))
    save_dir = base_dir / "intermediate" / "segmentation"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "candidate_segments.csv"
    candidate_df.to_csv(output_path, index=False)
    return output_path


def save_change_scores(
    change_score_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> Path | None:
    if not config.get("segmentation", {}).get("save_change_scores", False):
        return None

    base_dir = Path(output_dir) if output_dir is not None else Path(config.get("paths", {}).get("output_dir", "outputs"))
    save_dir = base_dir / "intermediate" / "segmentation"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "change_scores.csv"
    change_score_df.to_csv(output_path, index=False)
    return output_path


def _normalize_segmentation_features(seg_df: pd.DataFrame, feature_names: list[str], normalization: str) -> pd.DataFrame:
    normalized = seg_df.copy()
    for fish_id, group in normalized.groupby("fish_id", sort=False):
        values = group[feature_names].copy()
        if normalization == "zscore":
            means = values.mean(axis=0)
            stds = values.std(axis=0, ddof=0).replace(0, 1.0)
            values = (values - means) / stds
        elif normalization == "minmax":
            mins = values.min(axis=0)
            maxs = values.max(axis=0)
            spans = (maxs - mins).replace(0, 1.0)
            values = (values - mins) / spans
        normalized.loc[group.index, feature_names] = values.values
    return normalized


def _compute_change_scores(matrix: np.ndarray) -> np.ndarray:
    n = len(matrix)
    if n < 3:
        return np.zeros(n, dtype=float)

    scores = np.zeros(n, dtype=float)
    for idx in range(1, n - 1):
        prev_vec = matrix[idx - 1]
        next_vec = matrix[idx + 1]
        scores[idx] = float(np.linalg.norm(next_vec - prev_vec))
    return scores


def _compute_score_threshold(scores: np.ndarray, penalty: float) -> float:
    return float(scores.mean() + penalty * scores.std(ddof=0))


def _select_boundaries(scores: np.ndarray, min_segment_length: int, score_threshold: float, total_length: int) -> list[int]:
    if total_length <= min_segment_length * 2:
        return []

    if not np.isfinite(score_threshold) or score_threshold <= 0:
        return []

    peaks, properties = find_peaks(scores, height=score_threshold, distance=max(1, min_segment_length))
    if len(peaks) == 0:
        return []

    boundaries: list[int] = []
    last_boundary = 0
    for peak in peaks:
        boundary = int(peak)
        if boundary - last_boundary < min_segment_length:
            continue
        if total_length - boundary < min_segment_length:
            continue
        boundaries.append(boundary)
        last_boundary = boundary
    return boundaries
