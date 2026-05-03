from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HEAD_COLUMNS = ["head_x", "head_y", "head_z"]
BODY_COLUMNS = ["body_x", "body_y", "body_z"]
TAIL_COLUMNS = ["tail_x", "tail_y", "tail_z"]
QUALITY_COLUMNS = [
    "confidence",
    "triangulation_error",
    "track_quality",
    "interpolated",
    "outlier_corrected",
]


def compute_all_features(
    df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    feature_df = df.copy()
    feature_df["v"] = compute_velocity(feature_df, config)
    feature_df["p"] = compute_pose_change(feature_df, config)
    feature_df["c"] = compute_curvature_change(feature_df, config)
    feature_df["d_win"] = compute_window_displacement(feature_df, config)
    feature_df["E_move"] = compute_motion_energy(feature_df, config)
    feature_df["S_pose"] = compute_pose_stability(feature_df, config)
    feature_df["dv"] = compute_velocity_drop(feature_df, config)
    maybe_save_feature_table(feature_df, config, output_dir)
    return feature_df


def compute_velocity(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    return _group_apply_series(df, _compute_velocity_for_group)


def compute_pose_change(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    def calculate(group: pd.DataFrame) -> pd.Series:
        dt = group["timestamp"].diff().replace(0, np.nan)
        point_displacements = []
        for cols in [HEAD_COLUMNS, BODY_COLUMNS, TAIL_COLUMNS]:
            disp = _point_displacement(group, cols) / dt
            point_displacements.append(disp)
        result = pd.concat(point_displacements, axis=1).mean(axis=1)
        return result.fillna(0.0)

    return _group_apply_series(df, calculate)


def compute_curvature_change(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    def calculate(group: pd.DataFrame) -> pd.Series:
        angles = _compute_body_angles(group)
        dt = group["timestamp"].diff().replace(0, np.nan)
        change_rate = angles.diff().abs() / dt
        return change_rate.fillna(0.0)

    return _group_apply_series(df, calculate)


def compute_window_displacement(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    window_seconds = config.get("features", {}).get("displacement_window_seconds", 1.0)

    def calculate(group: pd.DataFrame) -> pd.Series:
        return _compute_window_metric(group, window_seconds, _compute_valid_body_window_displacement)

    return _group_apply_series(df, calculate)


def compute_motion_energy(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    window_seconds = config.get("features", {}).get("energy_window_seconds", 1.0)
    velocity = df["v"] if "v" in df.columns else compute_velocity(df, config)
    working = df.copy()
    working["v"] = velocity

    def calculate(group: pd.DataFrame) -> pd.Series:
        return _compute_window_metric(group, window_seconds, lambda window: float(np.mean(np.square(window["v"]))))

    return _group_apply_series(working, calculate)


def compute_pose_stability(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    window_seconds = config.get("features", {}).get("stability_window_seconds", 1.0)
    working = df.copy()
    if "p" not in working.columns:
        working["p"] = compute_pose_change(working, config)
    if "c" not in working.columns:
        working["c"] = compute_curvature_change(working, config)

    raw_stability = _group_apply_series(
        working,
        lambda group: _compute_window_metric(
            group,
            window_seconds,
            lambda window: float(window["p"].std(ddof=0) + window["c"].std(ddof=0)),
        ),
    )

    finite_values = raw_stability.replace([np.inf, -np.inf], np.nan).dropna()
    if finite_values.empty:
        return pd.Series(1.0, index=working.index)

    min_val = finite_values.min()
    max_val = finite_values.max()
    if np.isclose(max_val, min_val):
        return pd.Series(1.0, index=working.index)

    normalized = (raw_stability - min_val) / (max_val - min_val)
    stability = 1.0 - normalized
    return stability.fillna(1.0).clip(0.0, 1.0)


def compute_velocity_drop(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    window_seconds = config.get("features", {}).get("displacement_window_seconds", 1.0)
    velocity = df["v"] if "v" in df.columns else compute_velocity(df, config)
    working = df.copy()
    working["v"] = velocity

    def calculate(group: pd.DataFrame) -> pd.Series:
        return _compute_window_metric(group, window_seconds, lambda window: float(window["v"].iloc[0] - window["v"].iloc[-1]))

    return _group_apply_series(working, calculate)


def maybe_save_feature_table(
    df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> Path | None:
    output_config = config.get("output", {})
    if not output_config.get("save_feature_table", False):
        return None

    base_dir = Path(output_dir) if output_dir is not None else Path(config.get("paths", {}).get("output_dir", "outputs"))
    save_dir = base_dir / "intermediate" / "features"
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = output_config.get("feature_table_file", "feature_table.csv")
    output_path = save_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


def _group_apply_series(df: pd.DataFrame, func) -> pd.Series:
    result = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby("fish_id", sort=False):
        result.loc[group.index] = func(group)
    return result


def _compute_velocity_for_group(group: pd.DataFrame) -> pd.Series:
    dt = group["timestamp"].diff().replace(0, np.nan)
    displacement = _point_displacement(group, HEAD_COLUMNS)
    velocity = displacement / dt
    return velocity.fillna(0.0)


def _point_displacement(group: pd.DataFrame, columns: list[str]) -> pd.Series:
    coords = group[columns]
    dx = coords.iloc[:, 0].diff()
    dy = coords.iloc[:, 1].diff()
    dz = coords.iloc[:, 2].diff()
    return np.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2))


def _compute_body_angles(group: pd.DataFrame) -> pd.Series:
    head = group[HEAD_COLUMNS].to_numpy(dtype=float)
    body = group[BODY_COLUMNS].to_numpy(dtype=float)
    tail = group[TAIL_COLUMNS].to_numpy(dtype=float)

    vec1 = head - body
    vec2 = tail - body

    dot = np.sum(vec1 * vec2, axis=1)
    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)
    denom = norm1 * norm2
    cos_theta = np.divide(dot, denom, out=np.ones_like(dot), where=denom != 0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_theta))
    return pd.Series(angles, index=group.index)


def _compute_window_metric(group: pd.DataFrame, window_seconds: float, aggregator) -> pd.Series:
    values = []
    timestamps = group["timestamp"].to_numpy(dtype=float)
    for i in range(len(group)):
        current_time = timestamps[i]
        start_time = current_time - window_seconds
        start_idx = np.searchsorted(timestamps, start_time, side="left")
        window = group.iloc[start_idx : i + 1]
        values.append(aggregator(window))
    return pd.Series(values, index=group.index)


def _compute_valid_body_window_displacement(window: pd.DataFrame) -> float:
    valid_points = window.dropna(subset=BODY_COLUMNS)
    if len(valid_points) < 2:
        return 0.0
    return _euclidean_distance(valid_points[BODY_COLUMNS].iloc[0], valid_points[BODY_COLUMNS].iloc[-1])


def _euclidean_distance(point_a: pd.Series, point_b: pd.Series) -> float:
    diff = point_b.to_numpy(dtype=float) - point_a.to_numpy(dtype=float)
    return float(np.linalg.norm(diff))
