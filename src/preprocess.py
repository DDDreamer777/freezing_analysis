from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


COORD_COLUMNS = [
    "head_x",
    "head_y",
    "head_z",
    "body_x",
    "body_y",
    "body_z",
    "tail_x",
    "tail_y",
    "tail_z",
]


def preprocess_tracking_data(
    df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    processed = df.copy()
    processed = add_timestamps(processed, config)
    processed = apply_scaling(processed, config)
    processed = filter_low_quality_rows(processed, config)
    processed = smooth_coordinates(processed, config)
    processed = processed.reset_index(drop=True)
    maybe_save_preprocessed_tracks(processed, config, output_dir)
    return processed


def add_timestamps(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    frame_rate = config.get("preprocess", {}).get("frame_rate", 30.0)
    if frame_rate <= 0:
        raise ValueError("frame_rate 必须大于 0")

    result = df.copy()
    frame_zero = result["frame"].min()
    result["timestamp"] = (result["frame"] - frame_zero) / frame_rate
    return result


def apply_scaling(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    preprocess_config = config.get("preprocess", {})
    if not preprocess_config.get("apply_scaling", False):
        return df.copy()

    scale = preprocess_config.get("scale_mm_per_unit", 1.0)
    result = df.copy()
    for col in COORD_COLUMNS:
        if col in result.columns:
            result[col] = result[col] * scale
    return result


def filter_low_quality_rows(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    preprocess_config = config.get("preprocess", {})
    result = df.copy()

    if preprocess_config.get("drop_low_confidence", False) and "confidence" in result.columns:
        min_confidence = preprocess_config.get("min_confidence", 0.0)
        result = result[result["confidence"] >= min_confidence]

    if preprocess_config.get("drop_low_quality", False) and "track_quality" in result.columns:
        min_track_quality = preprocess_config.get("min_track_quality", 0.0)
        result = result[result["track_quality"] >= min_track_quality]

    return result.reset_index(drop=True)


def smooth_coordinates(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    preprocess_config = config.get("preprocess", {})
    if not preprocess_config.get("smoothing_enabled", False):
        return df.copy()

    window = int(preprocess_config.get("smoothing_window", 5))
    if window <= 1:
        return df.copy()

    result = df.copy()
    for _, group in result.groupby("fish_id", sort=False):
        smoothed = group[COORD_COLUMNS].rolling(window=window, min_periods=1, center=True).mean()
        result.loc[group.index, COORD_COLUMNS] = smoothed.values
    return result


def maybe_save_preprocessed_tracks(
    df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> Path | None:
    output_config = config.get("output", {})
    if not output_config.get("save_preprocessed_tracks", False):
        return None

    base_dir = Path(output_dir) if output_dir is not None else Path(config.get("paths", {}).get("output_dir", "outputs"))
    save_dir = base_dir / "intermediate" / "preprocess"
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = output_config.get("preprocessed_track_file", "preprocessed_tracks.csv")
    output_path = save_dir / filename
    df.to_csv(output_path, index=False)
    return output_path
