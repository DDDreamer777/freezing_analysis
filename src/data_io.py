from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


STANDARD_TRACK_COLUMNS = {
    "frame": "frame",
    "fish_id": "fish_id",
    "head_x": "head_x",
    "head_y": "head_y",
    "head_z": "head_z",
    "body_x": "body_x",
    "body_y": "body_y",
    "body_z": "body_z",
    "tail_x": "tail_x",
    "tail_y": "tail_y",
    "tail_z": "tail_z",
    "confidence": "confidence",
    "triangulation_error": "triangulation_error",
    "track_quality": "track_quality",
    "interpolated": "interpolated",
    "outlier_corrected": "outlier_corrected",
}

STANDARD_GT_COLUMNS = {
    "video_id": "video_id",
    "fish_id": "fish_id",
    "start_second": "start_second",
    "end_second": "end_second",
    "label": "label",
    "confidence": "confidence",
}


class ConfigError(ValueError):
    pass


class DataFormatError(ValueError):
    pass


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"配置文件格式错误: {path}")
    return config


def resolve_paths(config: dict[str, Any], project_root: str | Path | None = None) -> dict[str, Path]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    paths_config = config.get("paths", {})
    resolved: dict[str, Path] = {}
    for key, value in paths_config.items():
        resolved[key] = root / value
    return resolved


def load_tracking_data(track_file: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(track_file)
    _validate_tracking_columns(df, config)
    return df


def standardize_tracking_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_config = config.get("data", {})
    rename_map = {
        data_config["frame_col"]: "frame",
        data_config["id_col"]: "fish_id",
        data_config["head_x_col"]: "head_x",
        data_config["head_y_col"]: "head_y",
        data_config["head_z_col"]: "head_z",
        data_config["body_x_col"]: "body_x",
        data_config["body_y_col"]: "body_y",
        data_config["body_z_col"]: "body_z",
        data_config["tail_x_col"]: "tail_x",
        data_config["tail_y_col"]: "tail_y",
        data_config["tail_z_col"]: "tail_z",
    }

    optional_pairs = {
        data_config.get("confidence_col"): "confidence",
        data_config.get("triangulation_error_col"): "triangulation_error",
        data_config.get("track_quality_col"): "track_quality",
        data_config.get("interpolated_col"): "interpolated",
        data_config.get("outlier_corrected_col"): "outlier_corrected",
    }
    rename_map.update({src: dst for src, dst in optional_pairs.items() if src in df.columns})

    standardized = df.rename(columns=rename_map).copy()
    standardized = sort_tracking_data(standardized)

    standardized["frame"] = pd.to_numeric(standardized["frame"], errors="raise").astype(int)
    standardized["fish_id"] = standardized["fish_id"].astype(str)

    numeric_cols = [
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
    for col in numeric_cols:
        standardized[col] = pd.to_numeric(standardized[col], errors="coerce")

    for col in ["confidence", "triangulation_error", "track_quality"]:
        if col in standardized.columns:
            standardized[col] = pd.to_numeric(standardized[col], errors="coerce")

    for col in ["interpolated", "outlier_corrected"]:
        if col in standardized.columns:
            standardized[col] = standardized[col].astype("boolean")

    ordered_cols = [col for col in STANDARD_TRACK_COLUMNS if col in standardized.columns]
    remaining_cols = [col for col in standardized.columns if col not in ordered_cols]
    return standardized[ordered_cols + remaining_cols]


def sort_tracking_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["fish_id", "frame"]).reset_index(drop=True)


def load_gt_data(gt_file: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    gt_config = config.get("gt", {})
    file_type = gt_config.get("file_type", "xlsx")
    if file_type != "xlsx":
        raise DataFormatError(f"暂不支持的GT文件类型: {file_type}")

    sheet_name = gt_config.get("sheet_name", 0)
    df = pd.read_excel(gt_file, sheet_name=sheet_name)
    _validate_gt_columns(df, config)
    return df


def standardize_gt_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    gt_config = config.get("gt", {})
    rename_map = {
        gt_config["video_id_col"]: "video_id",
        gt_config["fish_id_col"]: "fish_id",
        gt_config["start_second_col"]: "start_second",
        gt_config["end_second_col"]: "end_second",
        gt_config["label_col"]: "label",
        gt_config["confidence_col"]: "confidence",
    }

    standardized = df.rename(columns=rename_map).copy()
    standardized["video_id"] = standardized["video_id"].astype(str)
    standardized["fish_id"] = standardized["fish_id"].astype(str)
    standardized["start_second"] = pd.to_numeric(standardized["start_second"], errors="raise")
    standardized["end_second"] = pd.to_numeric(standardized["end_second"], errors="raise")
    standardized["label"] = standardized["label"].astype(str)
    standardized["confidence"] = pd.to_numeric(standardized["confidence"], errors="raise").astype(int)

    ordered_cols = list(STANDARD_GT_COLUMNS.keys())
    remaining_cols = [col for col in standardized.columns if col not in ordered_cols]
    return standardized[ordered_cols + remaining_cols]


def _validate_tracking_columns(df: pd.DataFrame, config: dict[str, Any]) -> None:
    data_config = config.get("data", {})
    required_keys = [
        "frame_col",
        "id_col",
        "head_x_col",
        "head_y_col",
        "head_z_col",
        "body_x_col",
        "body_y_col",
        "body_z_col",
        "tail_x_col",
        "tail_y_col",
        "tail_z_col",
    ]
    missing_keys = [key for key in required_keys if key not in data_config]
    if missing_keys:
        raise ConfigError(f"data 配置缺少字段: {missing_keys}")

    missing_columns = [data_config[key] for key in required_keys if data_config[key] not in df.columns]
    if missing_columns:
        raise DataFormatError(f"轨迹数据缺少必要列: {missing_columns}")


def _validate_gt_columns(df: pd.DataFrame, config: dict[str, Any]) -> None:
    gt_config = config.get("gt", {})
    required_keys = [
        "video_id_col",
        "fish_id_col",
        "start_second_col",
        "end_second_col",
        "label_col",
        "confidence_col",
    ]
    missing_keys = [key for key in required_keys if key not in gt_config]
    if missing_keys:
        raise ConfigError(f"gt 配置缺少字段: {missing_keys}")

    missing_columns = [gt_config[key] for key in required_keys if gt_config[key] not in df.columns]
    if missing_columns:
        raise DataFormatError(f"GT数据缺少必要列: {missing_columns}")
