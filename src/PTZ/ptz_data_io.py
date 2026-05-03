from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PTZ_REQUIRED_COLUMNS = [
    "bodyparts_coords",
    "Head_x",
    "Head_y",
    "Trunk_x",
    "Trunk_y",
    "Tail_x",
    "Tail_y",
    "Z_y",
]

PTZ_OUTPUT_COLUMNS = [
    "frame",
    "fish_id",
    "head_x",
    "head_y",
    "head_z",
    "body_x",
    "body_y",
    "body_z",
    "tail_x",
    "tail_y",
    "tail_z",
    "confidence",
    "triangulation_error",
    "track_quality",
    "interpolated",
    "outlier_corrected",
    "is_virtual_original",
    "match_type",
    "interpolation_type",
    "outlier_score",
    "behavior_label",
]


class PTZDataFormatError(ValueError):
    pass


def load_ptz_tracking_files(
    ptz_files: list[str | Path] | tuple[str | Path, ...],
    scale_mm_per_pixel: float = 0.30,
    nrows: int | None = None,
    z_source_col: str = "Z_y",
) -> pd.DataFrame:
    converted = [
        load_ptz_tracking_file(
            ptz_file,
            scale_mm_per_pixel=scale_mm_per_pixel,
            nrows=nrows,
            z_source_col=z_source_col,
        )
        for ptz_file in ptz_files
    ]
    if not converted:
        return pd.DataFrame(columns=PTZ_OUTPUT_COLUMNS)
    return pd.concat(converted, ignore_index=True).sort_values(["fish_id", "frame"]).reset_index(drop=True)


def load_ptz_tracking_file(
    ptz_file: str | Path,
    scale_mm_per_pixel: float = 0.30,
    fish_id: str | int | None = None,
    nrows: int | None = None,
    z_source_col: str = "Z_y",
) -> pd.DataFrame:
    path = Path(ptz_file)
    df = pd.read_csv(path, nrows=nrows)
    _validate_ptz_columns(df, z_source_col)

    fish_id_value = str(fish_id) if fish_id is not None else _infer_fish_id_from_path(path)
    z_mm = pd.to_numeric(df[z_source_col], errors="coerce") * scale_mm_per_pixel

    converted = pd.DataFrame(
        {
            "frame": pd.to_numeric(df["bodyparts_coords"], errors="raise").astype(int),
            "fish_id": fish_id_value,
            "head_x": pd.to_numeric(df["Head_x"], errors="coerce") * scale_mm_per_pixel,
            "head_y": pd.to_numeric(df["Head_y"], errors="coerce") * scale_mm_per_pixel,
            "head_z": z_mm,
            "body_x": pd.to_numeric(df["Trunk_x"], errors="coerce") * scale_mm_per_pixel,
            "body_y": pd.to_numeric(df["Trunk_y"], errors="coerce") * scale_mm_per_pixel,
            "body_z": z_mm,
            "tail_x": pd.to_numeric(df["Tail_x"], errors="coerce") * scale_mm_per_pixel,
            "tail_y": pd.to_numeric(df["Tail_y"], errors="coerce") * scale_mm_per_pixel,
            "tail_z": z_mm,
            "confidence": 1.0,
            "triangulation_error": 0.0,
            "track_quality": 1.0,
            "interpolated": False,
            "outlier_corrected": False,
            "is_virtual_original": False,
            "match_type": "ptz_csv_data",
            "interpolation_type": "original",
            "outlier_score": 0.0,
        }
    )

    if "behavior.label" in df.columns:
        converted["behavior_label"] = df["behavior.label"]
    else:
        converted["behavior_label"] = pd.NA

    return converted[PTZ_OUTPUT_COLUMNS].sort_values(["fish_id", "frame"]).reset_index(drop=True)


def _validate_ptz_columns(df: pd.DataFrame, z_source_col: str) -> None:
    required_columns = list(dict.fromkeys([*PTZ_REQUIRED_COLUMNS, z_source_col]))
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise PTZDataFormatError(f"PTZ tracking data is missing required columns: {missing_columns}")


def _infer_fish_id_from_path(path: Path) -> str:
    match = re.search(r"PTZ[_-](\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return path.stem
