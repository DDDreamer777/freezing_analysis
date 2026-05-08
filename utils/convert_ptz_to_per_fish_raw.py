from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from PTZ.ptz_data_io import load_ptz_tracking_file


PIPELINE_COLUMNS = [
    "frame",
    "id",
    "3d_x",
    "3d_y",
    "3d_z",
    "confidence",
    "triangulation_error",
    "track_quality",
    "is_virtual_original",
    "match_type",
    "interpolated",
    "interpolation_type",
    "outlier_corrected",
    "outlier_score",
    "body_x",
    "body_y",
    "body_z",
    "tail_x",
    "tail_y",
    "tail_z",
    "behavior_label",
]


def convert_ptz_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    scale_mm_per_pixel: float = 0.30,
    merged_output: str | Path | None = None,
    write_merged: bool = True,
) -> pd.DataFrame:
    source_dir = Path(input_dir)
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted((path for path in source_dir.iterdir() if _is_ptz_source_file(path)), key=_ptz_sort_key)
    if not source_files:
        raise FileNotFoundError(f"No PTZ_*.csv files found in {source_dir}")

    summary_rows: list[dict[str, Any]] = []
    pipeline_frames: list[pd.DataFrame] = []
    for source_file in source_files:
        pipeline_df = _load_ptz_file_as_pipeline_df(source_file, scale_mm_per_pixel=scale_mm_per_pixel)
        converted = _write_ptz_pipeline_file(source_file, pipeline_df, save_dir)
        summary_rows.append(converted)
        pipeline_frames.append(pipeline_df)

    if write_merged:
        merged_path = Path(merged_output) if merged_output is not None else save_dir.parent / "ptz_tracks_3d_interpolated.csv"
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df = _sort_pipeline_tracks(pd.concat(pipeline_frames, ignore_index=True))
        merged_df.to_csv(merged_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = save_dir / "ptz_conversion_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def convert_ptz_file(
    source_file: str | Path,
    output_dir: str | Path,
    scale_mm_per_pixel: float = 0.30,
) -> dict[str, Any]:
    source_path = Path(source_file)
    save_dir = Path(output_dir)
    pipeline_df = _load_ptz_file_as_pipeline_df(source_path, scale_mm_per_pixel=scale_mm_per_pixel)
    return _write_ptz_pipeline_file(source_path, pipeline_df, save_dir)


def _load_ptz_file_as_pipeline_df(
    source_file: str | Path,
    scale_mm_per_pixel: float = 0.30,
) -> pd.DataFrame:
    tracks = load_ptz_tracking_file(source_file, scale_mm_per_pixel=scale_mm_per_pixel)
    return _to_pipeline_schema(tracks)


def _write_ptz_pipeline_file(
    source_file: str | Path,
    pipeline_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_path = Path(source_file)
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pipeline_df = _sort_pipeline_tracks(pipeline_df)
    fish_id = str(pipeline_df["id"].iloc[0]) if not pipeline_df.empty else _infer_fish_id_from_name(source_path)
    output_path = save_dir / f"ptz_{fish_id}_tracks_3d_interpolated.csv"
    pipeline_df.to_csv(output_path, index=False)

    label_counts = _format_label_counts(pipeline_df["behavior_label"]) if "behavior_label" in pipeline_df.columns else ""
    return {
        "source_file": str(source_path),
        "output_file": str(output_path),
        "fish_id": fish_id,
        "n_rows": int(len(pipeline_df)),
        "n_behavior_labels": int(pipeline_df["behavior_label"].notna().sum()),
        "label_counts": label_counts,
        "status": "ok",
    }


def _to_pipeline_schema(tracks: pd.DataFrame) -> pd.DataFrame:
    pipeline_df = pd.DataFrame(
        {
            "frame": tracks["frame"],
            "id": tracks["fish_id"],
            "3d_x": tracks["head_x"],
            "3d_y": tracks["head_y"],
            "3d_z": tracks["head_z"],
            "confidence": tracks["confidence"],
            "triangulation_error": tracks["triangulation_error"],
            "track_quality": tracks["track_quality"],
            "is_virtual_original": tracks["is_virtual_original"],
            "match_type": tracks["match_type"],
            "interpolated": tracks["interpolated"],
            "interpolation_type": tracks["interpolation_type"],
            "outlier_corrected": tracks["outlier_corrected"],
            "outlier_score": tracks["outlier_score"],
            "body_x": tracks["body_x"],
            "body_y": tracks["body_y"],
            "body_z": tracks["body_z"],
            "tail_x": tracks["tail_x"],
            "tail_y": tracks["tail_y"],
            "tail_z": tracks["tail_z"],
            "behavior_label": tracks["behavior_label"],
        }
    )
    return pipeline_df[PIPELINE_COLUMNS]


def _sort_pipeline_tracks(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.copy()
    fish_sort = pd.to_numeric(sorted_df["id"], errors="coerce")
    sorted_df["_fish_sort"] = fish_sort
    sorted_df["_frame_sort"] = pd.to_numeric(sorted_df["frame"], errors="coerce")
    sorted_df = sorted_df.sort_values(
        ["_fish_sort", "id", "_frame_sort", "frame"],
        kind="mergesort",
        na_position="last",
    )
    return sorted_df.drop(columns=["_fish_sort", "_frame_sort"]).reset_index(drop=True)


def _format_label_counts(labels: pd.Series) -> str:
    cleaned = labels.fillna("").astype(str).str.strip()
    cleaned = cleaned.replace({"NA": "", "NaN": "", "nan": "", "None": "", "<NA>": ""})
    display = cleaned.replace("", "unlabeled")
    return "; ".join(f"{label}:{int(count)}" for label, count in display.value_counts(sort=False).items())


def _ptz_sort_key(path: Path) -> tuple[int, str]:
    fish_id = _infer_fish_id_from_name(path)
    try:
        return int(fish_id), path.name
    except ValueError:
        return 10**9, path.name


def _is_ptz_source_file(path: Path) -> bool:
    return path.is_file() and re.fullmatch(r"PTZ_\d+\.csv", path.name) is not None


def _infer_fish_id_from_name(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", maxsplit=1)[1]
    return stem


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PTZ_*.csv files into per-fish and merged pipeline raw track files.")
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data" / "PTZ"), help="Directory containing PTZ_*.csv files.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "raw" / "PTZ"), help="Directory for per-fish converted files.")
    parser.add_argument(
        "--merged-output",
        default=None,
        help="Path for the merged PTZ experiment track file. Defaults to <output-dir>/../ptz_tracks_3d_interpolated.csv.",
    )
    parser.add_argument("--no-merged-output", action="store_true", help="Only write per-fish files and the conversion summary.")
    parser.add_argument("--scale-mm-per-pixel", type=float, default=0.30, help="Coordinate scale used for PTZ pixels.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary_df = convert_ptz_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scale_mm_per_pixel=args.scale_mm_per_pixel,
        merged_output=args.merged_output,
        write_merged=not args.no_merged_output,
    )
    print(summary_df[["fish_id", "n_rows", "output_file", "status"]].to_string(index=False))
    if not args.no_merged_output:
        merged_path = Path(args.merged_output) if args.merged_output is not None else Path(args.output_dir).parent / "ptz_tracks_3d_interpolated.csv"
        print(f"\nmerged_output_file {merged_path}")


if __name__ == "__main__":
    main()
