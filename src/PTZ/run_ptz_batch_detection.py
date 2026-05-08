from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from classification import classify_behavior_candidates
from data_io import load_config, load_tracking_data, standardize_tracking_columns
from detection import detect_pause_candidates
from features import compute_all_features
from preprocess import preprocess_tracking_data
from segmentation import segment_behavior
from PTZ.ptz_stagnation_accuracy import evaluate_ptz_stagnation


PTZ_TRACK_PATTERN = re.compile(r"ptz_(\d+)_tracks_3d_interpolated\.csv")


def run_ptz_batch_detection(
    config_path: str | Path | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    evaluate: bool = True,
    iou_threshold: float = 0.1,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    config_file = _resolve_path(config_path, root) if config_path is not None else root / "configs" / "default.yaml"
    track_dir = _resolve_path(input_dir, root) if input_dir is not None else root / "data" / "raw" / "PTZ"
    batch_output_dir = _resolve_path(output_dir, root) if output_dir is not None else root / "outputs" / "ptz_batch_classification"

    config = load_config(config_file)
    track_files = discover_ptz_track_files(track_dir)
    if not track_files:
        raise FileNotFoundError(f"No per-fish PTZ track files found in {track_dir}")

    per_fish_dir = batch_output_dir / "per_fish"
    run_records: list[dict[str, Any]] = []
    for track_file in track_files:
        fish_id = infer_fish_id_from_track_file(track_file)
        fish_output_dir = per_fish_dir / f"ptz_{fish_id}"
        record = run_single_ptz_file(track_file, config, fish_output_dir)
        run_records.append(record)
        print(_format_run_record(record), flush=True)

    merged_dir = batch_output_dir / "merged"
    merged_behavior_path = merged_dir / "final" / "behavior_events.csv"
    merged_feature_path = merged_dir / "intermediate" / "features" / "feature_table.csv"
    merged_behavior_df = merge_behavior_event_files(run_records, merged_behavior_path)
    merged_feature_df = merge_feature_table_files(run_records, merged_feature_path)
    summary_path = save_batch_run_summary(run_records, merged_dir / "batch_run_summary.csv")

    evaluation_result = None
    if evaluate:
        evaluation_result = evaluate_ptz_stagnation(
            behavior_events_path=merged_behavior_path,
            feature_table_path=merged_feature_path,
            output_dir=batch_output_dir / "ptz_stagnation_accuracy",
            iou_threshold=iou_threshold,
        )

    return {
        "run_records": run_records,
        "merged_behavior_df": merged_behavior_df,
        "merged_feature_df": merged_feature_df,
        "summary_path": summary_path,
        "evaluation_result": evaluation_result,
        "output_dir": batch_output_dir,
    }


def run_single_ptz_file(
    track_file: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    track_path = Path(track_file)
    fish_id = infer_fish_id_from_track_file(track_path)
    fish_output_dir = Path(output_dir)
    fish_config = copy.deepcopy(config)

    tracks_df = standardize_tracking_columns(load_tracking_data(track_path, fish_config), fish_config)
    processed_df = preprocess_tracking_data(tracks_df, fish_config, output_dir=fish_output_dir)
    feature_df = compute_all_features(processed_df, fish_config, output_dir=fish_output_dir)
    segment_df = segment_behavior(feature_df, fish_config, output_dir=fish_output_dir)
    scored_df, state_df, pause_event_df = detect_pause_candidates(segment_df, fish_config, output_dir=fish_output_dir)
    stagnation_df, twist_df, glide_df, final_behavior_df = classify_behavior_candidates(
        pause_event_df,
        state_df,
        scored_df,
        fish_config,
        output_dir=fish_output_dir,
    )

    return {
        "fish_id": fish_id,
        "track_file": track_path,
        "run_dir": fish_output_dir,
        "behavior_events_path": fish_output_dir / "final" / "behavior_events.csv",
        "feature_table_path": fish_output_dir / "intermediate" / "features" / "feature_table.csv",
        "n_tracks": int(len(tracks_df)),
        "n_segments": int(len(segment_df)),
        "n_pause_events": int(len(pause_event_df)),
        "n_stagnation_events": int(len(stagnation_df)),
        "n_twist_events": int(len(twist_df)),
        "n_glide_events": int(len(glide_df)),
        "n_final_behaviors": int(len(final_behavior_df)),
        "status": "ok",
    }


def discover_ptz_track_files(input_dir: str | Path) -> list[Path]:
    directory = Path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"PTZ input directory does not exist: {directory}")
    return sorted(
        (path for path in directory.iterdir() if PTZ_TRACK_PATTERN.fullmatch(path.name)),
        key=lambda path: int(PTZ_TRACK_PATTERN.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
    )


def infer_fish_id_from_track_file(track_file: str | Path) -> str:
    path = Path(track_file)
    match = PTZ_TRACK_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"Cannot infer PTZ fish id from file name: {path.name}")
    return match.group(1)


def merge_behavior_event_files(run_records: list[dict[str, Any]], output_path: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in run_records:
        path = Path(record["behavior_events_path"])
        if not path.exists():
            continue
        event_df = pd.read_csv(path, low_memory=False)
        if event_df.empty:
            continue
        event_df = event_df.copy()
        if "behavior_id" in event_df.columns:
            event_df["source_behavior_id"] = event_df["behavior_id"]
        else:
            event_df["source_behavior_id"] = pd.NA
        event_df["source_fish_id"] = str(record["fish_id"])
        event_df["source_run_dir"] = str(record["run_dir"])
        frames.append(event_df)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged["fish_id"] = merged["fish_id"].astype(str)
        merged = _sort_by_fish_and_frame(merged)
        merged["behavior_id"] = range(1, len(merged) + 1)
    else:
        merged = pd.DataFrame(columns=["behavior_id", "source_behavior_id", "source_fish_id", "source_run_dir"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return merged


def merge_feature_table_files(run_records: list[dict[str, Any]], output_path: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in run_records:
        path = Path(record["feature_table_path"])
        if not path.exists():
            continue
        feature_df = pd.read_csv(path, low_memory=False)
        if feature_df.empty:
            continue
        feature_df = feature_df.copy()
        feature_df["source_run_dir"] = str(record["run_dir"])
        frames.append(feature_df)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged["fish_id"] = merged["fish_id"].astype(str)
        merged = _sort_by_fish_and_frame(merged)
    else:
        merged = pd.DataFrame()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return merged


def save_batch_run_summary(run_records: list[dict[str, Any]], output_path: str | Path) -> Path:
    summary_df = pd.DataFrame([{key: str(value) if isinstance(value, Path) else value for key, value in record.items()} for record in run_records])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output, index=False)
    return output


def _sort_by_fish_and_frame(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.copy()
    sorted_df["_fish_sort"] = pd.to_numeric(sorted_df["fish_id"], errors="coerce")
    if "start_frame" in sorted_df.columns:
        sorted_df["_frame_sort"] = pd.to_numeric(sorted_df["start_frame"], errors="coerce")
    elif "frame" in sorted_df.columns:
        sorted_df["_frame_sort"] = pd.to_numeric(sorted_df["frame"], errors="coerce")
    else:
        sorted_df["_frame_sort"] = range(len(sorted_df))
    sorted_df = sorted_df.sort_values(["_fish_sort", "fish_id", "_frame_sort"], kind="mergesort", na_position="last")
    return sorted_df.drop(columns=["_fish_sort", "_frame_sort"]).reset_index(drop=True)


def _resolve_path(path: str | Path | None, root: Path) -> Path:
    if path is None:
        return root
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def _format_run_record(record: dict[str, Any]) -> str:
    return (
        f"fish {record['fish_id']}: tracks={record['n_tracks']}, segments={record['n_segments']}, "
        f"stagnation={record['n_stagnation_events']}, twist={record['n_twist_events']}, "
        f"glide={record['n_glide_events']}, final={record['n_final_behaviors']}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PTZ detection per fish, merge outputs, and evaluate stagnation accuracy.")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to config YAML file.")
    parser.add_argument("--input-dir", default=None, help="Directory containing per-fish PTZ pipeline track files.")
    parser.add_argument("--output-dir", default=None, help="Directory for PTZ batch outputs.")
    parser.add_argument("--project-root", default=None, help="Project root directory.")
    parser.add_argument("--iou-threshold", type=float, default=0.1, help="IoU threshold for PTZ stagnation event matching.")
    parser.add_argument("--no-evaluate", action="store_true", help="Run detection/classification only, without PTZ accuracy scoring.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_ptz_batch_detection(
        config_path=args.config_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        project_root=args.project_root,
        evaluate=not args.no_evaluate,
        iou_threshold=args.iou_threshold,
    )
    print(f"\noutput_dir {result['output_dir']}")
    print(f"summary_file {result['summary_path']}")
    print(f"merged_behaviors {len(result['merged_behavior_df'])}")
    print(f"merged_feature_rows {len(result['merged_feature_df'])}")
    if result["evaluation_result"] is not None:
        print(f"accuracy_dir {result['output_dir'] / 'ptz_stagnation_accuracy'}")


if __name__ == "__main__":
    main()
