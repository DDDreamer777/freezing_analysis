from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd

from classification import classify_behavior_candidates
from data_io import load_config, load_tracking_data, resolve_paths, standardize_tracking_columns
from detection import detect_pause_candidates
from features import compute_all_features
from preprocess import preprocess_tracking_data
from segmentation import segment_behavior


def run_detection(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    config_file = Path(config_path) if config_path is not None else root / "configs" / "default.yaml"
    config = load_config(config_file)
    paths = resolve_paths(config, root)

    resolved_output_dir = Path(output_dir) if output_dir is not None else paths.get("output_dir", root / "outputs")

    tracks_df = standardize_tracking_columns(load_tracking_data(paths["track_file"], config), config)
    processed_df = preprocess_tracking_data(tracks_df, config, output_dir=resolved_output_dir)
    feature_df = compute_all_features(processed_df, config, output_dir=resolved_output_dir)
    segment_df = segment_behavior(feature_df, config, output_dir=resolved_output_dir)
    scored_df, state_df, pause_event_df = detect_pause_candidates(segment_df, config, output_dir=resolved_output_dir)
    stagnation_df, twist_df, glide_df, final_behavior_df = classify_behavior_candidates(
        pause_event_df,
        state_df,
        scored_df,
        config,
        output_dir=resolved_output_dir,
    )

    return {
        "tracks_df": tracks_df,
        "processed_df": processed_df,
        "feature_df": feature_df,
        "segment_df": segment_df,
        "scored_df": scored_df,
        "state_df": state_df,
        "pause_event_df": pause_event_df,
        "stagnation_df": stagnation_df,
        "twist_df": twist_df,
        "glide_df": glide_df,
        "final_behavior_df": final_behavior_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full behavior detection pipeline.")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to config YAML file.")
    parser.add_argument("--project-root", dest="project_root", default=None, help="Project root directory.")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="Directory for pipeline outputs.")
    args = parser.parse_args()

    results = run_detection(
        config_path=args.config_path,
        project_root=args.project_root,
        output_dir=args.output_dir,
    )
    summary = {
        "tracks": len(results["tracks_df"]),
        "segments": len(results["segment_df"]),
        "pause_events": len(results["pause_event_df"]),
        "stagnation_events": len(results["stagnation_df"]),
        "twist_events": len(results["twist_df"]),
        "glide_events": len(results["glide_df"]),
        "final_behaviors": len(results["final_behavior_df"]),
    }
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
