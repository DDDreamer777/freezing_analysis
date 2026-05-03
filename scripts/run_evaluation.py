from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd

from data_io import load_config, load_gt_data, resolve_paths, standardize_gt_columns
from evaluation import evaluate_behavior_events


def run_evaluation(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    pred_event_file: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    config_file = Path(config_path) if config_path is not None else root / "configs" / "default.yaml"
    config = load_config(config_file)
    paths = resolve_paths(config, root)

    gt_path = paths["gt_file"]
    gt_df = standardize_gt_columns(load_gt_data(gt_path, config), config)

    resolved_output_dir = Path(output_dir) if output_dir is not None else paths.get("output_dir", root / "outputs")
    pred_path = Path(pred_event_file) if pred_event_file is not None else resolved_output_dir / "final" / "behavior_events.csv"
    pred_event_df = pd.read_csv(pred_path)

    matched_df, summary_df = evaluate_behavior_events(
        pred_event_df,
        gt_df,
        config,
        output_dir=resolved_output_dir,
    )
    return matched_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run event-level evaluation for behavior events.")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to config YAML file.")
    parser.add_argument("--project-root", dest="project_root", default=None, help="Project root directory.")
    parser.add_argument("--pred-file", dest="pred_event_file", default=None, help="Path to prediction event CSV file.")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="Directory for evaluation outputs.")
    args = parser.parse_args()

    _, summary_df = run_evaluation(
        config_path=args.config_path,
        project_root=args.project_root,
        pred_event_file=args.pred_event_file,
        output_dir=args.output_dir,
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
