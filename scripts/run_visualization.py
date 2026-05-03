from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd

from data_io import load_config, resolve_paths
from visualization import generate_visualizations


def run_visualization(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    feature_file: str | Path | None = None,
    segment_file: str | Path | None = None,
    state_file: str | Path | None = None,
    behavior_file: str | Path | None = None,
) -> dict[str, list[Path]]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    config_file = Path(config_path) if config_path is not None else root / "configs" / "default.yaml"
    config = load_config(config_file)
    paths = resolve_paths(config, root)

    resolved_output_dir = Path(output_dir) if output_dir is not None else paths.get("output_dir", root / "outputs")
    feature_path = Path(feature_file) if feature_file is not None else resolved_output_dir / "intermediate" / "features" / "feature_table.csv"
    segment_path = Path(segment_file) if segment_file is not None else resolved_output_dir / "intermediate" / "segmentation" / "candidate_segments.csv"
    state_path = Path(state_file) if state_file is not None else resolved_output_dir / "intermediate" / "detection" / "candidate_segment_states.csv"
    behavior_path = Path(behavior_file) if behavior_file is not None else resolved_output_dir / "final" / "behavior_events.csv"

    feature_df = pd.read_csv(feature_path)
    segment_df = pd.read_csv(segment_path)
    state_df = pd.read_csv(state_path)
    behavior_df = pd.read_csv(behavior_path)

    return generate_visualizations(
        feature_df,
        segment_df,
        state_df,
        behavior_df,
        config,
        output_dir=resolved_output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate behavior visualization artifacts from existing outputs.")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to config YAML file.")
    parser.add_argument("--project-root", dest="project_root", default=None, help="Project root directory.")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="Directory containing pipeline outputs.")
    parser.add_argument("--feature-file", dest="feature_file", default=None, help="Path to feature table CSV.")
    parser.add_argument("--segment-file", dest="segment_file", default=None, help="Path to candidate segments CSV.")
    parser.add_argument("--state-file", dest="state_file", default=None, help="Path to candidate segment states CSV.")
    parser.add_argument("--behavior-file", dest="behavior_file", default=None, help="Path to final behavior events CSV.")
    args = parser.parse_args()

    outputs = run_visualization(
        config_path=args.config_path,
        project_root=args.project_root,
        output_dir=args.output_dir,
        feature_file=args.feature_file,
        segment_file=args.segment_file,
        state_file=args.state_file,
        behavior_file=args.behavior_file,
    )
    summary = {key: len(value) for key, value in outputs.items()}
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
