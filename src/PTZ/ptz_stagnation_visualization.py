from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRED_LABEL = "stagnation"
GT_LABEL = "IM"
FEATURE_COLUMNS = ["v", "d_win", "E_move", "S_pose"]
POINTS = [
    ("head", "head_x", "head_y", "head_z"),
    ("body", "body_x", "body_y", "body_z"),
    ("tail", "tail_x", "tail_y", "tail_z"),
]


def visualize_ptz_stagnation_events(
    feature_table_path: str | Path,
    behavior_events_path: str | Path,
    output_dir: str | Path,
    event_matches_path: str | Path | None = None,
) -> dict[str, Any]:
    feature_df = pd.read_csv(feature_table_path, low_memory=False)
    event_df = pd.read_csv(behavior_events_path, low_memory=False)
    match_df = _read_optional_csv(event_matches_path)

    output_path = Path(output_dir)
    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    stagnation_events = _normalize_stagnation_events(event_df)
    summary_df = build_visualization_summary(feature_df, stagnation_events, match_df)
    summary_path = output_path / "visualization_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    figure_paths: list[Path] = []
    for event in stagnation_events.itertuples(index=False):
        match_row = _match_row_for_event(match_df, int(event.event_id))
        figure_paths.append(
            plot_stagnation_event(
                event=event,
                feature_df=feature_df,
                match_row=match_row,
                output_dir=figure_dir,
            )
        )

    return {
        "summary": summary_df,
        "summary_path": summary_path,
        "figure_paths": figure_paths,
    }


def build_visualization_summary(
    feature_df: pd.DataFrame,
    event_df: pd.DataFrame,
    match_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    normalized_features = _normalize_features(feature_df)
    normalized_events = _normalize_stagnation_events(event_df)
    matched_ids = _matched_pred_event_ids(match_df)

    rows: list[dict[str, Any]] = []
    for event in normalized_events.itertuples(index=False):
        window = _event_window(normalized_features, event)
        label_counts = _label_counts(window)
        im_frames = int((window["behavior_label_clean"] == GT_LABEL).sum()) if not window.empty else 0
        total_frames = int(len(window))
        row = {
            "event_id": int(event.event_id),
            "fish_id": str(event.fish_id),
            "start_frame": int(event.start_frame),
            "end_frame": int(event.end_frame),
            "start_time": float(event.start_time),
            "end_time": float(event.end_time),
            "duration": float(event.duration),
            "matched_im": int(event.event_id) in matched_ids,
            "im_frames": im_frames,
            "im_ratio": im_frames / total_frames if total_frames else 0.0,
            "total_frames": total_frames,
            "label_counts": _format_label_counts(label_counts),
        }
        row.update(_feature_summary(window))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_stagnation_event(
    event: Any,
    feature_df: pd.DataFrame,
    match_row: pd.Series | None,
    output_dir: str | Path,
) -> Path:
    normalized_features = _normalize_features(feature_df)
    window = _event_window(normalized_features, event)
    if window.empty:
        raise ValueError(f"No feature rows found for event {int(event.event_id)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    png_path = output_path / f"stagnation_event_{int(event.event_id):03d}_fish_{str(event.fish_id)}.png"

    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 1.0])
    ax_3d = fig.add_subplot(grid[:2, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[:2, 1])
    ax_text = fig.add_subplot(grid[:2, 2])
    metric_axes = [fig.add_subplot(grid[2, index]) for index in range(3)]
    ax_pose = metric_axes[2].twinx()

    _plot_3d_tracks(ax_3d, window)
    _plot_xy_tracks(ax_xy, window)
    _plot_metrics(metric_axes[0], window, ["v"], "Velocity")
    _plot_metrics(metric_axes[1], window, ["d_win", "E_move"], "Window Motion")
    _plot_metrics(metric_axes[2], window, ["S_pose"], "Pose Stability")
    _plot_label_regions(metric_axes[0], window)
    _plot_label_regions(metric_axes[1], window)
    _plot_label_regions(metric_axes[2], window)
    _plot_label_regions(ax_pose, window)
    ax_pose.set_axis_off()
    _plot_event_text(ax_text, event, window, match_row)

    title = (
        f"PTZ stagnation event {int(event.event_id)} | fish {str(event.fish_id)} | "
        f"frames {int(event.start_frame)}-{int(event.end_frame)}"
    )
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    return png_path


def _normalize_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    required = ["fish_id", "frame", "timestamp", "behavior_label", *FEATURE_COLUMNS]
    for _, x_col, y_col, z_col in POINTS:
        required.extend([x_col, y_col, z_col])
    _require_columns(feature_df, required)

    normalized = feature_df.copy()
    normalized["fish_id"] = normalized["fish_id"].astype(str)
    normalized["frame"] = pd.to_numeric(normalized["frame"], errors="raise").astype(int)
    normalized["timestamp"] = pd.to_numeric(normalized["timestamp"], errors="raise")
    normalized["behavior_label_clean"] = normalized["behavior_label"].fillna("").astype(str)
    normalized.loc[normalized["behavior_label_clean"].isin(["nan", "NA", "NaN"]), "behavior_label_clean"] = ""
    for column in FEATURE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for _, x_col, y_col, z_col in POINTS:
        for column in [x_col, y_col, z_col]:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.sort_values(["fish_id", "frame"]).reset_index(drop=True)


def _normalize_stagnation_events(event_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(event_df, ["fish_id", "start_frame", "end_frame", "start_time", "end_time", "duration", "behavior_label"])
    events = event_df[event_df["behavior_label"].astype(str).eq(PRED_LABEL)].copy().reset_index(drop=True)
    if events.empty:
        return pd.DataFrame(columns=["event_id", "fish_id", "start_frame", "end_frame", "start_time", "end_time", "duration", "behavior_label"])

    if "event_id" not in events.columns:
        if "behavior_id" in events.columns:
            events["event_id"] = events["behavior_id"]
        else:
            events["event_id"] = range(1, len(events) + 1)

    events["event_id"] = pd.to_numeric(events["event_id"], errors="raise").astype(int)
    events["fish_id"] = events["fish_id"].astype(str)
    events["start_frame"] = pd.to_numeric(events["start_frame"], errors="raise").astype(int)
    events["end_frame"] = pd.to_numeric(events["end_frame"], errors="raise").astype(int)
    events["start_time"] = pd.to_numeric(events["start_time"], errors="raise")
    events["end_time"] = pd.to_numeric(events["end_time"], errors="raise")
    events["duration"] = pd.to_numeric(events["duration"], errors="raise")
    return events[
        ["event_id", "fish_id", "start_frame", "end_frame", "start_time", "end_time", "duration", "behavior_label"]
    ].sort_values(["fish_id", "start_frame", "event_id"]).reset_index(drop=True)


def _event_window(feature_df: pd.DataFrame, event: Any) -> pd.DataFrame:
    return feature_df[
        feature_df["fish_id"].eq(str(event.fish_id))
        & feature_df["frame"].ge(int(event.start_frame))
        & feature_df["frame"].le(int(event.end_frame))
    ].copy()


def _feature_summary(window: pd.DataFrame) -> dict[str, float]:
    if window.empty:
        return {
            "mean_v": 0.0,
            "median_v": 0.0,
            "p90_v": 0.0,
            "max_v": 0.0,
            "mean_d_win": 0.0,
            "mean_E_move": 0.0,
            "mean_S_pose": 0.0,
        }
    return {
        "mean_v": float(window["v"].mean()),
        "median_v": float(window["v"].median()),
        "p90_v": float(window["v"].quantile(0.9)),
        "max_v": float(window["v"].max()),
        "mean_d_win": float(window["d_win"].mean()),
        "mean_E_move": float(window["E_move"].mean()),
        "mean_S_pose": float(window["S_pose"].mean()),
    }


def _label_counts(window: pd.DataFrame) -> dict[str, int]:
    if window.empty:
        return {}
    labels = window["behavior_label_clean"].replace("", "unlabeled")
    return {str(label): int(count) for label, count in labels.value_counts(sort=False).items()}


def _format_label_counts(label_counts: dict[str, int]) -> str:
    return "; ".join(f"{label}:{count}" for label, count in label_counts.items())


def _read_optional_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _matched_pred_event_ids(match_df: pd.DataFrame | None) -> set[int]:
    if match_df is None or match_df.empty or "pred_event_id" not in match_df.columns:
        return set()
    return set(pd.to_numeric(match_df["pred_event_id"], errors="coerce").dropna().astype(int).tolist())


def _match_row_for_event(match_df: pd.DataFrame, event_id: int) -> pd.Series | None:
    if match_df.empty or "pred_event_id" not in match_df.columns:
        return None
    rows = match_df[pd.to_numeric(match_df["pred_event_id"], errors="coerce").eq(event_id)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _plot_3d_tracks(ax: Any, window: pd.DataFrame) -> None:
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(window)))
    for point_name, x_col, y_col, z_col in POINTS:
        ax.plot(window[x_col], window[y_col], window[z_col], linewidth=1.8, label=point_name)
        ax.scatter(window[x_col], window[y_col], window[z_col], c=colors, s=7, alpha=0.55)
        ax.scatter(window[x_col].iloc[0], window[y_col].iloc[0], window[z_col].iloc[0], marker="o", s=45)
        ax.scatter(window[x_col].iloc[-1], window[y_col].iloc[-1], window[z_col].iloc[-1], marker="x", s=55)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("3D tracks, color = time")
    ax.legend(loc="upper left", fontsize=8)
    _set_equal_3d_axes(ax, window)


def _plot_xy_tracks(ax: Any, window: pd.DataFrame) -> None:
    for point_name, x_col, y_col, _ in POINTS:
        ax.plot(window[x_col], window[y_col], linewidth=1.8, label=point_name)
        ax.scatter(window[x_col].iloc[0], window[y_col].iloc[0], marker="o", s=40)
        ax.scatter(window[x_col].iloc[-1], window[y_col].iloc[-1], marker="x", s=50)
    im_window = window[window["behavior_label_clean"].eq(GT_LABEL)]
    if not im_window.empty:
        ax.scatter(im_window["head_x"], im_window["head_y"], s=10, c="#d62728", alpha=0.5, label="head during IM")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Top view")
    ax.legend(loc="best", fontsize=8)


def _plot_metrics(ax: Any, window: pd.DataFrame, columns: list[str], title: str) -> None:
    for column in columns:
        ax.plot(window["timestamp"], window[column], linewidth=1.5, label=column)
    ax.set_xlabel("time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _plot_label_regions(ax: Any, window: pd.DataFrame) -> None:
    label_colors = {
        GT_LABEL: "#d62728",
        "CL": "#9467bd",
        "TO": "#ff7f0e",
        "HYPE": "#2ca02c",
        "SW": "#1f77b4",
        "NT": "#8c564b",
    }
    y_min, y_max = ax.get_ylim()
    for label, color in label_colors.items():
        labeled = window[window["behavior_label_clean"].eq(label)]
        for start_time, end_time in _contiguous_time_ranges(labeled):
            ax.axvspan(start_time, end_time, color=color, alpha=0.12)
    ax.set_ylim(y_min, y_max)


def _plot_event_text(ax: Any, event: Any, window: pd.DataFrame, match_row: pd.Series | None) -> None:
    ax.axis("off")
    label_counts = _label_counts(window)
    im_frames = label_counts.get(GT_LABEL, 0)
    summary = _feature_summary(window)
    match_text = "matched IM: no"
    if match_row is not None:
        match_text = (
            f"matched IM: yes\n"
            f"gt_event_id: {int(match_row.get('gt_event_id', -1))}\n"
            f"match_iou: {float(match_row.get('match_iou', match_row.get('temporal_iou', 0.0))):.3f}"
        )
    text = "\n".join(
        [
            f"event_id: {int(event.event_id)}",
            f"fish_id: {str(event.fish_id)}",
            f"frames: {int(event.start_frame)}-{int(event.end_frame)}",
            f"time: {float(event.start_time):.3f}-{float(event.end_time):.3f} s",
            f"duration: {float(event.duration):.3f} s",
            match_text,
            f"frames in plot: {len(window)}",
            f"IM frames: {im_frames} ({im_frames / len(window):.3f})",
            f"labels: {_format_label_counts(label_counts)}",
            "",
            f"mean_v: {summary['mean_v']:.3f} mm/s",
            f"median_v: {summary['median_v']:.3f} mm/s",
            f"p90_v: {summary['p90_v']:.3f} mm/s",
            f"max_v: {summary['max_v']:.3f} mm/s",
            f"mean_d_win: {summary['mean_d_win']:.3f} mm",
            f"mean_E_move: {summary['mean_E_move']:.3f}",
            f"mean_S_pose: {summary['mean_S_pose']:.4f}",
        ]
    )
    ax.text(0.0, 1.0, text, transform=ax.transAxes, va="top", ha="left", fontsize=10, family="monospace")


def _contiguous_time_ranges(df: pd.DataFrame) -> list[tuple[float, float]]:
    if df.empty:
        return []
    ranges: list[tuple[float, float]] = []
    previous_frame: int | None = None
    current_start: float | None = None
    previous_time: float | None = None
    for row in df.sort_values("frame").itertuples(index=False):
        frame = int(row.frame)
        timestamp = float(row.timestamp)
        if previous_frame is None or frame != previous_frame + 1:
            if current_start is not None and previous_time is not None:
                ranges.append((current_start, previous_time))
            current_start = timestamp
        previous_frame = frame
        previous_time = timestamp
    if current_start is not None and previous_time is not None:
        ranges.append((current_start, previous_time))
    return ranges


def _set_equal_3d_axes(ax: Any, window: pd.DataFrame) -> None:
    values = []
    for _, x_col, y_col, z_col in POINTS:
        values.extend([window[x_col].to_numpy(), window[y_col].to_numpy(), window[z_col].to_numpy()])
    finite_values = np.concatenate([value[np.isfinite(value)] for value in values])
    if finite_values.size == 0:
        return
    ranges = []
    centers = []
    for columns in [(p[1], p[2], p[3]) for p in POINTS]:
        for column in columns:
            data = window[column].to_numpy(dtype=float)
            data = data[np.isfinite(data)]
            if data.size:
                ranges.append(float(data.max() - data.min()))
                centers.append((column, float((data.max() + data.min()) / 2)))
    radius = max(max(ranges) / 2 if ranges else 1.0, 1.0)
    center_map = dict(centers)
    x_center = np.mean([center_map.get(point[1], 0.0) for point in POINTS])
    y_center = np.mean([center_map.get(point[2], 0.0) for point in POINTS])
    z_center = np.mean([center_map.get(point[3], 0.0) for point in POINTS])
    ax.set_xlim(x_center - radius, x_center + radius)
    ax.set_ylim(y_center - radius, y_center + radius)
    ax.set_zlim(z_center - radius, z_center + radius)


def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(missing_columns[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PTZ stagnation diagnostic figures.")
    parser.add_argument("--feature-table", required=True, help="Path to feature_table.csv.")
    parser.add_argument("--behavior-events", required=True, help="Path to final behavior_events.csv.")
    parser.add_argument("--event-matches", default=None, help="Optional path to PTZ event_matches.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for figures and visualization_summary.csv.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    visualize_ptz_stagnation_events(
        feature_table_path=args.feature_table,
        behavior_events_path=args.behavior_events,
        event_matches_path=args.event_matches,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
