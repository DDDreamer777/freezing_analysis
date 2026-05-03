from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_TIMELINE_FEATURES = ["v", "d_win", "E_move", "S_pose", "dv"]


def prepare_visualization_dirs(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    base_output = Path(output_dir) if output_dir is not None else Path(config.get("paths", {}).get("output_dir", "outputs"))
    base_dir = base_output / "final" / "visualization"
    dirs = {
        "base": base_dir,
        "timeline": base_dir / "timeline",
        "trajectory": base_dir / "trajectory",
        "summary": base_dir / "summary",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def generate_visualizations(
    feature_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    state_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, list[Path]]:
    if not config.get("visualization", {}).get("enabled", True):
        return {"timeline": [], "trajectory": [], "summary": []}

    dirs = prepare_visualization_dirs(config, output_dir=output_dir)
    outputs = {"timeline": [], "trajectory": [], "summary": []}
    features = _normalize_feature_table(feature_df)
    behaviors = _normalize_behavior_table(behavior_df)
    states = state_df.copy() if state_df is not None else pd.DataFrame()
    segments = segment_df.copy() if segment_df is not None else pd.DataFrame()

    for fish_id, fish_df in features.groupby("fish_id", sort=True):
        path = dirs["timeline"] / f"fish_{fish_id}_overview.png"
        _plot_fish_overview(fish_df, behaviors[behaviors["fish_id"].eq(str(fish_id))], states, config, path)
        outputs["timeline"].append(path)

    max_events = int(config.get("visualization", {}).get("max_events_per_label", len(behaviors) or 0))
    selected_behaviors = _select_behaviors_for_visualization(behaviors, max_events=max_events)
    for event in selected_behaviors.itertuples(index=False):
        fish_df = features[features["fish_id"].eq(str(event.fish_id))]
        event_window = _event_context_window(fish_df, event, config)

        timeline_path = dirs["timeline"] / f"behavior_{int(event.behavior_id)}_timeline.png"
        _plot_behavior_timeline(event_window, event, config, timeline_path)
        outputs["timeline"].append(timeline_path)

        trajectory_path = dirs["trajectory"] / f"behavior_{int(event.behavior_id)}_xy.png"
        _plot_behavior_trajectory(event_window, event, trajectory_path)
        outputs["trajectory"].append(trajectory_path)

    summary_paths = [
        dirs["summary"] / "behavior_counts.png",
        dirs["summary"] / "behavior_duration_distribution.png",
        dirs["summary"] / "behavior_counts_by_fish.png",
    ]
    _plot_behavior_counts(behaviors, summary_paths[0])
    _plot_duration_distribution(behaviors, summary_paths[1])
    _plot_counts_by_fish(behaviors, summary_paths[2])
    outputs["summary"].extend(summary_paths)

    return outputs


def _normalize_feature_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    required = ["fish_id", "frame", "timestamp", "head_x", "head_y", "body_x", "body_y", "tail_x", "tail_y"]
    _require_columns(feature_df, required)
    normalized = feature_df.copy()
    normalized["fish_id"] = normalized["fish_id"].astype(str)

    numeric_columns = [
        "frame",
        "timestamp",
        "head_x",
        "head_y",
        "head_z",
        "body_x",
        "body_y",
        "body_z",
        "tail_x",
        "tail_y",
        "tail_z",
        "v",
        "d_win",
        "E_move",
        "S_pose",
        "dv",
        "p",
        "c",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")

    normalized["frame"] = normalized["frame"].astype(int)
    return normalized.sort_values(["fish_id", "frame"]).reset_index(drop=True)


def _normalize_behavior_table(behavior_df: pd.DataFrame) -> pd.DataFrame:
    if behavior_df is None or behavior_df.empty:
        return pd.DataFrame(columns=["behavior_id", "fish_id", "start_time", "end_time", "duration", "behavior_label"])
    required = ["fish_id", "start_time", "end_time", "duration", "behavior_label"]
    _require_columns(behavior_df, required)
    normalized = behavior_df.copy()
    normalized["fish_id"] = normalized["fish_id"].astype(str)
    normalized["start_time"] = pd.to_numeric(normalized["start_time"], errors="raise")
    normalized["end_time"] = pd.to_numeric(normalized["end_time"], errors="raise")
    normalized["duration"] = pd.to_numeric(normalized["duration"], errors="raise")
    normalized["behavior_label"] = normalized["behavior_label"].astype(str)
    if "behavior_id" not in normalized.columns:
        normalized["behavior_id"] = range(1, len(normalized) + 1)
    normalized["behavior_id"] = pd.to_numeric(normalized["behavior_id"], errors="raise").astype(int)
    return normalized.sort_values(["fish_id", "start_time", "behavior_id"]).reset_index(drop=True)


def _select_behaviors_for_visualization(behavior_df: pd.DataFrame, max_events: int) -> pd.DataFrame:
    if behavior_df.empty:
        return behavior_df
    if max_events <= 0:
        return behavior_df.iloc[0:0].copy()
    return (
        behavior_df.groupby("behavior_label", group_keys=False, sort=True)
        .head(max_events)
        .sort_values(["fish_id", "start_time", "behavior_id"])
        .reset_index(drop=True)
    )


def _event_context_window(fish_df: pd.DataFrame, event: Any, config: dict[str, Any]) -> pd.DataFrame:
    context = float(config.get("visualization", {}).get("default_time_context_seconds", 1.0))
    start_time = float(event.start_time) - context
    end_time = float(event.end_time) + context
    return fish_df[fish_df["timestamp"].between(start_time, end_time)].copy()


def _plot_fish_overview(
    fish_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    state_df: pd.DataFrame,
    config: dict[str, Any],
    output_path: Path,
) -> None:
    features = _timeline_features(config, fish_df)
    fig, axes = plt.subplots(len(features), 1, figsize=(12, max(3, len(features) * 2.0)), sharex=True)
    if len(features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        ax.plot(fish_df["timestamp"], fish_df[feature], linewidth=1.4, label=feature)
        _shade_behavior_events(ax, behavior_df)
        _shade_segment_states(ax, state_df, fish_df["fish_id"].iloc[0])
        ax.set_ylabel(feature)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Fish {fish_df['fish_id'].iloc[0]} overview")
    _save_figure(fig, output_path, config)


def _plot_behavior_timeline(
    event_window: pd.DataFrame,
    event: Any,
    config: dict[str, Any],
    output_path: Path,
) -> None:
    features = _timeline_features(config, event_window)
    fig, axes = plt.subplots(len(features), 1, figsize=(11, max(3, len(features) * 1.8)), sharex=True)
    if len(features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        ax.plot(event_window["timestamp"], event_window[feature], linewidth=1.6, label=feature)
        ax.axvspan(float(event.start_time), float(event.end_time), color="#d62728", alpha=0.14, label=str(event.behavior_label))
        ax.set_ylabel(feature)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Behavior {int(event.behavior_id)} timeline | fish {event.fish_id} | {event.behavior_label}")
    _save_figure(fig, output_path, config)


def _plot_behavior_trajectory(event_window: pd.DataFrame, event: Any, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for point_name, x_col, y_col in [("head", "head_x", "head_y"), ("body", "body_x", "body_y"), ("tail", "tail_x", "tail_y")]:
        ax.plot(event_window[x_col], event_window[y_col], linewidth=1.7, label=point_name)
        if not event_window.empty:
            ax.scatter(event_window[x_col].iloc[0], event_window[y_col].iloc[0], marker="o", s=35)
            ax.scatter(event_window[x_col].iloc[-1], event_window[y_col].iloc[-1], marker="x", s=45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Behavior {int(event.behavior_id)} XY trajectory | {event.behavior_label}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _save_figure(fig, output_path, {})


def _plot_behavior_counts(behavior_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = behavior_df["behavior_label"].value_counts() if not behavior_df.empty else pd.Series(dtype=int)
    counts.plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_title("Behavior counts")
    ax.set_xlabel("behavior")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_path, {})


def _plot_duration_distribution(behavior_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if not behavior_df.empty:
        ax.hist(behavior_df["duration"], bins=min(10, max(1, len(behavior_df))), color="#59a14f", alpha=0.85)
    ax.set_title("Behavior duration distribution")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_path, {})


def _plot_counts_by_fish(behavior_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not behavior_df.empty:
        pivot = behavior_df.pivot_table(index="fish_id", columns="behavior_label", values="behavior_id", aggfunc="count", fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Behavior counts by fish")
    ax.set_xlabel("fish_id")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _save_figure(fig, output_path, {})


def _timeline_features(config: dict[str, Any], feature_df: pd.DataFrame) -> list[str]:
    configured = config.get("visualization", {}).get("timeline_features", DEFAULT_TIMELINE_FEATURES)
    return [feature for feature in configured if feature in feature_df.columns]


def _shade_behavior_events(ax: Any, behavior_df: pd.DataFrame) -> None:
    if behavior_df.empty:
        return
    labels = sorted(behavior_df["behavior_label"].unique())
    colors = plt.cm.Set2(range(len(labels)))
    color_map = dict(zip(labels, colors))
    for row in behavior_df.itertuples(index=False):
        ax.axvspan(float(row.start_time), float(row.end_time), color=color_map[str(row.behavior_label)], alpha=0.12)


def _shade_segment_states(ax: Any, state_df: pd.DataFrame, fish_id: str) -> None:
    if state_df.empty or not {"fish_id", "start_time", "end_time", "segment_state"}.issubset(state_df.columns):
        return
    state_colors = {"pause_candidate": "#d62728", "transition": "#ff7f0e", "active": "#2ca02c"}
    fish_states = state_df[state_df["fish_id"].astype(str).eq(str(fish_id))]
    for row in fish_states.itertuples(index=False):
        color = state_colors.get(str(row.segment_state))
        if color is not None:
            ax.axvspan(float(row.start_time), float(row.end_time), color=color, alpha=0.05)


def _save_figure(fig: Any, output_path: Path, config: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("visualization", {}).get("dpi", 120)) if config else 120
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(missing_columns[0])
