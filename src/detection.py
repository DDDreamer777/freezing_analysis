from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


SOFT_SCORE_SHARPNESS = 8.0


def detect_pause_candidates(
    candidate_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored_df = score_candidate_segments(candidate_df, config)
    state_df = label_segment_states(scored_df, config)
    event_df = merge_pause_candidates(state_df, config)
    event_df = filter_candidate_events(event_df, config)
    save_detection_outputs(scored_df, state_df, event_df, config, output_dir)
    return scored_df, state_df, event_df


def score_candidate_segments(candidate_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    scored_df = candidate_df.copy()
    thresholds = config.get("thresholds", {})
    weights = config.get("detection", {}).get("pause_score_weights", {})

    scored_df["pause_signal_v"] = scored_df["mean_v"].apply(
        lambda value: _score_low_value_signal(value, float(thresholds["v_th_pause"]))
    )
    scored_df["pause_signal_d_win"] = scored_df["mean_d_win"].apply(
        lambda value: _score_low_value_signal(value, float(thresholds["D_pause"]))
    )
    scored_df["pause_signal_E_move"] = scored_df["mean_E_move"].apply(
        lambda value: _score_low_value_signal(value, float(thresholds["E_th_pause"]))
    )
    scored_df["pause_signal_S_pose"] = scored_df["mean_S_pose"].apply(
        lambda value: _score_high_value_signal(value, float(thresholds["S_th_pause"]))
    )

    scored_df["pause_score"] = scored_df.apply(
        lambda row: (
            float(weights.get("v", 0.0)) * row["pause_signal_v"]
            + float(weights.get("d_win", 0.0)) * row["pause_signal_d_win"]
            + float(weights.get("E_move", 0.0)) * row["pause_signal_E_move"]
            + float(weights.get("S_pose", 0.0)) * row["pause_signal_S_pose"]
        ),
        axis=1,
    )
    scored_df["activity_score"] = 1.0 - scored_df["pause_score"]
    scored_df["duration_support"] = scored_df["duration"].apply(
        lambda value: _clip_score(value / float(thresholds["T_min"]))
    )
    scored_df["passes_event_min_duration"] = scored_df["duration"] >= float(thresholds["T_min"])
    return scored_df


def label_segment_states(scored_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    state_df = scored_df.copy()
    detection_config = config.get("detection", {})
    enter_score = float(detection_config.get("pause_enter_score", 0.7))
    exit_score = float(detection_config.get("pause_exit_score", 0.4))

    def determine_state(row: pd.Series) -> str:
        if row["pause_score"] >= enter_score:
            return "pause_candidate"
        if row["pause_score"] <= exit_score:
            return "active"
        return "transition"

    state_df["segment_state"] = state_df.apply(determine_state, axis=1)
    state_df["is_pause_like"] = state_df["segment_state"] == "pause_candidate"
    state_df["is_active_like"] = state_df["segment_state"] == "active"
    state_df["is_transition_like"] = state_df["segment_state"] == "transition"
    return state_df


def merge_pause_candidates(state_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if state_df.empty:
        return pd.DataFrame(columns=_event_columns())

    detection_config = config.get("detection", {})
    merge_gap_seconds = float(detection_config.get("merge_gap_seconds", 0.3))
    bridge_max_duration = float(detection_config.get("bridge_max_duration_seconds", 0.5))

    events: list[dict[str, Any]] = []
    event_id = 0

    for fish_id, group in state_df.groupby("fish_id", sort=False):
        rows = list(group.sort_values("start_time").itertuples(index=False))
        i = 0
        while i < len(rows):
            current = rows[i]
            if current.segment_state != "pause_candidate":
                i += 1
                continue

            event_segments = [current]
            contains_transition_bridge = False
            j = i + 1
            while j < len(rows):
                next_row = rows[j]
                previous_row = event_segments[-1]
                gap_seconds = float(next_row.start_time - previous_row.end_time)
                if gap_seconds > merge_gap_seconds:
                    break

                if next_row.segment_state == "pause_candidate":
                    event_segments.append(next_row)
                    j += 1
                    continue

                if next_row.segment_state == "transition":
                    if j + 1 >= len(rows):
                        break
                    following_row = rows[j + 1]
                    bridge_gap_seconds = float(following_row.start_time - next_row.end_time)
                    if _can_bridge_transition(
                        next_row,
                        following_row,
                        bridge_max_duration,
                        merge_gap_seconds,
                        bridge_gap_seconds,
                    ):
                        event_segments.append(next_row)
                        event_segments.append(following_row)
                        contains_transition_bridge = True
                        j += 2
                        continue
                break

            event_id += 1
            events.append(_build_event_record(event_id, str(fish_id), event_segments, contains_transition_bridge))
            i = j

    if not events:
        return pd.DataFrame(columns=_event_columns())
    return pd.DataFrame(events)


def filter_candidate_events(event_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if event_df.empty:
        return event_df.copy()
    min_duration = float(config.get("thresholds", {}).get("T_min", 1.0))
    return event_df[event_df["duration"] >= min_duration].reset_index(drop=True)


def save_detection_outputs(
    scored_df: pd.DataFrame,
    state_df: pd.DataFrame,
    event_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    base_dir = Path(output_dir) if output_dir is not None else Path(config.get("paths", {}).get("output_dir", "outputs"))
    save_dir = base_dir / "intermediate" / "detection"
    save_dir.mkdir(parents=True, exist_ok=True)

    score_path = save_dir / "candidate_segment_scores.csv"
    state_path = save_dir / "candidate_segment_states.csv"
    event_path = save_dir / "pause_candidate_events.csv"

    scored_df.to_csv(score_path, index=False)
    state_df.to_csv(state_path, index=False)
    event_df.to_csv(event_path, index=False)

    return {
        "segment_scores": score_path,
        "segment_states": state_path,
        "pause_candidate_events": event_path,
    }


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _score_low_value_signal(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    ratio = float(value) / threshold
    return _sigmoid(SOFT_SCORE_SHARPNESS * (1.0 - ratio))


def _score_high_value_signal(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    ratio = float(value) / threshold
    return _sigmoid(SOFT_SCORE_SHARPNESS * (ratio - 1.0))


def _can_bridge_transition(
    transition_row,
    following_row,
    bridge_max_duration: float,
    merge_gap_seconds: float,
    bridge_gap_seconds: float,
) -> bool:
    return (
        transition_row.segment_state == "transition"
        and transition_row.duration <= bridge_max_duration
        and following_row.segment_state == "pause_candidate"
        and bridge_gap_seconds <= merge_gap_seconds
    )


def _build_event_record(
    event_id: int,
    fish_id: str,
    event_segments: list,
    contains_transition_bridge: bool,
) -> dict[str, Any]:
    first_segment = event_segments[0]
    last_segment = event_segments[-1]
    pause_scores = [float(segment.pause_score) for segment in event_segments]
    activity_scores = [float(segment.activity_score) for segment in event_segments]
    mean_v = [float(segment.mean_v) for segment in event_segments]
    mean_d_win = [float(segment.mean_d_win) for segment in event_segments]
    mean_E_move = [float(segment.mean_E_move) for segment in event_segments]
    mean_S_pose = [float(segment.mean_S_pose) for segment in event_segments]

    return {
        "event_id": event_id,
        "fish_id": fish_id,
        "start_frame": int(first_segment.start_frame),
        "end_frame": int(last_segment.end_frame),
        "start_time": float(first_segment.start_time),
        "end_time": float(last_segment.end_time),
        "duration": float(last_segment.end_time - first_segment.start_time),
        "n_segments": int(len(event_segments)),
        "segment_ids": ",".join(str(int(segment.segment_id)) for segment in event_segments),
        "mean_pause_score": sum(pause_scores) / len(pause_scores),
        "max_pause_score": max(pause_scores),
        "mean_activity_score": sum(activity_scores) / len(activity_scores),
        "contains_transition_bridge": contains_transition_bridge,
        "mean_v": sum(mean_v) / len(mean_v),
        "mean_d_win": sum(mean_d_win) / len(mean_d_win),
        "mean_E_move": sum(mean_E_move) / len(mean_E_move),
        "mean_S_pose": sum(mean_S_pose) / len(mean_S_pose),
    }


def _event_columns() -> list[str]:
    return [
        "event_id",
        "fish_id",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "duration",
        "n_segments",
        "segment_ids",
        "mean_pause_score",
        "max_pause_score",
        "mean_activity_score",
        "contains_transition_bridge",
        "mean_v",
        "mean_d_win",
        "mean_E_move",
        "mean_S_pose",
    ]
