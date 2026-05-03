from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


BEHAVIOR_PRIORITY_DEFAULT = ["stagnation", "twist", "glide"]


def classify_behavior_candidates(
    event_df: pd.DataFrame,
    state_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = build_classification_inputs(event_df, state_df, scored_df, config)
    stagnation_event_df = classify_stagnation_events(inputs, config)
    twist_segment_df = classify_twist_segments(inputs, config)
    glide_segment_df = classify_glide_segments(inputs, config)
    twist_event_df = merge_twist_segments_to_events(twist_segment_df, config)
    glide_event_df = merge_glide_segments_to_events(glide_segment_df, config)
    final_behavior_df = resolve_behavior_conflicts(stagnation_event_df, twist_event_df, glide_event_df, config)
    save_classification_outputs(
        stagnation_event_df,
        twist_event_df,
        glide_event_df,
        final_behavior_df,
        config,
        output_dir,
    )
    return stagnation_event_df, twist_event_df, glide_event_df, final_behavior_df


def build_classification_inputs(
    event_df: pd.DataFrame,
    state_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    segment_df = _prepare_segment_df(state_df, scored_df)
    pause_event_segment_ids = _extract_pause_event_segment_ids(event_df)
    thresholds = config.get("thresholds", {})

    stagnation_candidates = event_df.copy()
    glide_candidates = segment_df[
        (segment_df["segment_state"] == "transition")
        & (~segment_df["segment_id"].astype(str).isin(pause_event_segment_ids))
    ].copy()

    active_twist_candidates = segment_df[
        (segment_df["segment_state"] == "active")
        & (
            (segment_df["mean_d_win"] <= float(thresholds["D_pause"]))
            | (segment_df["mean_S_pose"] <= float(thresholds["S_th_twist"]))
            | (segment_df["mean_c"] >= float(thresholds["C_twist"]))
        )
    ].copy()

    twist_candidates = pd.concat(
        [
            segment_df[segment_df["segment_state"] == "transition"],
            segment_df[segment_df["segment_state"] == "pause_candidate"],
            active_twist_candidates,
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["segment_id"]).reset_index(drop=True)

    return {
        "stagnation_candidates": stagnation_candidates,
        "glide_candidates": glide_candidates,
        "twist_candidates": twist_candidates,
    }


def classify_stagnation_events(inputs: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    if not config.get("classification", {}).get("enable_stagnation_detection", True):
        return pd.DataFrame(columns=_stagnation_event_columns())

    candidates = inputs["stagnation_candidates"].copy()
    if candidates.empty:
        return pd.DataFrame(columns=_stagnation_event_columns())

    thresholds = config.get("thresholds", {})
    candidates["rule_v"] = candidates["mean_v"] <= float(thresholds["v_th_pause"])
    candidates["rule_d_win"] = candidates["mean_d_win"] <= float(thresholds["D_pause"])
    candidates["rule_E_move"] = candidates["mean_E_move"] <= float(thresholds["E_th_pause"])
    candidates["rule_S_pose"] = candidates["mean_S_pose"] >= float(thresholds["S_th_pause"])
    candidates["passes_stagnation_rules"] = (
        candidates["rule_v"]
        & candidates["rule_d_win"]
        & candidates["rule_E_move"]
        & candidates["rule_S_pose"]
        & (candidates["duration"] >= float(thresholds["T_min"]))
    )
    candidates["behavior_label"] = "stagnation"
    candidates["classification_score"] = candidates["mean_pause_score"]
    result = candidates[candidates["passes_stagnation_rules"]].reset_index(drop=True)
    if result.empty:
        return pd.DataFrame(columns=_stagnation_event_columns())
    return result


def classify_twist_segments(inputs: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    if not config.get("classification", {}).get("enable_twist_detection", True):
        return pd.DataFrame(columns=_classified_segment_columns())

    candidates = inputs["twist_candidates"].copy()
    if candidates.empty:
        return pd.DataFrame(columns=_classified_segment_columns())

    thresholds = config.get("thresholds", {})
    min_score = float(config.get("classification", {}).get("twist_min_score", 0.6))

    candidates["twist_signal_d_win"] = candidates["mean_d_win"].apply(
        lambda value: _score_low_value_signal(value, float(thresholds["D_pause"]))
    )
    candidates["twist_signal_S_pose"] = candidates["mean_S_pose"].apply(
        lambda value: _score_low_pose_twist_signal(value, float(thresholds["S_th_twist"]))
    )
    candidates["twist_signal_p"] = candidates["mean_p"].apply(
        lambda value: _score_high_value_signal(value, float(thresholds["P_pause"]))
    )
    candidates["twist_signal_c"] = candidates["mean_c"].apply(
        lambda value: _score_high_value_signal(value, float(thresholds["C_twist"]))
    )
    candidates["classification_score"] = (
        0.25 * candidates["twist_signal_d_win"]
        + 0.25 * candidates["twist_signal_S_pose"]
        + 0.20 * candidates["twist_signal_p"]
        + 0.30 * candidates["twist_signal_c"]
    )
    candidates["rule_d_win"] = candidates["mean_d_win"] <= float(thresholds["D_pause"])
    candidates["rule_S_pose"] = candidates["mean_S_pose"] <= float(thresholds["S_th_twist"])
    candidates["rule_p"] = candidates["mean_p"] >= float(thresholds["P_pause"])
    candidates["rule_c"] = candidates["mean_c"] >= float(thresholds["C_twist"])
    candidates["passes_twist_rules"] = (
        candidates["rule_d_win"]
        & candidates["rule_S_pose"]
        & candidates["rule_p"]
        & candidates["rule_c"]
        & (candidates["classification_score"] >= min_score)
    )
    candidates["behavior_label"] = "twist"
    result = candidates[candidates["passes_twist_rules"]].reset_index(drop=True)
    if result.empty:
        return pd.DataFrame(columns=_classified_segment_columns())
    return result


def classify_glide_segments(inputs: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    if not config.get("classification", {}).get("enable_glide_detection", True):
        return pd.DataFrame(columns=_classified_segment_columns())

    candidates = inputs["glide_candidates"].copy()
    if candidates.empty:
        return pd.DataFrame(columns=_classified_segment_columns())

    thresholds = config.get("thresholds", {})
    min_score = float(config.get("classification", {}).get("glide_min_score", 0.6))

    candidates["glide_signal_v"] = candidates["mean_v"].apply(
        lambda value: _score_low_value_signal(value, float(thresholds["v_th_glide"]))
    )
    candidates["glide_signal_d_win"] = candidates["mean_d_win"].apply(
        lambda value: _score_low_value_signal(value, float(thresholds["D_glide"]))
    )
    candidates["glide_signal_E_move"] = candidates["mean_E_move"].apply(
        lambda value: _score_low_value_signal(value, float(thresholds["E_th_pause"]))
    )
    candidates["glide_signal_S_pose"] = candidates["mean_S_pose"].apply(
        lambda value: _score_high_value_signal(value, float(thresholds["S_th_twist"]))
    )
    candidates["glide_signal_dv"] = candidates["mean_dv"].apply(
        lambda value: _score_high_value_signal(value, float(thresholds["dv_th_glide"]))
    )
    candidates["classification_score"] = (
        0.25 * candidates["glide_signal_v"]
        + 0.20 * candidates["glide_signal_d_win"]
        + 0.20 * candidates["glide_signal_E_move"]
        + 0.15 * candidates["glide_signal_S_pose"]
        + 0.20 * candidates["glide_signal_dv"]
    )
    candidates["rule_v"] = candidates["mean_v"] <= float(thresholds["v_th_glide"])
    candidates["rule_d_win"] = candidates["mean_d_win"] <= float(thresholds["D_glide"])
    candidates["rule_E_move"] = candidates["mean_E_move"] <= float(thresholds["E_th_pause"])
    candidates["rule_S_pose"] = candidates["mean_S_pose"] >= float(thresholds["S_th_twist"])
    candidates["rule_dv"] = candidates["mean_dv"] >= float(thresholds["dv_th_glide"])
    candidates["passes_glide_rules"] = (
        candidates["rule_v"]
        & candidates["rule_d_win"]
        & candidates["rule_E_move"]
        & candidates["rule_S_pose"]
        & candidates["rule_dv"]
        & (candidates["classification_score"] >= min_score)
    )
    candidates["behavior_label"] = "glide"
    result = candidates[candidates["passes_glide_rules"]].reset_index(drop=True)
    if result.empty:
        return pd.DataFrame(columns=_classified_segment_columns())
    return result


def merge_twist_segments_to_events(twist_segment_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return _merge_classified_segments_to_events(twist_segment_df, config, "twist")


def merge_glide_segments_to_events(glide_segment_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return _merge_classified_segments_to_events(glide_segment_df, config, "glide")


def resolve_behavior_conflicts(
    stagnation_event_df: pd.DataFrame,
    twist_event_df: pd.DataFrame,
    glide_event_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    priority_order = config.get("classification", {}).get("final_priority", BEHAVIOR_PRIORITY_DEFAULT)
    priority_map = {label: index for index, label in enumerate(priority_order)}

    combined_rows: list[dict[str, Any]] = []
    combined_rows.extend(_normalize_behavior_events(stagnation_event_df, "stagnation_event", priority_map))
    combined_rows.extend(_normalize_behavior_events(twist_event_df, "twist_event", priority_map))
    combined_rows.extend(_normalize_behavior_events(glide_event_df, "glide_event", priority_map))
    if not combined_rows:
        return pd.DataFrame(columns=_final_behavior_columns())

    ordered_rows = sorted(
        combined_rows,
        key=lambda row: (row["priority_rank"], row["fish_id"], row["start_time"], row["end_time"]),
    )
    accepted_rows: list[dict[str, Any]] = []
    for row in ordered_rows:
        if any(_events_overlap(row, kept_row) for kept_row in accepted_rows if kept_row["fish_id"] == row["fish_id"]):
            continue
        accepted_rows.append(row)

    final_rows = []
    for behavior_id, row in enumerate(sorted(accepted_rows, key=lambda item: (item["fish_id"], item["start_time"], item["priority_rank"])), start=1):
        final_rows.append(
            {
                "behavior_id": behavior_id,
                "fish_id": row["fish_id"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "duration": row["duration"],
                "behavior_label": row["behavior_label"],
                "source_type": row["source_type"],
                "source_ids": row["source_ids"],
                "classification_score": row["classification_score"],
                "resolved_by_priority": True,
            }
        )

    return pd.DataFrame(final_rows)


def save_classification_outputs(
    stagnation_event_df: pd.DataFrame,
    twist_event_df: pd.DataFrame,
    glide_event_df: pd.DataFrame,
    final_behavior_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    base_dir = Path(output_dir) if output_dir is not None else Path(config.get("paths", {}).get("output_dir", "outputs"))
    intermediate_dir = base_dir / "intermediate" / "classification"
    final_dir = base_dir / "final"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    stagnation_path = intermediate_dir / "stagnation_events.csv"
    twist_path = intermediate_dir / "twist_events.csv"
    glide_path = intermediate_dir / "glide_events.csv"
    before_eval_path = intermediate_dir / "final_behavior_events_before_eval.csv"
    final_path = final_dir / "behavior_events.csv"

    stagnation_event_df.to_csv(stagnation_path, index=False)
    twist_event_df.to_csv(twist_path, index=False)
    glide_event_df.to_csv(glide_path, index=False)
    final_behavior_df.to_csv(before_eval_path, index=False)
    final_behavior_df.to_csv(final_path, index=False)

    return {
        "stagnation_events": stagnation_path,
        "twist_events": twist_path,
        "glide_events": glide_path,
        "behavior_events_before_eval": before_eval_path,
        "behavior_events": final_path,
    }


def _prepare_segment_df(state_df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    segment_df = state_df.copy()
    missing_columns = [
        column
        for column in [
            "mean_v",
            "mean_d_win",
            "mean_E_move",
            "mean_S_pose",
            "mean_p",
            "mean_c",
            "mean_dv",
            "pause_score",
            "activity_score",
        ]
        if column not in segment_df.columns and column in scored_df.columns
    ]
    if missing_columns:
        segment_df = segment_df.merge(scored_df[["segment_id", *missing_columns]], on="segment_id", how="left")
    return segment_df


def _extract_pause_event_segment_ids(event_df: pd.DataFrame) -> set[str]:
    segment_ids: set[str] = set()
    if event_df.empty or "segment_ids" not in event_df.columns:
        return segment_ids
    for value in event_df["segment_ids"].dropna():
        segment_ids.update(part.strip() for part in str(value).split(",") if part.strip())
    return segment_ids


def _merge_classified_segments_to_events(
    segment_df: pd.DataFrame,
    config: dict[str, Any],
    behavior_label: str,
) -> pd.DataFrame:
    if segment_df.empty:
        return pd.DataFrame(columns=_merged_event_columns())

    merge_gap_seconds = float(config.get("detection", {}).get("merge_gap_seconds", 0.3))
    events: list[dict[str, Any]] = []
    event_id = 0

    for fish_id, group in segment_df.groupby("fish_id", sort=False):
        rows = list(group.sort_values("start_time").itertuples(index=False))
        current_group = [rows[0]]
        for row in rows[1:]:
            gap_seconds = float(row.start_time - current_group[-1].end_time)
            if gap_seconds <= merge_gap_seconds:
                current_group.append(row)
                continue
            event_id += 1
            events.append(_build_merged_event_record(event_id, str(fish_id), current_group, behavior_label))
            current_group = [row]
        event_id += 1
        events.append(_build_merged_event_record(event_id, str(fish_id), current_group, behavior_label))

    return pd.DataFrame(events)


def _build_merged_event_record(event_id: int, fish_id: str, rows: list, behavior_label: str) -> dict[str, Any]:
    first_row = rows[0]
    last_row = rows[-1]
    score_values = [float(row.classification_score) for row in rows]
    record = {
        "event_id": event_id,
        "fish_id": fish_id,
        "start_frame": int(first_row.start_frame),
        "end_frame": int(last_row.end_frame),
        "start_time": float(first_row.start_time),
        "end_time": float(last_row.end_time),
        "duration": float(last_row.end_time - first_row.start_time),
        "n_segments": len(rows),
        "segment_ids": ",".join(str(int(row.segment_id)) for row in rows),
        "behavior_label": behavior_label,
        "classification_score": sum(score_values) / len(score_values),
        "max_classification_score": max(score_values),
    }

    if behavior_label == "twist":
        record.update(
            {
                "mean_d_win": sum(float(row.mean_d_win) for row in rows) / len(rows),
                "mean_S_pose": sum(float(row.mean_S_pose) for row in rows) / len(rows),
                "mean_p": sum(float(row.mean_p) for row in rows) / len(rows),
                "mean_c": sum(float(row.mean_c) for row in rows) / len(rows),
            }
        )
    else:
        record.update(
            {
                "mean_v": sum(float(row.mean_v) for row in rows) / len(rows),
                "mean_d_win": sum(float(row.mean_d_win) for row in rows) / len(rows),
                "mean_E_move": sum(float(row.mean_E_move) for row in rows) / len(rows),
                "mean_S_pose": sum(float(row.mean_S_pose) for row in rows) / len(rows),
                "mean_dv": sum(float(row.mean_dv) for row in rows) / len(rows),
            }
        )
    return record


def _normalize_behavior_events(event_df: pd.DataFrame, source_type: str, priority_map: dict[str, int]) -> list[dict[str, Any]]:
    normalized_rows = []
    if event_df.empty:
        return normalized_rows
    for row in event_df.itertuples(index=False):
        normalized_rows.append(
            {
                "fish_id": str(row.fish_id),
                "start_frame": int(row.start_frame),
                "end_frame": int(row.end_frame),
                "start_time": float(row.start_time),
                "end_time": float(row.end_time),
                "duration": float(row.duration),
                "behavior_label": str(row.behavior_label),
                "source_type": source_type,
                "source_ids": str(getattr(row, "event_id", getattr(row, "segment_ids", ""))),
                "classification_score": float(row.classification_score),
                "priority_rank": priority_map.get(str(row.behavior_label), len(priority_map)),
            }
        )
    return normalized_rows


def _events_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return not (left["end_time"] < right["start_time"] or right["end_time"] < left["start_time"])


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _score_low_value_signal(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clip_score(1.0 - float(value) / threshold)


def _score_high_value_signal(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clip_score(float(value) / threshold)


def _score_low_pose_twist_signal(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clip_score((threshold - float(value)) / threshold)


def _classified_segment_columns() -> list[str]:
    return [
        "segment_id",
        "fish_id",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "duration",
        "behavior_label",
        "classification_score",
    ]


def _merged_event_columns() -> list[str]:
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
        "behavior_label",
        "classification_score",
        "max_classification_score",
    ]


def _stagnation_event_columns() -> list[str]:
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
        "behavior_label",
        "classification_score",
        "passes_stagnation_rules",
        "rule_v",
        "rule_d_win",
        "rule_E_move",
        "rule_S_pose",
    ]


def _final_behavior_columns() -> list[str]:
    return [
        "behavior_id",
        "fish_id",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "duration",
        "behavior_label",
        "source_type",
        "source_ids",
        "classification_score",
        "resolved_by_priority",
    ]
