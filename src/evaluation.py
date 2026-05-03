from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_GT_COLUMNS = ["fish_id", "start_second", "end_second", "label"]
REQUIRED_PRED_COLUMNS = ["fish_id", "start_time", "end_time", "behavior_label"]


def evaluate_behavior_events(
    pred_event_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_event_df = normalize_event_labels(build_gt_event_table(gt_df, config), config, source="gt")
    pred_event_df = normalize_event_labels(build_pred_event_table(pred_event_df, config), config, source="pred")
    candidate_df = build_event_match_candidates(gt_event_df, pred_event_df)
    matched_df = match_event_pairs(candidate_df, config)
    unmatched_gt_df, unmatched_pred_df = build_unmatched_events(gt_event_df, pred_event_df, matched_df)
    summary_df = summarize_event_metrics(gt_event_df, pred_event_df, matched_df)
    save_evaluation_outputs(matched_df, unmatched_gt_df, unmatched_pred_df, summary_df, config, output_dir)
    return matched_df, summary_df


def build_gt_event_table(gt_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    _require_columns(gt_df, REQUIRED_GT_COLUMNS)
    event_df = gt_df.copy().reset_index(drop=True)
    event_df = event_df.rename(
        columns={
            "start_second": "start_time",
            "end_second": "end_time",
            "label": "label",
        }
    )
    event_df["fish_id"] = event_df["fish_id"].astype(str)
    event_df["label"] = event_df["label"].astype(str)
    event_df["start_time"] = pd.to_numeric(event_df["start_time"], errors="raise")
    event_df["end_time"] = pd.to_numeric(event_df["end_time"], errors="raise")
    event_df["duration"] = event_df["end_time"] - event_df["start_time"]
    event_df.insert(0, "event_id", range(1, len(event_df) + 1))
    return event_df[["event_id", "fish_id", "label", "start_time", "end_time", "duration"]]


def build_pred_event_table(pred_event_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    _require_columns(pred_event_df, REQUIRED_PRED_COLUMNS)
    event_df = pred_event_df.copy().reset_index(drop=True)
    event_df = event_df.rename(columns={"behavior_label": "label"})
    event_df["fish_id"] = event_df["fish_id"].astype(str)
    event_df["label"] = event_df["label"].astype(str)
    event_df["start_time"] = pd.to_numeric(event_df["start_time"], errors="raise")
    event_df["end_time"] = pd.to_numeric(event_df["end_time"], errors="raise")
    if "duration" not in event_df.columns:
        event_df["duration"] = event_df["end_time"] - event_df["start_time"]
    if "behavior_id" in event_df.columns:
        event_df["event_id"] = pd.to_numeric(event_df["behavior_id"], errors="raise").astype(int)
    elif "event_id" in event_df.columns:
        event_df["event_id"] = pd.to_numeric(event_df["event_id"], errors="raise").astype(int)
    else:
        event_df.insert(0, "event_id", range(1, len(event_df) + 1))
    return event_df[["event_id", "fish_id", "label", "start_time", "end_time", "duration"]]


def normalize_event_labels(
    event_df: pd.DataFrame,
    config: dict[str, Any],
    source: str,
) -> pd.DataFrame:
    label_map = config.get("evaluation", {}).get("label_map", {}).get(source, {})
    normalized = event_df.copy()
    normalized["label"] = normalized["label"].map(lambda value: label_map.get(str(value), str(value)))
    return normalized


def build_event_match_candidates(
    gt_event_df: pd.DataFrame,
    pred_event_df: pd.DataFrame,
) -> pd.DataFrame:
    candidate_rows: list[dict[str, Any]] = []
    for gt_row in gt_event_df.itertuples(index=False):
        for pred_row in pred_event_df.itertuples(index=False):
            if gt_row.fish_id != pred_row.fish_id or gt_row.label != pred_row.label:
                continue
            overlap_duration = min(float(gt_row.end_time), float(pred_row.end_time)) - max(float(gt_row.start_time), float(pred_row.start_time))
            if overlap_duration <= 0:
                continue
            union_duration = max(float(gt_row.end_time), float(pred_row.end_time)) - min(float(gt_row.start_time), float(pred_row.start_time))
            temporal_iou = overlap_duration / union_duration if union_duration > 0 else 0.0
            candidate_rows.append(
                {
                    "gt_event_id": int(gt_row.event_id),
                    "pred_event_id": int(pred_row.event_id),
                    "fish_id": str(gt_row.fish_id),
                    "label": str(gt_row.label),
                    "gt_start_time": float(gt_row.start_time),
                    "gt_end_time": float(gt_row.end_time),
                    "pred_start_time": float(pred_row.start_time),
                    "pred_end_time": float(pred_row.end_time),
                    "overlap_duration": float(overlap_duration),
                    "union_duration": float(union_duration),
                    "temporal_iou": float(temporal_iou),
                }
            )
    if not candidate_rows:
        return pd.DataFrame(columns=_matched_columns())
    return pd.DataFrame(candidate_rows)


def match_event_pairs(
    candidate_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame(columns=_matched_columns())

    threshold = float(config.get("evaluation", {}).get("event_iou_threshold", 0.3))
    eligible_df = candidate_df[candidate_df["temporal_iou"] >= threshold].copy()
    if eligible_df.empty:
        return pd.DataFrame(columns=_matched_columns())

    eligible_df = eligible_df.sort_values(
        ["temporal_iou", "overlap_duration", "gt_event_id", "pred_event_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    matched_gt_ids: set[int] = set()
    matched_pred_ids: set[int] = set()
    matched_rows: list[dict[str, Any]] = []
    for row in eligible_df.itertuples(index=False):
        if row.gt_event_id in matched_gt_ids or row.pred_event_id in matched_pred_ids:
            continue
        matched_gt_ids.add(int(row.gt_event_id))
        matched_pred_ids.add(int(row.pred_event_id))
        matched_rows.append({column: getattr(row, column) for column in _matched_columns()})

    if not matched_rows:
        return pd.DataFrame(columns=_matched_columns())
    return pd.DataFrame(matched_rows)


def build_unmatched_events(
    gt_event_df: pd.DataFrame,
    pred_event_df: pd.DataFrame,
    matched_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched_gt_ids = set(matched_df["gt_event_id"].tolist()) if not matched_df.empty else set()
    matched_pred_ids = set(matched_df["pred_event_id"].tolist()) if not matched_df.empty else set()
    unmatched_gt_df = gt_event_df[~gt_event_df["event_id"].isin(matched_gt_ids)].reset_index(drop=True)
    unmatched_pred_df = pred_event_df[~pred_event_df["event_id"].isin(matched_pred_ids)].reset_index(drop=True)
    return unmatched_gt_df, unmatched_pred_df


def summarize_event_metrics(
    gt_event_df: pd.DataFrame,
    pred_event_df: pd.DataFrame,
    matched_df: pd.DataFrame,
) -> pd.DataFrame:
    labels = sorted(set(gt_event_df["label"].tolist()) | set(pred_event_df["label"].tolist()))
    summary_rows: list[dict[str, Any]] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = len(gt_event_df)
    total_pred = len(pred_event_df)

    for label in labels:
        tp = int((matched_df["label"] == label).sum()) if not matched_df.empty else 0
        support_gt = int((gt_event_df["label"] == label).sum())
        support_pred = int((pred_event_df["label"] == label).sum())
        fp = support_pred - tp
        fn = support_gt - tp
        precision = tp / support_pred if support_pred > 0 else 0.0
        recall = tp / support_gt if support_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        summary_rows.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support_gt": support_gt,
                "support_pred": support_pred,
            }
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / total_pred if total_pred > 0 else 0.0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0.0
    summary_rows.append(
        {
            "label": "overall",
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": float(overall_precision),
            "recall": float(overall_recall),
            "f1": float(overall_f1),
            "support_gt": total_gt,
            "support_pred": total_pred,
        }
    )
    return pd.DataFrame(summary_rows)


def save_evaluation_outputs(
    matched_df: pd.DataFrame,
    unmatched_gt_df: pd.DataFrame,
    unmatched_pred_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    base_dir = Path(output_dir) if output_dir is not None else Path(config.get("paths", {}).get("output_dir", "outputs"))
    intermediate_dir = base_dir / "intermediate" / "evaluation"
    final_dir = base_dir / "final"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    matched_path = intermediate_dir / "matched_events.csv"
    unmatched_gt_path = intermediate_dir / "unmatched_gt_events.csv"
    unmatched_pred_path = intermediate_dir / "unmatched_pred_events.csv"
    summary_path = final_dir / "evaluation_summary.csv"

    matched_df.to_csv(matched_path, index=False)
    unmatched_gt_df.to_csv(unmatched_gt_path, index=False)
    unmatched_pred_df.to_csv(unmatched_pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return {
        "matched_events": matched_path,
        "unmatched_gt_events": unmatched_gt_path,
        "unmatched_pred_events": unmatched_pred_path,
        "evaluation_summary": summary_path,
    }


def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(missing_columns[0])


def _matched_columns() -> list[str]:
    return [
        "gt_event_id",
        "pred_event_id",
        "fish_id",
        "label",
        "gt_start_time",
        "gt_end_time",
        "pred_start_time",
        "pred_end_time",
        "overlap_duration",
        "union_duration",
        "temporal_iou",
    ]
