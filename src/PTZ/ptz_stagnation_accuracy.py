from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


GT_LABEL = "IM"
PRED_LABEL = "stagnation"
UNKNOWN_LABEL = ""


def evaluate_ptz_stagnation(
    behavior_events_path: str | Path,
    feature_table_path: str | Path,
    output_dir: str | Path,
    iou_threshold: float = 0.1,
) -> dict[str, Any]:
    feature_df = pd.read_csv(feature_table_path, low_memory=False)
    behavior_df = pd.read_csv(behavior_events_path, low_memory=False)

    frame_predictions = expand_stagnation_predictions(feature_df, behavior_df)
    frame_predictions["behavior_label_clean"] = _clean_behavior_label_series(frame_predictions["behavior_label"])
    frame_predictions["label_status"] = frame_predictions["behavior_label_clean"].apply(_label_status)
    frame_predictions["evaluable_frame"] = frame_predictions["label_status"].ne("unknown")
    frame_predictions["gt_stagnation"] = frame_predictions["behavior_label_clean"].eq(GT_LABEL)
    frame_predictions["outcome"] = frame_predictions.apply(_frame_outcome, axis=1)

    frame_metrics = summarize_frame_metrics(frame_predictions)
    frame_metrics_by_fish = summarize_frame_metrics(frame_predictions, group_cols=["fish_id"])

    gt_im_events = extract_gt_im_events(feature_df)
    pred_stagnation_events = normalize_pred_stagnation_events(behavior_df)
    event_matches, unmatched_pred_events, unmatched_gt_events = match_events_by_iou(
        pred_stagnation_events,
        gt_im_events,
        iou_threshold=iou_threshold,
    )
    prediction_review = categorize_pred_stagnation_events(pred_stagnation_events, feature_df, event_matches)
    event_metrics = summarize_event_metrics(gt_im_events, prediction_review, event_matches)

    output_paths = save_ptz_stagnation_accuracy_outputs(
        output_dir=output_dir,
        frame_predictions=frame_predictions,
        frame_metrics=frame_metrics,
        frame_metrics_by_fish=frame_metrics_by_fish,
        gt_im_events=gt_im_events,
        pred_stagnation_events=pred_stagnation_events,
        event_matches=event_matches,
        unmatched_pred_events=unmatched_pred_events,
        unmatched_gt_events=unmatched_gt_events,
        prediction_review=prediction_review,
        event_metrics=event_metrics,
        iou_threshold=iou_threshold,
    )

    return {
        "frame_predictions": frame_predictions,
        "frame_metrics": frame_metrics,
        "frame_metrics_by_fish": frame_metrics_by_fish,
        "gt_im_events": gt_im_events,
        "pred_stagnation_events": pred_stagnation_events,
        "event_matches": event_matches,
        "unmatched_pred_events": unmatched_pred_events,
        "unmatched_gt_events": unmatched_gt_events,
        "prediction_review": prediction_review,
        "event_metrics": event_metrics,
        "output_paths": output_paths,
    }


def expand_stagnation_predictions(frame_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame_df, ["fish_id", "frame"])
    _require_columns(event_df, ["fish_id", "start_frame", "end_frame", "behavior_label"])

    expanded = frame_df.copy()
    expanded["fish_id"] = expanded["fish_id"].astype(str)
    expanded["frame"] = pd.to_numeric(expanded["frame"], errors="raise").astype(int)
    expanded["pred_stagnation"] = False

    if event_df.empty:
        return expanded.sort_values(["fish_id", "frame"]).reset_index(drop=True)

    pred_events = event_df[event_df["behavior_label"].astype(str).eq(PRED_LABEL)].copy()
    for row in pred_events.itertuples(index=False):
        fish_id = str(row.fish_id)
        start_frame = int(row.start_frame)
        end_frame = int(row.end_frame)
        mask = (
            expanded["fish_id"].eq(fish_id)
            & expanded["frame"].ge(start_frame)
            & expanded["frame"].le(end_frame)
        )
        expanded.loc[mask, "pred_stagnation"] = True

    return expanded.sort_values(["fish_id", "frame"]).reset_index(drop=True)


def summarize_frame_metrics(
    frame_eval: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    _require_columns(frame_eval, ["gt_stagnation", "pred_stagnation"])

    if group_cols is None:
        return pd.DataFrame([_summarize_frame_group(frame_eval, {"scope": "overall"})])

    rows: list[dict[str, Any]] = []
    grouped = frame_eval.copy()
    for column in group_cols:
        grouped[column] = grouped[column].astype(str)
    for key, group_df in grouped.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        labels = {column: value for column, value in zip(group_cols, key)}
        rows.append(_summarize_frame_group(group_df, labels))
    return pd.DataFrame(rows)


def extract_gt_im_events(frame_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame_df, ["fish_id", "frame", "timestamp", "behavior_label"])

    normalized = frame_df.copy()
    normalized["fish_id"] = normalized["fish_id"].astype(str)
    normalized["frame"] = pd.to_numeric(normalized["frame"], errors="raise").astype(int)
    normalized["timestamp"] = pd.to_numeric(normalized["timestamp"], errors="raise")
    im_frames = normalized[normalized["behavior_label"].astype(str).eq(GT_LABEL)].copy()
    if im_frames.empty:
        return pd.DataFrame(columns=_gt_event_columns())

    im_frames = im_frames.sort_values(["fish_id", "frame"]).reset_index(drop=True)
    events: list[dict[str, Any]] = []
    event_id = 0

    for fish_id, fish_df in im_frames.groupby("fish_id", sort=True):
        current_rows: list[Any] = []
        previous_frame: int | None = None
        for row in fish_df.itertuples(index=False):
            frame = int(row.frame)
            if previous_frame is None or frame == previous_frame + 1:
                current_rows.append(row)
            else:
                event_id += 1
                events.append(_build_gt_event(event_id, str(fish_id), current_rows))
                current_rows = [row]
            previous_frame = frame
        if current_rows:
            event_id += 1
            events.append(_build_gt_event(event_id, str(fish_id), current_rows))

    return pd.DataFrame(events, columns=_gt_event_columns())


def normalize_pred_stagnation_events(event_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(event_df, ["fish_id", "start_frame", "end_frame", "start_time", "end_time", "behavior_label"])

    pred = event_df[event_df["behavior_label"].astype(str).eq(PRED_LABEL)].copy().reset_index(drop=True)
    if pred.empty:
        return pd.DataFrame(columns=_pred_event_columns())

    pred["fish_id"] = pred["fish_id"].astype(str)
    pred["start_frame"] = pd.to_numeric(pred["start_frame"], errors="raise").astype(int)
    pred["end_frame"] = pd.to_numeric(pred["end_frame"], errors="raise").astype(int)
    pred["start_time"] = pd.to_numeric(pred["start_time"], errors="raise")
    pred["end_time"] = pd.to_numeric(pred["end_time"], errors="raise")
    pred["duration"] = (
        pd.to_numeric(pred["duration"], errors="raise")
        if "duration" in pred.columns
        else pred["end_time"] - pred["start_time"]
    )

    if "behavior_id" in pred.columns:
        pred["event_id"] = pd.to_numeric(pred["behavior_id"], errors="raise").astype(int)
    elif "event_id" in pred.columns:
        pred["event_id"] = pd.to_numeric(pred["event_id"], errors="raise").astype(int)
    else:
        pred["event_id"] = range(1, len(pred) + 1)

    return pred[_pred_event_columns()].sort_values(["fish_id", "start_frame", "end_frame"]).reset_index(drop=True)


def categorize_pred_stagnation_events(
    pred_events: pd.DataFrame,
    feature_df: pd.DataFrame,
    event_matches: pd.DataFrame,
) -> pd.DataFrame:
    if pred_events.empty:
        return pd.DataFrame(columns=_prediction_review_columns())

    _require_columns(feature_df, ["fish_id", "frame", "behavior_label"])
    features = feature_df.copy()
    features["fish_id"] = features["fish_id"].astype(str)
    features["frame"] = pd.to_numeric(features["frame"], errors="raise").astype(int)
    features["behavior_label_clean"] = _clean_behavior_label_series(features["behavior_label"])

    matched_pred_ids = set()
    if not event_matches.empty and "pred_event_id" in event_matches.columns:
        matched_pred_ids = set(pd.to_numeric(event_matches["pred_event_id"], errors="coerce").dropna().astype(int).tolist())

    rows: list[dict[str, Any]] = []
    for event in pred_events.itertuples(index=False):
        event_id = int(event.event_id)
        window = features[
            features["fish_id"].eq(str(event.fish_id))
            & features["frame"].ge(int(event.start_frame))
            & features["frame"].le(int(event.end_frame))
        ]
        labels = window["behavior_label_clean"] if not window.empty else pd.Series(dtype=str)
        im_frames = int(labels.eq(GT_LABEL).sum())
        explicit_non_im_frames = int((labels.ne(UNKNOWN_LABEL) & labels.ne(GT_LABEL)).sum())
        unlabeled_frames = int(labels.eq(UNKNOWN_LABEL).sum())
        total_frames = int(len(window))
        label_counts = _format_label_counts(labels)

        if event_id in matched_pred_ids:
            review_category = "matched_im"
            evaluable_for_metrics = True
        elif explicit_non_im_frames > 0:
            review_category = "conflict_non_im"
            evaluable_for_metrics = True
        else:
            review_category = "unlabeled_candidate"
            evaluable_for_metrics = False

        rows.append(
            {
                "event_id": event_id,
                "fish_id": str(event.fish_id),
                "start_frame": int(event.start_frame),
                "end_frame": int(event.end_frame),
                "start_time": float(event.start_time),
                "end_time": float(event.end_time),
                "duration": float(event.duration),
                "behavior_label": str(event.behavior_label),
                "review_category": review_category,
                "evaluable_for_metrics": bool(evaluable_for_metrics),
                "matched_im": bool(event_id in matched_pred_ids),
                "im_frames": im_frames,
                "explicit_non_im_frames": explicit_non_im_frames,
                "unlabeled_frames": unlabeled_frames,
                "total_frames": total_frames,
                "label_counts": label_counts,
            }
        )

    return pd.DataFrame(rows, columns=_prediction_review_columns())


def match_events_by_iou(
    pred_events: pd.DataFrame,
    gt_events: pd.DataFrame,
    iou_threshold: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pred_events.copy()
    gt = gt_events.copy()
    if not pred.empty:
        pred["fish_id"] = pred["fish_id"].astype(str)
    if not gt.empty:
        gt["fish_id"] = gt["fish_id"].astype(str)

    candidates: list[dict[str, Any]] = []
    for pred_row in pred.itertuples(index=False):
        for gt_row in gt.itertuples(index=False):
            if str(pred_row.fish_id) != str(gt_row.fish_id):
                continue
            temporal_iou, overlap_duration, union_duration = _temporal_iou(pred_row, gt_row)
            frame_iou, overlap_frames, union_frames = _frame_iou(pred_row, gt_row)
            score_iou = temporal_iou if union_duration > 0 else frame_iou
            if score_iou < iou_threshold:
                continue
            candidates.append(
                {
                    "pred_event_id": int(pred_row.event_id),
                    "gt_event_id": int(gt_row.gt_event_id),
                    "fish_id": str(pred_row.fish_id),
                    "pred_start_frame": int(pred_row.start_frame),
                    "pred_end_frame": int(pred_row.end_frame),
                    "gt_start_frame": int(gt_row.start_frame),
                    "gt_end_frame": int(gt_row.end_frame),
                    "pred_start_time": float(pred_row.start_time),
                    "pred_end_time": float(pred_row.end_time),
                    "gt_start_time": float(gt_row.start_time),
                    "gt_end_time": float(gt_row.end_time),
                    "overlap_duration": float(overlap_duration),
                    "union_duration": float(union_duration),
                    "temporal_iou": float(temporal_iou),
                    "overlap_frames": int(overlap_frames),
                    "union_frames": int(union_frames),
                    "frame_iou": float(frame_iou),
                    "match_iou": float(score_iou),
                }
            )

    if not candidates:
        matches = pd.DataFrame(columns=_match_columns())
        return matches, pred.reset_index(drop=True), gt.reset_index(drop=True)

    candidate_df = pd.DataFrame(candidates)
    candidate_df = candidate_df.sort_values(
        ["match_iou", "overlap_duration", "overlap_frames", "gt_event_id", "pred_event_id"],
        ascending=[False, False, False, True, True],
    )

    matched_gt_ids: set[int] = set()
    matched_pred_ids: set[int] = set()
    matched_rows: list[dict[str, Any]] = []
    for row in candidate_df.itertuples(index=False):
        gt_id = int(row.gt_event_id)
        pred_id = int(row.pred_event_id)
        if gt_id in matched_gt_ids or pred_id in matched_pred_ids:
            continue
        matched_gt_ids.add(gt_id)
        matched_pred_ids.add(pred_id)
        matched_rows.append({column: getattr(row, column) for column in _match_columns()})

    matches = pd.DataFrame(matched_rows, columns=_match_columns())
    unmatched_pred = pred[~pred["event_id"].isin(matched_pred_ids)].reset_index(drop=True)
    unmatched_gt = gt[~gt["gt_event_id"].isin(matched_gt_ids)].reset_index(drop=True)
    return matches, unmatched_pred, unmatched_gt


def summarize_event_metrics(
    gt_events: pd.DataFrame,
    pred_events: pd.DataFrame,
    event_matches: pd.DataFrame,
) -> pd.DataFrame:
    tp = int(len(event_matches))
    if "evaluable_for_metrics" in pred_events.columns:
        evaluable_pred = pred_events[pred_events["evaluable_for_metrics"].astype(bool)].copy()
        ignored_unlabeled = int((pred_events["review_category"] == "unlabeled_candidate").sum())
        conflict_non_im = int((pred_events["review_category"] == "conflict_non_im").sum())
    else:
        evaluable_pred = pred_events.copy()
        ignored_unlabeled = 0
        conflict_non_im = max(0, int(len(pred_events) - tp))
    fp = int(len(evaluable_pred) - tp)
    fn = int(len(gt_events) - tp)
    precision = tp / len(evaluable_pred) if len(evaluable_pred) > 0 else 0.0
    recall = tp / len(gt_events) if len(gt_events) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "label": PRED_LABEL,
                "gt_label": GT_LABEL,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support_gt_events": int(len(gt_events)),
                "support_pred_events": int(len(pred_events)),
                "support_evaluable_pred_events": int(len(evaluable_pred)),
                "matched_events": tp,
                "conflict_non_im_events": conflict_non_im,
                "ignored_unlabeled_candidates": ignored_unlabeled,
                "gt_total_duration": float(gt_events["duration"].sum()) if "duration" in gt_events else 0.0,
                "pred_total_duration": float(pred_events["duration"].sum()) if "duration" in pred_events else 0.0,
                "evaluable_pred_total_duration": float(evaluable_pred["duration"].sum()) if "duration" in evaluable_pred else 0.0,
            }
        ]
    )


def save_ptz_stagnation_accuracy_outputs(
    output_dir: str | Path,
    frame_predictions: pd.DataFrame,
    frame_metrics: pd.DataFrame,
    frame_metrics_by_fish: pd.DataFrame,
    gt_im_events: pd.DataFrame,
    pred_stagnation_events: pd.DataFrame,
    event_matches: pd.DataFrame,
    unmatched_pred_events: pd.DataFrame,
    unmatched_gt_events: pd.DataFrame,
    prediction_review: pd.DataFrame,
    event_metrics: pd.DataFrame,
    iou_threshold: float,
) -> dict[str, Path]:
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "frame_metrics": save_dir / "frame_metrics.csv",
        "frame_metrics_by_fish": save_dir / "frame_metrics_by_fish.csv",
        "frame_predictions": save_dir / "frame_predictions.csv",
        "gt_im_events": save_dir / "gt_im_events.csv",
        "pred_stagnation_events": save_dir / "pred_stagnation_events.csv",
        "event_matches": save_dir / "event_matches.csv",
        "unmatched_pred_events": save_dir / "unmatched_pred_events.csv",
        "unmatched_gt_events": save_dir / "unmatched_gt_events.csv",
        "prediction_review": save_dir / "prediction_review.csv",
        "event_metrics": save_dir / "event_metrics.csv",
        "summary": save_dir / "summary.json",
    }

    frame_metrics.to_csv(paths["frame_metrics"], index=False)
    frame_metrics_by_fish.to_csv(paths["frame_metrics_by_fish"], index=False)
    frame_predictions.to_csv(paths["frame_predictions"], index=False)
    gt_im_events.to_csv(paths["gt_im_events"], index=False)
    pred_stagnation_events.to_csv(paths["pred_stagnation_events"], index=False)
    event_matches.to_csv(paths["event_matches"], index=False)
    unmatched_pred_events.to_csv(paths["unmatched_pred_events"], index=False)
    unmatched_gt_events.to_csv(paths["unmatched_gt_events"], index=False)
    prediction_review.to_csv(paths["prediction_review"], index=False)
    event_metrics.to_csv(paths["event_metrics"], index=False)

    summary = {
        "gt_label": GT_LABEL,
        "pred_label": PRED_LABEL,
        "iou_threshold": float(iou_threshold),
        "frame_metrics": _records_for_json(frame_metrics),
        "frame_metrics_by_fish": _records_for_json(frame_metrics_by_fish),
        "event_metrics": _records_for_json(event_metrics),
        "prediction_review_counts": _records_for_json(
            prediction_review["review_category"].value_counts().rename_axis("review_category").reset_index(name="count")
        ) if not prediction_review.empty else [],
        "n_gt_im_events": int(len(gt_im_events)),
        "n_pred_stagnation_events": int(len(pred_stagnation_events)),
        "n_event_matches": int(len(event_matches)),
        "n_unmatched_gt_events": int(len(unmatched_gt_events)),
        "n_unmatched_pred_events": int(len(unmatched_pred_events)),
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return paths


def _summarize_frame_group(frame_eval: pd.DataFrame, labels: dict[str, Any]) -> dict[str, Any]:
    if "evaluable_frame" in frame_eval.columns:
        unknown_frames = int((~frame_eval["evaluable_frame"].astype(bool)).sum())
        metric_df = frame_eval[frame_eval["evaluable_frame"].astype(bool)].copy()
    else:
        unknown_frames = 0
        metric_df = frame_eval.copy()

    gt = metric_df["gt_stagnation"].astype(bool)
    pred = metric_df["pred_stagnation"].astype(bool)

    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    tn = int((~gt & ~pred).sum())
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0

    row = dict(labels)
    row.update(
        {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "specificity": float(specificity),
            "support_gt_frames": int(gt.sum()),
            "support_pred_frames": int(pred.sum()),
            "total_frames": int(total),
            "unknown_frames": unknown_frames,
        }
    )
    return row


def _build_gt_event(event_id: int, fish_id: str, rows: list[Any]) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    start_time = float(first.timestamp)
    end_time = float(last.timestamp)
    return {
        "gt_event_id": int(event_id),
        "fish_id": fish_id,
        "start_frame": int(first.frame),
        "end_frame": int(last.frame),
        "start_time": start_time,
        "end_time": end_time,
        "duration": float(end_time - start_time),
        "n_frames": int(len(rows)),
        "label": GT_LABEL,
    }


def _temporal_iou(pred_row: Any, gt_row: Any) -> tuple[float, float, float]:
    overlap = min(float(pred_row.end_time), float(gt_row.end_time)) - max(float(pred_row.start_time), float(gt_row.start_time))
    overlap = max(0.0, overlap)
    union = max(float(pred_row.end_time), float(gt_row.end_time)) - min(float(pred_row.start_time), float(gt_row.start_time))
    iou = overlap / union if union > 0 else 0.0
    return float(iou), float(overlap), float(union)


def _frame_iou(pred_row: Any, gt_row: Any) -> tuple[float, int, int]:
    overlap_start = max(int(pred_row.start_frame), int(gt_row.start_frame))
    overlap_end = min(int(pred_row.end_frame), int(gt_row.end_frame))
    overlap = max(0, overlap_end - overlap_start + 1)
    union_start = min(int(pred_row.start_frame), int(gt_row.start_frame))
    union_end = max(int(pred_row.end_frame), int(gt_row.end_frame))
    union = max(0, union_end - union_start + 1)
    iou = overlap / union if union > 0 else 0.0
    return float(iou), int(overlap), int(union)


def _frame_outcome(row: pd.Series) -> str:
    if "evaluable_frame" in row and not bool(row["evaluable_frame"]):
        return "UNKNOWN"
    if bool(row["gt_stagnation"]) and bool(row["pred_stagnation"]):
        return "TP"
    if not bool(row["gt_stagnation"]) and bool(row["pred_stagnation"]):
        return "FP"
    if bool(row["gt_stagnation"]) and not bool(row["pred_stagnation"]):
        return "FN"
    return "TN"


def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(missing_columns[0])


def _records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def _clean_behavior_label_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.replace({"NA": "", "NaN": "", "nan": "", "None": "", "<NA>": ""})


def _label_status(label: str) -> str:
    if label == UNKNOWN_LABEL:
        return "unknown"
    if label == GT_LABEL:
        return "gt_im"
    return "explicit_non_im"


def _format_label_counts(labels: pd.Series) -> str:
    if labels.empty:
        return ""
    display = labels.replace("", "unlabeled")
    return "; ".join(f"{label}:{int(count)}" for label, count in display.value_counts(sort=False).items())


def _gt_event_columns() -> list[str]:
    return ["gt_event_id", "fish_id", "start_frame", "end_frame", "start_time", "end_time", "duration", "n_frames", "label"]


def _pred_event_columns() -> list[str]:
    return ["event_id", "fish_id", "start_frame", "end_frame", "start_time", "end_time", "duration", "behavior_label"]


def _prediction_review_columns() -> list[str]:
    return [
        "event_id",
        "fish_id",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "duration",
        "behavior_label",
        "review_category",
        "evaluable_for_metrics",
        "matched_im",
        "im_frames",
        "explicit_non_im_frames",
        "unlabeled_frames",
        "total_frames",
        "label_counts",
    ]


def _match_columns() -> list[str]:
    return [
        "pred_event_id",
        "gt_event_id",
        "fish_id",
        "pred_start_frame",
        "pred_end_frame",
        "gt_start_frame",
        "gt_end_frame",
        "pred_start_time",
        "pred_end_time",
        "gt_start_time",
        "gt_end_time",
        "overlap_duration",
        "union_duration",
        "temporal_iou",
        "overlap_frames",
        "union_frames",
        "frame_iou",
        "match_iou",
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PTZ stagnation predictions against PTZ IM labels.")
    parser.add_argument("--behavior-events", required=True, help="Path to final behavior_events.csv.")
    parser.add_argument("--feature-table", required=True, help="Path to intermediate feature_table.csv with behavior_label.")
    parser.add_argument("--output-dir", required=True, help="Directory for PTZ stagnation accuracy outputs.")
    parser.add_argument("--iou-threshold", type=float, default=0.1, help="Temporal IoU threshold for event matching.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate_ptz_stagnation(
        behavior_events_path=args.behavior_events,
        feature_table_path=args.feature_table,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()
