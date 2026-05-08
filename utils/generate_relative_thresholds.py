from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import yaml

from data_io import load_tracking_data, resolve_paths, standardize_tracking_columns
from features import compute_all_features
from preprocess import preprocess_tracking_data
from segmentation import segment_behavior


REQUIRED_COLUMNS = [
    "fish_id",
    "segment_id",
    "mean_v",
    "mean_d_win",
    "mean_E_move",
    "mean_S_pose",
    "mean_dv",
]
GMM_RANDOM_STATE = 0
MIN_GMM_SAMPLES = 6
MIN_COMPONENT_WEIGHT = 0.1
MIN_COMPONENT_SEPARATION = 1.0
GRID_POINTS = 512
LOGIT_EPSILON = 1e-6


def generate_relative_thresholds(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    segment_file: str | Path | None = None,
    output_file: str | Path | None = None,
    dataset_name: str | None = None,
    bootstrap_if_missing: bool = False,
    force_bootstrap: bool = False,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    config_file = Path(config_path) if config_path is not None else root / "configs" / "default.yaml"
    config = _load_base_config(config_file)
    paths = resolve_paths(config, root)

    resolved_segment_path = Path(segment_file) if segment_file is not None else _default_segment_path(paths, root)
    candidate_df, bootstrap_used = _load_or_build_candidate_segments(
        resolved_segment_path,
        config,
        root,
        output_dir=paths.get("output_dir", root / "outputs"),
        bootstrap_if_missing=bootstrap_if_missing,
        force_bootstrap=force_bootstrap,
    )
    _validate_candidate_segments(candidate_df)

    thresholds, diagnostics = _compute_relative_thresholds(candidate_df)

    resolved_output_path = Path(output_file) if output_file is not None else _default_output_path(config, root, dataset_name)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "dataset_name": dataset_name or resolved_output_path.stem,
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "method_version": "relative_threshold_v1",
            "source_config": str(config_file),
            "source_segment_file": str(resolved_segment_path),
            "bootstrap_used": bootstrap_used,
            "aggregation_policy": "per_fish_then_dataset_median",
        },
        "thresholds": thresholds,
        "diagnostics": diagnostics,
    }
    with resolved_output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    return {
        "output_file": resolved_output_path,
        "thresholds": thresholds,
        "diagnostics": diagnostics,
        "bootstrap_used": bootstrap_used,
    }


def _load_base_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式错误: {config_path}")
    return config


def _default_segment_path(paths: dict[str, Path], project_root: Path) -> Path:
    output_dir = paths.get("output_dir", project_root / "outputs")
    return output_dir / "intermediate" / "segmentation" / "candidate_segments.csv"


def _default_output_path(config: dict[str, Any], project_root: Path, dataset_name: str | None) -> Path:
    relative_threshold_file = config.get("adaptive", {}).get("relative_threshold_file")
    if relative_threshold_file:
        return project_root / relative_threshold_file
    name = dataset_name or "current_dataset"
    return project_root / "configs" / "adaptive" / f"{name}_thresholds.yaml"


def _load_or_build_candidate_segments(
    segment_path: Path,
    config: dict[str, Any],
    project_root: Path,
    output_dir: Path,
    bootstrap_if_missing: bool,
    force_bootstrap: bool,
) -> tuple[pd.DataFrame, bool]:
    if segment_path.exists() and not force_bootstrap:
        return pd.read_csv(segment_path), False
    if not bootstrap_if_missing and not force_bootstrap:
        raise FileNotFoundError(segment_path)

    paths = resolve_paths(config, project_root)
    track_path = paths.get("track_file")
    if track_path is None:
        raise KeyError("track_file")

    tracks_df = standardize_tracking_columns(load_tracking_data(track_path, config), config)
    processed_df = preprocess_tracking_data(tracks_df, config, output_dir=output_dir)
    feature_df = compute_all_features(processed_df, config, output_dir=output_dir)
    candidate_df = segment_behavior(feature_df, config, output_dir=output_dir)
    return candidate_df, True


def _validate_candidate_segments(candidate_df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in candidate_df.columns]
    if missing_columns:
        raise KeyError(missing_columns[0])


def _compute_relative_thresholds(candidate_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    per_fish_thresholds: list[dict[str, float]] = []
    per_fish_diagnostics: list[dict[str, dict[str, Any]]] = []
    for _, fish_df in candidate_df.groupby("fish_id", sort=True):
        thresholds, diagnostics = _estimate_thresholds_for_fish(fish_df)
        per_fish_thresholds.append(thresholds)
        per_fish_diagnostics.append(diagnostics)

    thresholds = {
        key: float(pd.Series([row[key] for row in per_fish_thresholds], dtype=float).median())
        for key in per_fish_thresholds[0]
    }
    thresholds = _enforce_threshold_ordering(thresholds)
    diagnostics = _aggregate_diagnostics(per_fish_diagnostics)
    return thresholds, diagnostics


def _estimate_thresholds_for_fish(fish_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    v_thresholds, v_diagnostics = _estimate_low_signal_pair(
        fish_df["mean_v"],
        pause_quantile=0.10,
        glide_quantile=0.25,
        strict_probability=0.9,
        loose_probability=0.6,
        transform_name="log1p",
        method_name="gmm",
    )
    d_thresholds, d_diagnostics = _estimate_low_signal_pair(
        fish_df["mean_d_win"],
        pause_quantile=0.30,
        glide_quantile=0.45,
        strict_probability=0.5,
        loose_probability=0.2,
        transform_name="log1p",
        method_name="gmm",
    )
    e_threshold, e_diagnostics = _estimate_single_signal_threshold(
        fish_df["mean_E_move"],
        fallback_quantile=0.20,
        target_probability=0.7,
        direction="low",
        transform_name="log1p",
        method_name="gmm",
    )
    s_thresholds, s_diagnostics = _estimate_high_signal_pair(
        fish_df["mean_S_pose"],
        pause_quantile=0.80,
        twist_quantile=0.25,
        strict_probability=0.7,
        loose_probability=0.3,
        transform_name="logit",
        method_name="gmm_logit",
    )
    positive_dv = fish_df.loc[fish_df["mean_dv"] > 0, "mean_dv"]
    if positive_dv.empty:
        positive_dv = fish_df["mean_dv"].clip(lower=0.0)
    dv_threshold, dv_diagnostics = _estimate_single_signal_threshold(
        positive_dv,
        fallback_quantile=0.80,
        target_probability=0.7,
        direction="high",
        transform_name="log1p",
        method_name="gmm",
    )

    thresholds = {
        "v_th_pause": v_thresholds["strict"],
        "v_th_glide": v_thresholds["loose"],
        "D_pause": d_thresholds["strict"],
        "D_glide": d_thresholds["loose"],
        "E_th_pause": e_threshold,
        "S_th_pause": s_thresholds["strict"],
        "S_th_twist": s_thresholds["loose"],
        "dv_th_glide": dv_threshold,
    }
    thresholds = _enforce_threshold_ordering(thresholds)
    diagnostics = {
        "v": v_diagnostics,
        "d_win": d_diagnostics,
        "E_move": e_diagnostics,
        "S_pose": s_diagnostics,
        "dv": dv_diagnostics,
    }
    return thresholds, diagnostics


def _estimate_low_signal_pair(
    series: pd.Series,
    pause_quantile: float,
    glide_quantile: float,
    strict_probability: float,
    loose_probability: float,
    transform_name: str,
    method_name: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    fallback = {
        "strict": float(series.quantile(pause_quantile)),
        "loose": float(series.quantile(glide_quantile)),
    }
    result = _estimate_gmm_pair(
        series,
        direction="low",
        strict_probability=strict_probability,
        loose_probability=loose_probability,
        transform_name=transform_name,
    )
    if result is None:
        return fallback, {"method": "quantile_fallback", "fallback_used": True}
    return result, {"method": method_name, "fallback_used": False}


def _estimate_high_signal_pair(
    series: pd.Series,
    pause_quantile: float,
    twist_quantile: float,
    strict_probability: float,
    loose_probability: float,
    transform_name: str,
    method_name: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    fallback = {
        "strict": float(series.quantile(pause_quantile)),
        "loose": float(series.quantile(twist_quantile)),
    }
    result = _estimate_gmm_pair(
        series,
        direction="high",
        strict_probability=strict_probability,
        loose_probability=loose_probability,
        transform_name=transform_name,
    )
    if result is None:
        return fallback, {"method": "quantile_fallback", "fallback_used": True}
    return result, {"method": method_name, "fallback_used": False}


def _estimate_single_signal_threshold(
    series: pd.Series,
    fallback_quantile: float,
    target_probability: float,
    direction: str,
    transform_name: str,
    method_name: str,
) -> tuple[float, dict[str, Any]]:
    fallback = float(series.quantile(fallback_quantile))
    result = _estimate_gmm_single(
        series,
        direction=direction,
        target_probability=target_probability,
        transform_name=transform_name,
    )
    if result is None:
        return fallback, {"method": "quantile_fallback", "fallback_used": True}
    return result, {"method": method_name, "fallback_used": False}


def _estimate_gmm_pair(
    series: pd.Series,
    direction: str,
    strict_probability: float,
    loose_probability: float,
    transform_name: str,
) -> dict[str, float] | None:
    transformed, inverse_transform = _transform_series(series, transform_name)
    model = _fit_two_component_gmm(transformed)
    if model is None:
        return None

    strict_threshold = _solve_probability_threshold(model, transformed, inverse_transform, direction, strict_probability)
    loose_threshold = _solve_probability_threshold(model, transformed, inverse_transform, direction, loose_probability)
    if strict_threshold is None or loose_threshold is None:
        return None
    return {"strict": strict_threshold, "loose": loose_threshold}


def _estimate_gmm_single(
    series: pd.Series,
    direction: str,
    target_probability: float,
    transform_name: str,
) -> float | None:
    transformed, inverse_transform = _transform_series(series, transform_name)
    model = _fit_two_component_gmm(transformed)
    if model is None:
        return None
    return _solve_probability_threshold(model, transformed, inverse_transform, direction, target_probability)


def _transform_series(series: pd.Series, transform_name: str) -> tuple[pd.Series, Any]:
    cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if transform_name == "log1p":
        clipped = cleaned.clip(lower=0.0)
        return np.log1p(clipped), np.expm1
    if transform_name == "logit":
        clipped = cleaned.clip(LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
        return np.log(clipped / (1.0 - clipped)), lambda value: 1.0 / (1.0 + np.exp(-value))
    return cleaned, lambda value: float(value)


def _fit_two_component_gmm(transformed: pd.Series) -> GaussianMixture | None:
    values = transformed.to_numpy(dtype=float)
    if len(values) < MIN_GMM_SAMPLES:
        return None

    model = GaussianMixture(n_components=2, random_state=GMM_RANDOM_STATE, reg_covar=1e-6)
    model.fit(values.reshape(-1, 1))
    if np.any(model.weights_ < MIN_COMPONENT_WEIGHT):
        return None

    variances = model.covariances_.reshape(-1)
    means = model.means_.reshape(-1)
    pooled_std = np.sqrt(max(float(np.mean(variances)), 1e-12))
    separation = abs(float(means[0] - means[1])) / pooled_std
    if separation < MIN_COMPONENT_SEPARATION:
        return None
    return model


def _solve_probability_threshold(
    model: GaussianMixture,
    transformed: pd.Series,
    inverse_transform,
    direction: str,
    target_probability: float,
) -> float | None:
    values = transformed.to_numpy(dtype=float)
    lower = float(np.quantile(values, 0.05))
    upper = float(np.quantile(values, 0.95))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return None

    component_index = int(np.argmin(model.means_.reshape(-1))) if direction == "low" else int(np.argmax(model.means_.reshape(-1)))
    grid = np.linspace(lower, upper, GRID_POINTS)
    probabilities = model.predict_proba(grid.reshape(-1, 1))[:, component_index]
    threshold_index = int(np.argmin(np.abs(probabilities - target_probability)))
    transformed_threshold = float(grid[threshold_index])
    threshold = float(inverse_transform(transformed_threshold))

    original_values = inverse_transform(values)
    min_allowed = float(np.quantile(original_values, 0.05))
    max_allowed = float(np.quantile(original_values, 0.95))
    if not np.isfinite(threshold) or threshold < min_allowed or threshold > max_allowed:
        return None
    return threshold


def _aggregate_diagnostics(per_fish_diagnostics: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for signal_name in per_fish_diagnostics[0]:
        methods = [row[signal_name]["method"] for row in per_fish_diagnostics]
        fallbacks = [bool(row[signal_name]["fallback_used"]) for row in per_fish_diagnostics]
        diagnostics[signal_name] = {
            "method": methods[0] if len(set(methods)) == 1 else "mixed",
            "fallback_used": any(fallbacks),
        }
    return diagnostics


def _enforce_threshold_ordering(thresholds: dict[str, float]) -> dict[str, float]:
    ordered = dict(thresholds)
    ordered["v_th_pause"], ordered["v_th_glide"] = sorted([ordered["v_th_pause"], ordered["v_th_glide"]])
    ordered["D_pause"], ordered["D_glide"] = sorted([ordered["D_pause"], ordered["D_glide"]])
    ordered["S_th_twist"], ordered["S_th_pause"] = sorted([ordered["S_th_twist"], ordered["S_th_pause"]])
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate relative threshold overlay from candidate segments.")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to config YAML file.")
    parser.add_argument("--project-root", dest="project_root", default=None, help="Project root directory.")
    parser.add_argument("--segment-file", dest="segment_file", default=None, help="Path to candidate segments CSV.")
    parser.add_argument("--output-file", dest="output_file", default=None, help="Path to adaptive threshold YAML.")
    parser.add_argument("--dataset-name", dest="dataset_name", default=None, help="Dataset name for metadata.")
    parser.add_argument("--bootstrap-if-missing", dest="bootstrap_if_missing", action="store_true")
    parser.add_argument("--force-bootstrap", dest="force_bootstrap", action="store_true")
    args = parser.parse_args()

    result = generate_relative_thresholds(
        config_path=args.config_path,
        project_root=args.project_root,
        segment_file=args.segment_file,
        output_file=args.output_file,
        dataset_name=args.dataset_name,
        bootstrap_if_missing=args.bootstrap_if_missing,
        force_bootstrap=args.force_bootstrap,
    )
    print(pd.Series(result["thresholds"]).to_string())


if __name__ == "__main__":
    main()
