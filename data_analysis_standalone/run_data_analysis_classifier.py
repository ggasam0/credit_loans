from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from workflow_common import (
    ISSUE_YEAR_COL,
    SHARED_ROW_ID_COL,
    TARGET_COL,
    build_validation_to_test_reject_curve,
    compute_metrics,
    evaluate_fixed_reject_rate,
    materialize_or_load_stratified_split,
    optimize_binary_threshold_business,
    save_json,
    stratified_train_val_test_split,
    threshold_by_target_reject_rate,
)


MISSING_LABEL = "__MISSING__"
OTHER_LABEL = "__OTHER__"


def _to_serializable_float(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _load_feature_columns(df: pd.DataFrame, feature_selection_json: Path | None) -> tuple[list[str], list[str]]:
    excluded = {TARGET_COL, ISSUE_YEAR_COL, SHARED_ROW_ID_COL}
    if feature_selection_json is not None and feature_selection_json.exists():
        payload = json.loads(feature_selection_json.read_text(encoding="utf-8"))
        numeric_cols = [c for c in payload.get("selected_numeric_columns", []) if c in df.columns and c not in excluded]
        categorical_cols = [
            c for c in payload.get("selected_categorical_columns", []) if c in df.columns and c not in excluded
        ]
        return numeric_cols, categorical_cols

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _build_numeric_profile(
    series: pd.Series,
    y: pd.Series,
    *,
    max_bins: int,
    min_bin_count: int,
    smoothing: float,
    global_rate: float,
) -> dict[str, Any] | None:
    s_num = pd.to_numeric(series, errors="coerce")
    non_missing = s_num[s_num.notna()]
    if non_missing.empty:
        return None

    n_unique = int(non_missing.nunique(dropna=True))
    n_bins = int(min(max_bins, n_unique))
    if n_bins <= 1:
        edges = np.array([-np.inf, np.inf], dtype=float)
    else:
        try:
            _, q_edges = pd.qcut(non_missing, q=n_bins, retbins=True, duplicates="drop")
            edges = np.asarray(q_edges, dtype=float)
            edges[0] = -np.inf
            edges[-1] = np.inf
        except Exception:
            edges = np.array([-np.inf, np.inf], dtype=float)

    bucket = pd.cut(s_num, bins=edges, include_lowest=True)
    bucket_text = bucket.astype(str)
    bucket_text = bucket_text.where(s_num.notna(), MISSING_LABEL)

    tmp = pd.DataFrame({"bucket": bucket_text, "target": y.astype(int)})
    stats = tmp.groupby("bucket", dropna=False)["target"].agg(["count", "sum"]).reset_index()
    stats = stats.rename(columns={"sum": "bad"})
    stats["smoothed_bad_rate"] = (stats["bad"] + smoothing * global_rate) / (stats["count"] + smoothing)

    for i, row in stats.iterrows():
        if int(row["count"]) < min_bin_count:
            stats.at[i, "bucket"] = OTHER_LABEL
    stats = stats.groupby("bucket", dropna=False)[["count", "bad"]].sum().reset_index()
    stats["smoothed_bad_rate"] = (stats["bad"] + smoothing * global_rate) / (stats["count"] + smoothing)
    stats = stats.sort_values("smoothed_bad_rate", ascending=False).reset_index(drop=True)

    bucket_risk = {str(r["bucket"]): _to_serializable_float(r["smoothed_bad_rate"]) for _, r in stats.iterrows()}
    default_risk = _to_serializable_float(global_rate)
    weight = float(max(stats["smoothed_bad_rate"]) - min(stats["smoothed_bad_rate"])) if len(stats) > 1 else 0.0
    weight = max(weight, 1e-6)

    profile = {
        "feature_type": "numeric",
        "edges": [float(e) for e in edges.tolist()],
        "bucket_risk": bucket_risk,
        "default_risk": default_risk,
        "weight": weight,
        "stats": [
            {
                "bucket": str(r["bucket"]),
                "count": int(r["count"]),
                "bad": int(r["bad"]),
                "smoothed_bad_rate": _to_serializable_float(r["smoothed_bad_rate"]),
            }
            for _, r in stats.iterrows()
        ],
    }
    return profile


def _build_categorical_profile(
    series: pd.Series,
    y: pd.Series,
    *,
    min_category_count: int,
    smoothing: float,
    global_rate: float,
) -> dict[str, Any] | None:
    s = series.astype(str).fillna(MISSING_LABEL).replace({"nan": MISSING_LABEL, "": MISSING_LABEL})
    tmp = pd.DataFrame({"cat": s, "target": y.astype(int)})
    stats = tmp.groupby("cat", dropna=False)["target"].agg(["count", "sum"]).reset_index()
    stats = stats.rename(columns={"sum": "bad"})

    rare = set(stats.loc[stats["count"] < min_category_count, "cat"].astype(str).tolist())
    s_group = s.map(lambda v: OTHER_LABEL if str(v) in rare else str(v))

    tmp2 = pd.DataFrame({"cat": s_group, "target": y.astype(int)})
    stats2 = tmp2.groupby("cat", dropna=False)["target"].agg(["count", "sum"]).reset_index()
    stats2 = stats2.rename(columns={"sum": "bad"})
    stats2["smoothed_bad_rate"] = (stats2["bad"] + smoothing * global_rate) / (stats2["count"] + smoothing)
    stats2 = stats2.sort_values("smoothed_bad_rate", ascending=False).reset_index(drop=True)

    category_risk = {str(r["cat"]): _to_serializable_float(r["smoothed_bad_rate"]) for _, r in stats2.iterrows()}
    keep_categories = [c for c in category_risk.keys() if c != OTHER_LABEL]
    default_risk = category_risk.get(OTHER_LABEL, _to_serializable_float(global_rate))
    weight = float(max(stats2["smoothed_bad_rate"]) - min(stats2["smoothed_bad_rate"])) if len(stats2) > 1 else 0.0
    weight = max(weight, 1e-6)

    profile = {
        "feature_type": "categorical",
        "keep_categories": keep_categories,
        "category_risk": category_risk,
        "default_risk": default_risk,
        "weight": weight,
        "stats": [
            {
                "bucket": str(r["cat"]),
                "count": int(r["count"]),
                "bad": int(r["bad"]),
                "smoothed_bad_rate": _to_serializable_float(r["smoothed_bad_rate"]),
            }
            for _, r in stats2.iterrows()
        ],
    }
    return profile


def _build_profiles(
    train_fit_df: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    max_bins: int,
    min_bin_count: int,
    min_category_count: int,
    smoothing: float,
) -> dict[str, dict[str, Any]]:
    y = train_fit_df[TARGET_COL].astype(int)
    global_rate = float(y.mean())
    profiles: dict[str, dict[str, Any]] = {}

    for col in numeric_cols:
        if col not in train_fit_df.columns:
            continue
        profile = _build_numeric_profile(
            train_fit_df[col],
            y,
            max_bins=max_bins,
            min_bin_count=min_bin_count,
            smoothing=smoothing,
            global_rate=global_rate,
        )
        if profile is not None:
            profiles[col] = profile

    for col in categorical_cols:
        if col not in train_fit_df.columns:
            continue
        profile = _build_categorical_profile(
            train_fit_df[col],
            y,
            min_category_count=min_category_count,
            smoothing=smoothing,
            global_rate=global_rate,
        )
        if profile is not None:
            profiles[col] = profile

    return profiles


def _risk_from_profile(series: pd.Series, profile: dict[str, Any]) -> pd.Series:
    if profile["feature_type"] == "numeric":
        s_num = pd.to_numeric(series, errors="coerce")
        edges = np.asarray(profile["edges"], dtype=float)
        bucket = pd.cut(s_num, bins=edges, include_lowest=True)
        bucket_text = bucket.astype(str).where(s_num.notna(), MISSING_LABEL)
        risk_map = profile["bucket_risk"]
        default_risk = float(profile["default_risk"])
        return bucket_text.map(lambda k: float(risk_map.get(str(k), risk_map.get(OTHER_LABEL, default_risk)))).astype(float)

    s = series.astype(str).fillna(MISSING_LABEL).replace({"nan": MISSING_LABEL, "": MISSING_LABEL})
    keep = set(profile.get("keep_categories", []))
    grouped = s.map(lambda v: str(v) if str(v) in keep else OTHER_LABEL)
    risk_map = profile["category_risk"]
    default_risk = float(profile["default_risk"])
    return grouped.map(lambda k: float(risk_map.get(str(k), risk_map.get(OTHER_LABEL, default_risk)))).astype(float)


def _score_frame(df: pd.DataFrame, profiles: dict[str, dict[str, Any]]) -> tuple[np.ndarray, pd.DataFrame]:
    contrib = pd.DataFrame(index=df.index)
    weighted_sum = np.zeros(len(df), dtype=float)
    weight_sum = np.zeros(len(df), dtype=float)

    for col, profile in profiles.items():
        if col not in df.columns:
            continue
        risk = _risk_from_profile(df[col], profile).to_numpy(dtype=float)
        w = float(profile.get("weight", 1.0))
        weighted_sum += risk * w
        weight_sum += w
        contrib[f"risk_{col}"] = risk

    safe_weight = np.where(weight_sum > 0, weight_sum, 1.0)
    score = weighted_sum / safe_weight
    return score.astype(float), contrib


def _compute_business_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tp = float(((y == 1) & (p == 1)).sum())
    fp = float(((y == 0) & (p == 1)).sum())
    tn = float(((y == 0) & (p == 0)).sum())
    fn = float(((y == 1) & (p == 0)).sum())
    n = float(max(len(y), 1))
    reject = tp + fp
    approve = tn + fn
    precision = tp / reject if reject > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "reject_rate": reject / n,
        "approve_rate": approve / n,
        "default_capture_recall": recall,
        "bad_rate_in_reject_precision": precision,
        "bad_rate_in_approve": fn / approve if approve > 0 else 0.0,
    }


def _optimize_threshold_high_precision(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    threshold_objective: str,
    precision_floor: float,
    min_reject_rate: float,
    max_reject_rate: float,
    min_recall: float,
    target_reject_rate: float,
) -> dict[str, Any]:
    return optimize_binary_threshold_business(
        y_true,
        score,
        threshold_objective=threshold_objective,
        precision_floor=precision_floor,
        recall_floor=min_recall,
        min_reject_rate=min_reject_rate,
        max_reject_rate=max_reject_rate,
        target_reject_rate=target_reject_rate,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_dir / "analysis"
    model_dir = output_dir / "model"
    data_dir = output_dir / "data"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    required = [TARGET_COL, ISSUE_YEAR_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[ISSUE_YEAR_COL] = pd.to_numeric(df[ISSUE_YEAR_COL], errors="coerce")
    df = df[df[TARGET_COL].notna() & df[ISSUE_YEAR_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    shared_split_dir = str(args.shared_split_dir).strip()
    if shared_split_dir:
        train_fit_df, val_df, test_df, split_summary = materialize_or_load_stratified_split(
            df,
            split_dir=Path(shared_split_dir),
            target_col=TARGET_COL,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            force_rebuild=bool(args.force_rebuild_shared_split),
        )
    else:
        train_fit_df, val_df, test_df, split_summary = stratified_train_val_test_split(
            df,
            target_col=TARGET_COL,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )
    train_df = pd.concat([train_fit_df, val_df], axis=0).reset_index(drop=True)

    df.to_csv(data_dir / "analysis_processed.csv", index=False)
    train_fit_df.to_csv(data_dir / "train_fit.csv", index=False)
    val_df.to_csv(data_dir / "validation.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    save_json(
        data_dir / "split_summary.json",
        {
            "input_csv": str(input_csv),
            "split_mode": "stratified",
            "split_summary": split_summary,
        },
    )

    feature_selection_json = Path(args.feature_selection_json) if str(args.feature_selection_json).strip() else None
    numeric_cols, categorical_cols = _load_feature_columns(df, feature_selection_json)
    selected_cols = [*numeric_cols, *categorical_cols]
    if not selected_cols:
        raise ValueError("No selected features found for analysis-based classifier.")

    profiles = _build_profiles(
        train_fit_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        max_bins=args.max_bins,
        min_bin_count=args.min_bin_count,
        min_category_count=args.min_category_count,
        smoothing=args.smoothing,
    )
    if not profiles:
        raise ValueError("No usable feature profiles were built.")

    val_score, _ = _score_frame(val_df, profiles)
    test_score, test_contrib = _score_frame(test_df, profiles)

    if args.threshold_policy == "fixed_reject_rate":
        fixed_threshold = threshold_by_target_reject_rate(
            val_score,
            target_reject_rate=args.target_reject_rate,
        )
        threshold = float(fixed_threshold["threshold"])
        threshold_info = {
            "metric": "fixed_reject_rate_quantile",
            "best_threshold": threshold,
            "best_score": float("nan"),
            "precision_floor": None,
            "recall_floor": None,
            "min_reject_rate": None,
            "max_reject_rate": None,
            "target_reject_rate": float(args.target_reject_rate),
            "constraint_satisfied": True,
            "fallback_metric": None,
            "validation_reject_rate": float(fixed_threshold["actual_reject_rate"]),
        }
    else:
        threshold_info = _optimize_threshold_high_precision(
            val_df[TARGET_COL].astype(int).to_numpy(),
            val_score,
            threshold_objective=args.threshold_objective,
            precision_floor=args.precision_floor,
            min_reject_rate=args.min_reject_rate,
            max_reject_rate=args.max_reject_rate,
            min_recall=args.min_recall,
            target_reject_rate=args.target_reject_rate,
        )
        threshold = float(threshold_info["best_threshold"])

    val_pred = (val_score >= threshold).astype(int)
    test_pred = (test_score >= threshold).astype(int)

    val_metrics = compute_metrics(val_df[TARGET_COL].astype(int), val_pred, val_score)
    test_metrics = compute_metrics(test_df[TARGET_COL].astype(int), test_pred, test_score)
    val_business = _compute_business_metrics(val_df[TARGET_COL].astype(int).to_numpy(), val_pred)
    test_business = _compute_business_metrics(test_df[TARGET_COL].astype(int).to_numpy(), test_pred)
    val_fixed = evaluate_fixed_reject_rate(
        val_df[TARGET_COL].astype(int),
        val_score,
        target_reject_rate=args.target_reject_rate,
        threshold=threshold,
    )
    test_fixed = evaluate_fixed_reject_rate(
        test_df[TARGET_COL].astype(int),
        test_score,
        target_reject_rate=args.target_reject_rate,
        threshold=threshold,
    )
    gains_lift_curve = build_validation_to_test_reject_curve(
        y_val=val_df[TARGET_COL].astype(int),
        p_val=val_score,
        y_test=test_df[TARGET_COL].astype(int),
        p_test=test_score,
        reject_rates=[i / 100.0 for i in range(5, 51, 5)],
    )
    gains_lift_curve_csv = model_dir / "analysis_gains_lift_curve.csv"
    gains_lift_curve.to_csv(gains_lift_curve_csv, index=False)

    pred_df = test_df[[TARGET_COL, ISSUE_YEAR_COL]].copy().reset_index(drop=True)
    pred_df["analysis_score"] = test_score
    pred_df["analysis_pred"] = test_pred
    pred_df["threshold_used"] = threshold
    pred_df = pd.concat([pred_df, test_contrib.reset_index(drop=True)], axis=1)
    pred_csv = model_dir / "analysis_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    profile_rows: list[dict[str, Any]] = []
    for feature, profile in profiles.items():
        for row in profile.get("stats", []):
            profile_rows.append(
                {
                    "feature": feature,
                    "feature_type": profile.get("feature_type"),
                    "feature_weight": float(profile.get("weight", 0.0)),
                    "bucket": row.get("bucket"),
                    "count": int(row.get("count", 0)),
                    "bad": int(row.get("bad", 0)),
                    "smoothed_bad_rate": float(row.get("smoothed_bad_rate", np.nan)),
                }
            )
    profile_df = pd.DataFrame(profile_rows).sort_values(
        ["feature_weight", "smoothed_bad_rate", "count"], ascending=[False, False, False]
    )
    profile_csv = analysis_dir / "feature_profiles.csv"
    profile_df.to_csv(profile_csv, index=False)

    profiles_json = analysis_dir / "feature_profiles.json"
    save_json(profiles_json, profiles)

    report = {
        "input_csv": str(input_csv),
        "rows_used": int(len(df)),
        "split_mode": "stratified",
        "split_summary": split_summary,
        "strategy": {
            "type": "analysis_based_risk_profile_classifier",
            "threshold_policy": str(args.threshold_policy),
            "objective": str(args.threshold_objective),
            "precision_floor": float(args.precision_floor),
            "min_reject_rate": float(args.min_reject_rate),
            "max_reject_rate": float(args.max_reject_rate),
            "target_reject_rate": float(args.target_reject_rate),
            "min_recall": float(args.min_recall),
        },
        "train_rows": int(len(train_df)),
        "train_fit_rows": int(len(train_fit_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_selection_json": None if feature_selection_json is None else str(feature_selection_json),
        "selected_numeric_columns": numeric_cols,
        "selected_categorical_columns": categorical_cols,
        "selected_feature_count": int(len(profiles)),
        "threshold_tuning": {
            "metric": threshold_info.get("metric"),
            "best_threshold": threshold,
            "best_score": float(threshold_info.get("best_score", np.nan)),
            "precision_floor": (
                float(threshold_info.get("precision_floor"))
                if threshold_info.get("precision_floor") is not None
                else float(args.precision_floor)
            ),
            "recall_floor": (
                float(threshold_info.get("recall_floor"))
                if threshold_info.get("recall_floor") is not None
                else float(args.min_recall)
            ),
            "min_reject_rate": (
                float(threshold_info.get("min_reject_rate"))
                if threshold_info.get("min_reject_rate") is not None
                else float(args.min_reject_rate)
            ),
            "max_reject_rate": (
                float(threshold_info.get("max_reject_rate"))
                if threshold_info.get("max_reject_rate") is not None
                else float(args.max_reject_rate)
            ),
            "target_reject_rate": (
                float(threshold_info.get("target_reject_rate"))
                if threshold_info.get("target_reject_rate") is not None
                else float(args.target_reject_rate)
            ),
            "constraint_satisfied": bool(threshold_info.get("constraint_satisfied", False)),
            "fallback_metric": threshold_info.get("fallback_metric"),
            "validation_precision": float(threshold_info.get("validation_precision", val_business["bad_rate_in_reject_precision"])),
            "validation_recall": float(threshold_info.get("validation_recall", val_business["default_capture_recall"])),
            "validation_reject_rate": float(threshold_info.get("validation_reject_rate", val_business["reject_rate"])),
            "validation_metrics_at_best_threshold": val_metrics,
            "validation_business_metrics_at_best_threshold": val_business,
            "validation_fixed_metrics_at_target_reject_rate": val_fixed,
        },
        "test_metrics": test_metrics,
        "test_business_metrics": test_business,
        "test_fixed_metrics_at_target_reject_rate": test_fixed,
        "artifacts": {
            "data_dir": str(data_dir),
            "feature_profiles_json": str(profiles_json),
            "feature_profiles_csv": str(profile_csv),
            "predictions_csv": str(pred_csv),
            "gains_lift_curve_csv": str(gains_lift_curve_csv),
        },
    }
    save_json(output_dir / "run_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone data-analysis classifier with low-volume high-confidence reject strategy."
    )
    parser.add_argument("--input", default="data/shared/shared_subset.csv")
    parser.add_argument("--output-dir", default="output/data_analysis_standalone")
    parser.add_argument(
        "--feature-selection-json",
        default="output/ml_standalone/analysis/feature_selection.json",
    )

    parser.add_argument(
        "--shared-split-dir",
        type=str,
        default="data/shared/splits/stratified",
        help="Shared split directory reused by all model families.",
    )
    parser.add_argument(
        "--force-rebuild-shared-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild shared split files instead of loading existing ones.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--max-bins", type=int, default=8)
    parser.add_argument("--min-bin-count", type=int, default=80)
    parser.add_argument("--min-category-count", type=int, default=80)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument(
        "--threshold-policy",
        default="fixed_reject_rate",
        choices=["fixed_reject_rate", "business_search"],
    )
    parser.add_argument(
        "--threshold-objective",
        default="precision_at_reject_rate_bounds",
        choices=["precision_at_reject_rate_bounds", "recall_at_precision_floor"],
    )
    parser.add_argument("--precision-floor", type=float, default=0.40)
    parser.add_argument("--min-reject-rate", type=float, default=0.05)
    parser.add_argument("--max-reject-rate", type=float, default=0.60)
    parser.add_argument("--target-reject-rate", type=float, default=0.35)
    parser.add_argument("--min-recall", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


