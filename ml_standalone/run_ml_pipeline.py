from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from workflow_common import (
    ISSUE_YEAR_COL,
    SHARED_ROW_ID_COL,
    TARGET_COL,
    TEXT_COL,
    build_validation_to_test_reject_curve,
    compute_metrics,
    downsample_binary_equal,
    evaluate_fixed_reject_rate,
    materialize_or_load_stratified_split,
    optimize_binary_threshold,
    save_json,
    stratified_train_val_test_split,
    threshold_by_target_reject_rate,
)

try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def _squeeze_text_column(frame: pd.DataFrame) -> pd.Series:
    if frame.shape[1] == 0:
        return pd.Series([], dtype=str)
    return frame.iloc[:, 0].fillna("").astype(str)


def _infer_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in df.columns:
        if col in {TARGET_COL, TEXT_COL, ISSUE_YEAR_COL, SHARED_ROW_ID_COL}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _drop_high_missing_columns(
    df: pd.DataFrame,
    cols: list[str],
    *,
    missing_threshold: float,
) -> tuple[list[str], dict[str, float]]:
    kept: list[str] = []
    dropped: dict[str, float] = {}
    for col in cols:
        ratio = float(df[col].isna().mean())
        if ratio > missing_threshold:
            dropped[col] = ratio
        else:
            kept.append(col)
    return kept, dropped


def _drop_constant_columns(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for col in cols:
        if int(df[col].nunique(dropna=False)) <= 1:
            dropped.append(col)
        else:
            kept.append(col)
    return kept, dropped


def _drop_high_correlation_numeric(
    df: pd.DataFrame,
    numeric_cols: list[str],
    *,
    corr_threshold: float,
) -> tuple[list[str], list[str]]:
    if len(numeric_cols) <= 1:
        return numeric_cols, []

    corr = df[numeric_cols].corr().abs()
    drop_set: set[str] = set()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    target_corr = {
        col: abs(float(pd.to_numeric(df[col], errors="coerce").corr(y)))
        if pd.to_numeric(df[col], errors="coerce").notna().any()
        else 0.0
        for col in numeric_cols
    }
    missing_ratio = {col: float(df[col].isna().mean()) for col in numeric_cols}

    for i, col_i in enumerate(numeric_cols):
        if col_i in drop_set:
            continue
        for col_j in numeric_cols[i + 1 :]:
            if col_j in drop_set:
                continue
            if float(corr.loc[col_i, col_j]) < corr_threshold:
                continue
            score_i = (target_corr.get(col_i, 0.0), -missing_ratio.get(col_i, 1.0))
            score_j = (target_corr.get(col_j, 0.0), -missing_ratio.get(col_j, 1.0))
            if score_i >= score_j:
                drop_set.add(col_j)
            else:
                drop_set.add(col_i)
                break

    kept = [c for c in numeric_cols if c not in drop_set]
    dropped = sorted(drop_set)
    return kept, dropped


def _build_preprocessor(
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    include_text: bool,
    text_max_features: int,
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    if include_text:
        transformers.append(
            (
                "txt",
                Pipeline(
                    steps=[
                        ("squeeze", FunctionTransformer(_squeeze_text_column, validate=False)),
                        ("tfidf", TfidfVectorizer(max_features=text_max_features, ngram_range=(1, 2), min_df=5)),
                    ]
                ),
                [TEXT_COL],
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns after feature selection.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _predict_prob(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        return np.asarray(pipe.predict_proba(x)[:, 1], dtype=float)
    decision = np.asarray(pipe.decision_function(x), dtype=float)
    return 1.0 / (1.0 + np.exp(-decision))


def _compute_business_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    yt = pd.Series(y_true).astype(int).to_numpy()
    yp = np.asarray(y_pred, dtype=int).reshape(-1)
    if yt.size == 0:
        return {
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
            "reject_rate": 0.0,
            "approve_rate": 0.0,
            "default_capture_recall": 0.0,
            "bad_rate_in_reject_precision": 0.0,
            "bad_rate_in_approve": 0.0,
        }
    tp = float(((yt == 1) & (yp == 1)).sum())
    fp = float(((yt == 0) & (yp == 1)).sum())
    tn = float(((yt == 0) & (yp == 0)).sum())
    fn = float(((yt == 1) & (yp == 0)).sum())
    n = float(yt.size)
    reject = tp + fp
    approve = tn + fn
    precision = tp / reject if reject > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bad_in_approve = fn / approve if approve > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "reject_rate": reject / n,
        "approve_rate": approve / n,
        "default_capture_recall": recall,
        "bad_rate_in_reject_precision": precision,
        "bad_rate_in_approve": bad_in_approve,
    }


def _build_manual_mask_by_uncertainty(
    y_prob: np.ndarray,
    *,
    threshold: float,
    max_manual_rate: float,
) -> np.ndarray:
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    n = p.size
    manual_mask = np.zeros(n, dtype=bool)
    if n == 0 or max_manual_rate <= 0:
        return manual_mask
    k = int(np.floor(float(max_manual_rate) * n))
    if k <= 0:
        return manual_mask
    k = min(k, n - 1) if n > 1 else 0
    if k <= 0:
        return manual_mask
    uncertainty = np.abs(p - float(threshold))
    order = np.argsort(uncertainty)
    manual_mask[order[:k]] = True
    return manual_mask


def _evaluate_manual_policy(
    y_true: pd.Series,
    y_prob: np.ndarray,
    *,
    threshold: float,
    max_manual_rate: float,
) -> dict[str, Any]:
    y = pd.Series(y_true).astype(int).to_numpy()
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    manual_mask = _build_manual_mask_by_uncertainty(
        p,
        threshold=float(threshold),
        max_manual_rate=float(max_manual_rate),
    )
    auto_mask = ~manual_mask
    pred_all = (p >= float(threshold)).astype(int)

    n = max(len(y), 1)
    auto_count = int(auto_mask.sum())
    manual_count = int(manual_mask.sum())

    if auto_count > 0:
        y_auto = pd.Series(y[auto_mask])
        pred_auto = pred_all[auto_mask]
        prob_auto = p[auto_mask]
        auto_metrics = compute_metrics(y_auto, pred_auto, prob_auto)
        auto_business = _compute_business_metrics(y_auto, pred_auto)
    else:
        auto_metrics = {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
        }
        auto_business = {
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
            "reject_rate": 0.0,
            "approve_rate": 0.0,
            "default_capture_recall": 0.0,
            "bad_rate_in_reject_precision": 0.0,
            "bad_rate_in_approve": 0.0,
        }

    tp_auto = float(((y == 1) & auto_mask & (pred_all == 1)).sum())
    fp_auto = float(((y == 0) & auto_mask & (pred_all == 1)).sum())
    fn_auto = float(((y == 1) & auto_mask & (pred_all == 0)).sum())
    auto_reject = tp_auto + fp_auto
    auto_approve = float(auto_count) - auto_reject
    total_pos = float((y == 1).sum())

    portfolio_business = {
        "precision_reject_auto": tp_auto / auto_reject if auto_reject > 0 else 0.0,
        "recall_default_auto_on_total": tp_auto / total_pos if total_pos > 0 else 0.0,
        "reject_rate_auto_on_total": auto_reject / float(n),
        "auto_rate": float(auto_count) / float(n),
        "manual_rate": float(manual_count) / float(n),
        "bad_rate_in_auto_approve": fn_auto / auto_approve if auto_approve > 0 else 0.0,
        "auto_reject_count": float(auto_reject),
        "auto_approve_count": float(auto_approve),
    }

    return {
        "manual_mask": manual_mask,
        "auto_mask": auto_mask,
        "pred_all": pred_all,
        "auto_count": auto_count,
        "manual_count": manual_count,
        "auto_metrics": auto_metrics,
        "auto_business": auto_business,
        "portfolio_business": portfolio_business,
    }


def _optimize_threshold_with_manual_review(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    *,
    precision_floor: float,
    min_reject_rate: float,
    max_reject_rate: float,
    max_manual_rate: float,
    min_auto_rate: float,
) -> dict[str, Any]:
    y = pd.Series(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    if y.size == 0 or p.size == 0 or y.size != p.size:
        return {
            "best_threshold": 0.5,
            "best_score": float("nan"),
            "metric": "manual_review_recall_at_precision_floor",
            "precision_floor": float(precision_floor),
            "min_reject_rate": float(min_reject_rate),
            "max_reject_rate": float(max_reject_rate),
            "max_manual_rate": float(max_manual_rate),
            "min_auto_rate": float(min_auto_rate),
            "constraint_satisfied": False,
            "fallback_metric": "recall_at_precision_floor_with_reject_rate_bounds",
        }

    grid = np.linspace(0.01, 0.99, 197)
    best_threshold = 0.5
    best_score = float("-inf")
    best_precision = float("-inf")
    best_auto_rate = float("-inf")
    satisfied = False

    for threshold in grid:
        evaluated = _evaluate_manual_policy(
            y,
            p,
            threshold=float(threshold),
            max_manual_rate=float(max_manual_rate),
        )
        pb = evaluated["portfolio_business"]
        precision = float(pb["precision_reject_auto"])
        recall = float(pb["recall_default_auto_on_total"])
        reject_rate = float(pb["reject_rate_auto_on_total"])
        auto_rate = float(pb["auto_rate"])

        if precision < float(precision_floor):
            continue
        if reject_rate < float(min_reject_rate) or reject_rate > float(max_reject_rate):
            continue
        if auto_rate < float(min_auto_rate):
            continue

        satisfied = True
        better = (
            recall > best_score
            or (np.isclose(recall, best_score) and precision > best_precision)
            or (np.isclose(recall, best_score) and np.isclose(precision, best_precision) and auto_rate > best_auto_rate)
            or (
                np.isclose(recall, best_score)
                and np.isclose(precision, best_precision)
                and np.isclose(auto_rate, best_auto_rate)
                and threshold > best_threshold
            )
        )
        if better:
            best_threshold = float(threshold)
            best_score = float(recall)
            best_precision = float(precision)
            best_auto_rate = float(auto_rate)

    if not satisfied:
        fallback = _optimize_threshold_business(
            y,
            p,
            threshold_objective="recall_at_precision_floor",
            precision_floor=precision_floor,
            recall_floor=0.0,
            min_reject_rate=min_reject_rate,
            max_reject_rate=max_reject_rate,
        )
        return {
            **fallback,
            "metric": "manual_review_fallback_to_binary_threshold",
            "max_manual_rate": float(max_manual_rate),
            "min_auto_rate": float(min_auto_rate),
            "constraint_satisfied": False,
            "fallback_metric": fallback.get("metric"),
        }

    return {
        "best_threshold": float(best_threshold),
        "best_score": float(best_score),
        "metric": "manual_review_recall_at_precision_floor",
        "precision_floor": float(precision_floor),
        "min_reject_rate": float(min_reject_rate),
        "max_reject_rate": float(max_reject_rate),
        "max_manual_rate": float(max_manual_rate),
        "min_auto_rate": float(min_auto_rate),
        "constraint_satisfied": True,
        "fallback_metric": None,
    }


def _optimize_threshold_business(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold_objective: str,
    precision_floor: float,
    recall_floor: float,
    min_reject_rate: float,
    max_reject_rate: float,
) -> dict[str, Any]:
    allowed_objectives = {"recall_at_precision_floor", "precision_at_reject_rate_bounds"}
    if threshold_objective not in allowed_objectives:
        raise ValueError(f"Unsupported threshold objective: {threshold_objective}")

    metric_name = (
        "recall_at_precision_floor_with_reject_rate_bounds"
        if threshold_objective == "recall_at_precision_floor"
        else "precision_at_reject_rate_bounds"
    )
    y = pd.Series(y_true).astype(int).to_numpy()
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    if y.size == 0 or p.size == 0 or y.size != p.size:
        return {
            "best_threshold": 0.5,
            "best_score": float("nan"),
            "metric": metric_name,
            "precision_floor": float(precision_floor),
            "recall_floor": float(recall_floor),
            "min_reject_rate": float(min_reject_rate),
            "max_reject_rate": float(max_reject_rate),
            "constraint_satisfied": False,
            "fallback_metric": (
                "recall_at_precision_floor" if threshold_objective == "recall_at_precision_floor" else "precision"
            ),
        }

    grid = np.linspace(0.01, 0.99, 197)
    best_score = float("-inf")
    best_threshold = 0.5
    best_precision = float("-inf")
    best_recall = float("-inf")
    best_reject_rate = float("inf")
    satisfied = False

    for threshold in grid:
        pred = (p >= threshold).astype(int)
        biz = _compute_business_metrics(pd.Series(y), pred)
        precision = float(biz["bad_rate_in_reject_precision"])
        recall = float(biz["default_capture_recall"])
        reject_rate = float(biz["reject_rate"])
        if reject_rate < float(min_reject_rate) or reject_rate > float(max_reject_rate):
            continue
        if precision < float(precision_floor):
            continue
        if recall < float(recall_floor):
            continue
        satisfied = True
        if threshold_objective == "precision_at_reject_rate_bounds":
            score = precision
            better = (
                score > best_score
                or (np.isclose(score, best_score) and recall > best_recall)
                or (
                    np.isclose(score, best_score)
                    and np.isclose(recall, best_recall)
                    and reject_rate < best_reject_rate
                )
                or (
                    np.isclose(score, best_score)
                    and np.isclose(recall, best_recall)
                    and np.isclose(reject_rate, best_reject_rate)
                    and threshold > best_threshold
                )
            )
        else:
            score = recall
            better = (
                score > best_score
                or (np.isclose(score, best_score) and precision > best_precision)
                or (
                    np.isclose(score, best_score)
                    and np.isclose(precision, best_precision)
                    and reject_rate < best_reject_rate
                )
                or (
                    np.isclose(score, best_score)
                    and np.isclose(precision, best_precision)
                    and np.isclose(reject_rate, best_reject_rate)
                    and threshold > best_threshold
                )
            )
        if better:
            best_score = score
            best_threshold = float(threshold)
            best_precision = precision
            best_recall = recall
            best_reject_rate = reject_rate

    if not satisfied:
        fallback_metric = "recall_at_precision_floor" if threshold_objective == "recall_at_precision_floor" else "precision"
        fallback = optimize_binary_threshold(y, p, metric=fallback_metric, precision_floor=precision_floor)
        precision_satisfied = bool(fallback.get("constraint_satisfied", False)) if fallback_metric == "recall_at_precision_floor" else True
        fallback["metric"] = (
            "recall_at_precision_floor_fallback_no_reject_bound_solution"
            if threshold_objective == "recall_at_precision_floor"
            else "precision_fallback_no_reject_bound_solution"
        )
        fallback["constraint_satisfied"] = False
        fallback["reject_rate_constraint_satisfied"] = False
        fallback["precision_constraint_satisfied"] = precision_satisfied
        fallback["recall_floor"] = float(recall_floor)
        fallback["min_reject_rate"] = float(min_reject_rate)
        fallback["max_reject_rate"] = float(max_reject_rate)
        return fallback

    return {
        "best_threshold": float(best_threshold),
        "best_score": float(best_score),
        "metric": metric_name,
        "precision_floor": float(precision_floor),
        "recall_floor": float(recall_floor),
        "min_reject_rate": float(min_reject_rate),
        "max_reject_rate": float(max_reject_rate),
        "constraint_satisfied": True,
        "reject_rate_constraint_satisfied": True,
        "precision_constraint_satisfied": True,
        "fallback_metric": None,
    }


def _fit_model_with_threshold(
    *,
    name: str,
    estimator: Any,
    preprocessor: ColumnTransformer,
    x_train_fit: pd.DataFrame,
    y_train_fit: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test_eval: pd.DataFrame,
    y_test_eval: pd.Series,
    x_test_raw: pd.DataFrame,
    y_test_raw: pd.Series,
    cv_folds: int,
    threshold_policy: str,
    target_reject_rate: float,
    threshold_objective: str,
    precision_floor: float,
    recall_floor: float,
    min_reject_rate: float,
    max_reject_rate: float,
    enable_manual_review: bool,
    max_manual_rate: float,
    min_auto_rate: float,
    random_state: int,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipe, x_train_fit, y_train_fit, cv=cv, scoring="balanced_accuracy", n_jobs=1)

    pipe.fit(x_train_fit, y_train_fit)

    val_prob = _predict_prob(pipe, x_val)
    test_eval_prob = _predict_prob(pipe, x_test_eval)
    test_raw_prob = _predict_prob(pipe, x_test_raw)

    if threshold_policy == "fixed_reject_rate":
        threshold_info = threshold_by_target_reject_rate(
            val_prob,
            target_reject_rate=target_reject_rate,
        )
        threshold_info = {
            "metric": "fixed_reject_rate_quantile",
            "best_threshold": float(threshold_info["threshold"]),
            "best_score": float("nan"),
            "target_reject_rate": float(target_reject_rate),
            "validation_reject_rate": float(threshold_info["actual_reject_rate"]),
            "precision_floor": None,
            "recall_floor": None,
            "min_reject_rate": None,
            "max_reject_rate": None,
            "constraint_satisfied": True,
            "fallback_metric": None,
        }
    elif enable_manual_review:
        threshold_info = _optimize_threshold_with_manual_review(
            y_val,
            val_prob,
            precision_floor=precision_floor,
            min_reject_rate=min_reject_rate,
            max_reject_rate=max_reject_rate,
            max_manual_rate=max_manual_rate,
            min_auto_rate=min_auto_rate,
        )
    else:
        threshold_info = _optimize_threshold_business(
            y_val,
            val_prob,
            threshold_objective=threshold_objective,
            precision_floor=precision_floor,
            recall_floor=recall_floor,
            min_reject_rate=min_reject_rate,
            max_reject_rate=max_reject_rate,
        )
    threshold = float(threshold_info["best_threshold"])
    val_pred = (val_prob >= threshold).astype(int)
    test_eval_pred = (test_eval_prob >= threshold).astype(int)
    test_raw_pred = (test_raw_prob >= threshold).astype(int)

    val_fixed_metrics = evaluate_fixed_reject_rate(
        y_val,
        val_prob,
        target_reject_rate=target_reject_rate,
        threshold=threshold,
    )
    test_eval_fixed_metrics = evaluate_fixed_reject_rate(
        y_test_eval,
        test_eval_prob,
        target_reject_rate=target_reject_rate,
        threshold=threshold,
    )
    test_raw_fixed_metrics = evaluate_fixed_reject_rate(
        y_test_raw,
        test_raw_prob,
        target_reject_rate=target_reject_rate,
        threshold=threshold,
    )
    gains_lift_curve = build_validation_to_test_reject_curve(
        y_val=y_val,
        p_val=val_prob,
        y_test=y_test_raw,
        p_test=test_raw_prob,
        reject_rates=[i / 100.0 for i in range(5, 51, 5)],
    )

    manual_rate = float(max_manual_rate) if enable_manual_review else 0.0
    val_manual = _evaluate_manual_policy(
        y_val,
        val_prob,
        threshold=threshold,
        max_manual_rate=manual_rate,
    )
    test_eval_manual = _evaluate_manual_policy(
        y_test_eval,
        test_eval_prob,
        threshold=threshold,
        max_manual_rate=manual_rate,
    )
    test_raw_manual = _evaluate_manual_policy(
        y_test_raw,
        test_raw_prob,
        threshold=threshold,
        max_manual_rate=manual_rate,
    )

    val_business = _compute_business_metrics(y_val, val_pred)
    test_eval_business = _compute_business_metrics(y_test_eval, test_eval_pred)
    test_raw_business = _compute_business_metrics(y_test_raw, test_raw_pred)
    val_metrics = compute_metrics(y_val, val_pred, val_prob)
    test_eval_metrics = compute_metrics(y_test_eval, test_eval_pred, test_eval_prob)
    test_raw_metrics = compute_metrics(y_test_raw, test_raw_pred, test_raw_prob)
    preds = pd.DataFrame(
        {
            "y_true": y_test_raw.astype(int).to_numpy(),
            "y_prob": test_raw_prob,
            "y_pred_binary": test_raw_pred,
            "is_manual_review": test_raw_manual["manual_mask"].astype(int),
            "y_pred_with_manual": np.where(test_raw_manual["manual_mask"], -1, test_raw_pred),
            "threshold_used": threshold,
            "evaluation_set": "test_raw",
            "model_name": name,
        }
    )

    payload = {
        "model_name": name,
        "cv_balanced_accuracy_mean": float(np.mean(cv_scores)),
        "cv_balanced_accuracy_std": float(np.std(cv_scores)),
        "threshold_tuning": {
            "metric": threshold_info["metric"],
            "precision_floor": threshold_info.get("precision_floor"),
            "recall_floor": threshold_info.get("recall_floor"),
            "min_reject_rate": threshold_info.get("min_reject_rate"),
            "max_reject_rate": threshold_info.get("max_reject_rate"),
            "max_manual_rate": threshold_info.get("max_manual_rate"),
            "min_auto_rate": threshold_info.get("min_auto_rate"),
            "constraint_satisfied": bool(threshold_info.get("constraint_satisfied", False)),
            "fallback_metric": threshold_info.get("fallback_metric"),
            "best_threshold": threshold,
            "best_score": float(threshold_info["best_score"]),
            "target_reject_rate": float(target_reject_rate),
            "validation_metrics_at_best_threshold": val_metrics,
            "validation_business_metrics_at_best_threshold": val_business,
            "validation_fixed_metrics_at_target_reject_rate": val_fixed_metrics,
            "validation_manual_policy_at_best_threshold": {
                "auto_count": int(val_manual["auto_count"]),
                "manual_count": int(val_manual["manual_count"]),
                "auto_metrics": val_manual["auto_metrics"],
                "auto_business": val_manual["auto_business"],
                "portfolio_business": val_manual["portfolio_business"],
            },
        },
        "manual_review_enabled": bool(enable_manual_review),
        "test_metrics_eval": test_eval_metrics,
        "test_business_metrics_eval": test_eval_business,
        "test_fixed_metrics_eval": test_eval_fixed_metrics,
        "test_metrics_raw": test_raw_metrics,
        "test_business_metrics_raw": test_raw_business,
        "test_fixed_metrics_raw": test_raw_fixed_metrics,
        "gains_lift_curve_rows": gains_lift_curve.to_dict(orient="records"),
        "test_manual_policy_eval": {
            "auto_count": int(test_eval_manual["auto_count"]),
            "manual_count": int(test_eval_manual["manual_count"]),
            "auto_metrics": test_eval_manual["auto_metrics"],
            "auto_business": test_eval_manual["auto_business"],
            "portfolio_business": test_eval_manual["portfolio_business"],
        },
        "test_manual_policy_raw": {
            "auto_count": int(test_raw_manual["auto_count"]),
            "manual_count": int(test_raw_manual["manual_count"]),
            "auto_metrics": test_raw_manual["auto_metrics"],
            "auto_business": test_raw_manual["auto_business"],
            "portfolio_business": test_raw_manual["portfolio_business"],
        },
    }
    return pipe, payload, preds


def _build_eda_reports(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols, categorical_cols = _infer_columns(df)

    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_ratio")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing.to_csv(output_dir / "missing_ratio.csv", index=False)

    numeric_summary = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T if numeric_cols else pd.DataFrame()
    if not numeric_summary.empty:
        numeric_summary.to_csv(output_dir / "numeric_summary.csv")

    cat_rows: list[dict[str, Any]] = []
    for col in categorical_cols:
        vc = df[col].astype(str).value_counts(dropna=False)
        top_value = None if vc.empty else str(vc.index[0])
        top_ratio = 0.0 if vc.empty else float(vc.iloc[0] / max(len(df), 1))
        cat_rows.append(
            {
                "column": col,
                "n_unique": int(df[col].nunique(dropna=False)),
                "top_value": top_value,
                "top_ratio": top_ratio,
                "missing_ratio": float(df[col].isna().mean()),
            }
        )
    cat_df = pd.DataFrame(cat_rows)
    if not cat_df.empty:
        cat_df.to_csv(output_dir / "categorical_summary.csv", index=False)

    by_year = (
        df.groupby(ISSUE_YEAR_COL)[TARGET_COL]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "default_rate"})
    )
    by_year.to_csv(output_dir / "target_by_year.csv", index=False)

    report = {
        "rows_total": int(len(df)),
        "cols_total": int(len(df.columns)),
        "target_distribution": {str(k): int(v) for k, v in df[TARGET_COL].astype(int).value_counts().to_dict().items()},
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_target_distribution": {
            str(k): int(v) for k, v in train_df[TARGET_COL].astype(int).value_counts().to_dict().items()
        },
        "test_target_distribution": {
            str(k): int(v) for k, v in test_df[TARGET_COL].astype(int).value_counts().to_dict().items()
        },
        "numeric_columns_initial": numeric_cols,
        "categorical_columns_initial": categorical_cols,
        "artifacts": {
            "missing_ratio_csv": str(output_dir / "missing_ratio.csv"),
            "numeric_summary_csv": str(output_dir / "numeric_summary.csv"),
            "categorical_summary_csv": str(output_dir / "categorical_summary.csv"),
            "target_by_year_csv": str(output_dir / "target_by_year.csv"),
        },
    }
    save_json(output_dir / "eda_report.json", report)
    return report


def _select_features(
    df: pd.DataFrame,
    *,
    missing_threshold: float,
    corr_threshold: float,
) -> dict[str, Any]:
    numeric_cols, categorical_cols = _infer_columns(df)

    numeric_cols, dropped_missing_num = _drop_high_missing_columns(df, numeric_cols, missing_threshold=missing_threshold)
    categorical_cols, dropped_missing_cat = _drop_high_missing_columns(df, categorical_cols, missing_threshold=missing_threshold)

    numeric_cols, dropped_const_num = _drop_constant_columns(df, numeric_cols)
    categorical_cols, dropped_const_cat = _drop_constant_columns(df, categorical_cols)

    numeric_cols, dropped_corr_num = _drop_high_correlation_numeric(
        df,
        numeric_cols,
        corr_threshold=corr_threshold,
    )

    selected = [*numeric_cols, *categorical_cols, TEXT_COL]
    report = {
        "selected_numeric_columns": numeric_cols,
        "selected_categorical_columns": categorical_cols,
        "selected_text_column": TEXT_COL,
        "selected_all_columns": selected,
        "dropped_by_missing_ratio": {
            "numeric": dropped_missing_num,
            "categorical": dropped_missing_cat,
        },
        "dropped_by_constant": {
            "numeric": dropped_const_num,
            "categorical": dropped_const_cat,
        },
        "dropped_by_high_correlation_numeric": dropped_corr_num,
        "missing_threshold": float(missing_threshold),
        "corr_threshold": float(corr_threshold),
    }
    return report


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
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
    required = {TARGET_COL, TEXT_COL, ISSUE_YEAR_COL}
    missing_required = sorted(c for c in required if c not in df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df[ISSUE_YEAR_COL] = pd.to_numeric(df[ISSUE_YEAR_COL], errors="coerce")
    df = df[df[ISSUE_YEAR_COL].notna()].copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df = df[df[TEXT_COL].str.strip() != ""].copy()

    shared_split_dir = str(args.shared_split_dir).strip()
    if shared_split_dir:
        train_fit_raw_df, val_df, test_df_raw, split_summary = materialize_or_load_stratified_split(
            df,
            split_dir=Path(shared_split_dir),
            target_col=TARGET_COL,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            force_rebuild=bool(args.force_rebuild_shared_split),
        )
    else:
        train_fit_raw_df, val_df, test_df_raw, split_summary = stratified_train_val_test_split(
            df,
            target_col=TARGET_COL,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )
    train_df_raw = pd.concat([train_fit_raw_df, val_df], axis=0).reset_index(drop=True)

    df.to_csv(data_dir / "ml_processed.csv", index=False)
    train_fit_raw_df.to_csv(data_dir / "train_fit.csv", index=False)
    val_df.to_csv(data_dir / "validation.csv", index=False)
    test_df_raw.to_csv(data_dir / "test.csv", index=False)
    save_json(
        data_dir / "split_summary.json",
        {
            "input_csv": str(input_csv),
            "split_mode": "stratified",
            "split_summary": split_summary,
        },
    )

    eda_report = _build_eda_reports(df, output_dir=analysis_dir, train_df=train_df_raw, test_df=test_df_raw)

    train_balance_summary: dict[str, int] | None = None
    train_fit_df = train_fit_raw_df.copy()
    if args.balance_train:
        train_fit_df, train_balance_summary = downsample_binary_equal(
            train_fit_raw_df,
            target_col=TARGET_COL,
            random_state=args.random_state,
            pos_cap=args.train_pos_cap,
        )
    train_fit_df.to_csv(data_dir / "train_fit_for_training.csv", index=False)

    test_balance_summary: dict[str, int] | None = None
    test_eval_df = test_df_raw.copy()
    if args.balance_test:
        test_eval_df, test_balance_summary = downsample_binary_equal(
            test_df_raw,
            target_col=TARGET_COL,
            random_state=args.random_state,
            pos_cap=args.test_pos_cap,
        )
    test_eval_df.to_csv(data_dir / "test_eval_for_metrics.csv", index=False)

    selection_report = _select_features(
        train_fit_raw_df,
        missing_threshold=args.max_missing_ratio,
        corr_threshold=args.max_numeric_corr,
    )
    save_json(analysis_dir / "feature_selection.json", selection_report)

    feature_cols = selection_report["selected_all_columns"]
    x_train_fit = train_fit_df[feature_cols].copy()
    y_train_fit = train_fit_df[TARGET_COL].astype(int).copy()
    x_val = val_df[feature_cols].copy()
    y_val = val_df[TARGET_COL].astype(int).copy()
    x_test_eval = test_eval_df[feature_cols].copy()
    y_test_eval = test_eval_df[TARGET_COL].astype(int).copy()
    x_test_raw = test_df_raw[feature_cols].copy()
    y_test_raw = test_df_raw[TARGET_COL].astype(int).copy()

    num_cols = selection_report["selected_numeric_columns"]
    cat_cols = selection_report["selected_categorical_columns"]
    preprocessor_tabular = _build_preprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        include_text=False,
        text_max_features=args.text_max_features,
    )
    preprocessor_text = _build_preprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        include_text=True,
        text_max_features=args.text_max_features,
    )

    candidates: list[tuple[str, Any, ColumnTransformer]] = []
    candidates.append(
        (
            "logistic_tabular",
            LogisticRegression(
                max_iter=1800,
                class_weight="balanced",
                random_state=args.random_state,
            ),
            preprocessor_tabular,
        )
    )
    candidates.append(
        (
            "logistic_text_fusion",
            LogisticRegression(
                max_iter=1800,
                class_weight="balanced",
                random_state=args.random_state,
            ),
            preprocessor_text,
        )
    )
    if HAS_XGBOOST:
        pos_rate = float(y_train_fit.mean())
        scale_pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-8)
        candidates.append(
            (
                "xgboost_tabular",
                XGBClassifier(
                    n_estimators=260,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    reg_lambda=1.0,
                    tree_method="hist",
                    n_jobs=args.n_jobs,
                    random_state=args.random_state,
                    scale_pos_weight=scale_pos_weight,
                ),
                preprocessor_tabular,
            )
        )

    all_results: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    best_tuple: tuple[float, float, float] | None = None
    best_model: Pipeline | None = None
    best_name: str | None = None

    for name, estimator, prep in candidates:
        model, result, pred = _fit_model_with_threshold(
            name=name,
            estimator=clone(estimator),
            preprocessor=clone(prep),
            x_train_fit=x_train_fit,
            y_train_fit=y_train_fit,
            x_val=x_val,
            y_val=y_val,
            x_test_eval=x_test_eval,
            y_test_eval=y_test_eval,
            x_test_raw=x_test_raw,
            y_test_raw=y_test_raw,
            cv_folds=args.cv_folds,
            threshold_policy=args.threshold_policy,
            target_reject_rate=args.target_reject_rate,
            threshold_objective=args.threshold_objective,
            precision_floor=args.precision_floor,
            recall_floor=args.recall_floor,
            min_reject_rate=args.min_reject_rate,
            max_reject_rate=args.max_reject_rate,
            enable_manual_review=args.enable_manual_review,
            max_manual_rate=args.max_manual_rate,
            min_auto_rate=args.min_auto_rate,
            random_state=args.random_state,
        )
        all_results.append(result)
        pred_frames.append(pred)

        if args.threshold_policy == "fixed_reject_rate":
            val_fixed = result["threshold_tuning"]["validation_fixed_metrics_at_target_reject_rate"]
            score = (
                float(val_fixed["precision_at_target_reject_rate"]),
                float(val_fixed["recall_at_target_reject_rate"]),
                -abs(float(val_fixed["actual_reject_rate"]) - float(args.target_reject_rate)),
            )
        elif args.enable_manual_review:
            val_policy = result["threshold_tuning"]["validation_manual_policy_at_best_threshold"]["portfolio_business"]
            score = (
                float(val_policy["recall_default_auto_on_total"]),
                float(val_policy["precision_reject_auto"]),
                float(val_policy["auto_rate"]),
            )
        else:
            val_metrics = result["threshold_tuning"]["validation_business_metrics_at_best_threshold"]
            if args.threshold_objective == "precision_at_reject_rate_bounds":
                score = (
                    float(val_metrics["bad_rate_in_reject_precision"]),
                    float(val_metrics["default_capture_recall"]),
                    -float(val_metrics["reject_rate"]),
                )
            else:
                score = (
                    float(result["threshold_tuning"]["best_score"]),
                    float(val_metrics["bad_rate_in_reject_precision"]),
                    -float(val_metrics["reject_rate"]),
                )
        if best_tuple is None or score > best_tuple:
            best_tuple = score
            best_model = model
            best_name = name

    if best_model is None or best_name is None:
        raise RuntimeError("No model was trained.")

    pred_df = pd.concat(pred_frames, axis=0, ignore_index=True)
    pred_df.to_csv(model_dir / "model_predictions.csv", index=False)
    joblib.dump(best_model, model_dir / "best_model.joblib")

    summary_rows: list[dict[str, Any]] = []
    for result in all_results:
        val_metrics = result["threshold_tuning"]["validation_metrics_at_best_threshold"]
        val_business = result["threshold_tuning"]["validation_business_metrics_at_best_threshold"]
        val_fixed = result["threshold_tuning"]["validation_fixed_metrics_at_target_reject_rate"]
        test_eval_fixed = result["test_fixed_metrics_eval"]
        test_raw_fixed = result["test_fixed_metrics_raw"]
        val_manual_pb = result["threshold_tuning"]["validation_manual_policy_at_best_threshold"]["portfolio_business"]
        test_raw_manual_pb = result["test_manual_policy_raw"]["portfolio_business"]
        summary_rows.append(
            {
                "model_name": result["model_name"],
                "selected_by_best_score": int(result["model_name"] == best_name),
                "cv_balanced_accuracy_mean": result["cv_balanced_accuracy_mean"],
                "threshold_metric": result["threshold_tuning"]["metric"],
                "threshold": result["threshold_tuning"]["best_threshold"],
                "threshold_best_score": result["threshold_tuning"]["best_score"],
                "target_reject_rate": result["threshold_tuning"]["target_reject_rate"],
                "recall_at_precision_floor_val": result["threshold_tuning"]["best_score"],
                "precision_reject_val": val_metrics["precision"],
                "recall_default_val": val_metrics["recall"],
                "reject_rate_val": val_business["reject_rate"],
                "precision_reject_val_at_target_rr": val_fixed["precision_at_target_reject_rate"],
                "recall_default_val_at_target_rr": val_fixed["recall_at_target_reject_rate"],
                "reject_rate_val_at_target_rr": val_fixed["actual_reject_rate"],
                "precision_reject_test_eval": result["test_metrics_eval"]["precision"],
                "recall_default_test_eval": result["test_metrics_eval"]["recall"],
                "reject_rate_test_eval": result["test_business_metrics_eval"]["reject_rate"],
                "precision_reject_test_eval_at_target_rr": test_eval_fixed["precision_at_target_reject_rate"],
                "recall_default_test_eval_at_target_rr": test_eval_fixed["recall_at_target_reject_rate"],
                "approval_bad_rate_test_eval_at_target_rr": test_eval_fixed["approval_bad_rate"],
                "lift_test_eval_at_target_rr": test_eval_fixed["lift_at_target_reject_rate"],
                "precision_reject_test_raw": result["test_metrics_raw"]["precision"],
                "recall_default_test_raw": result["test_metrics_raw"]["recall"],
                "reject_rate_test_raw": result["test_business_metrics_raw"]["reject_rate"],
                "precision_reject_test_raw_at_target_rr": test_raw_fixed["precision_at_target_reject_rate"],
                "recall_default_test_raw_at_target_rr": test_raw_fixed["recall_at_target_reject_rate"],
                "approval_bad_rate_test_raw_at_target_rr": test_raw_fixed["approval_bad_rate"],
                "lift_test_raw_at_target_rr": test_raw_fixed["lift_at_target_reject_rate"],
                "f1_test_raw": result["test_metrics_raw"]["f1"],
                "pr_auc_test_raw": result["test_metrics_raw"]["pr_auc"],
                "roc_auc_test_raw": result["test_metrics_raw"]["roc_auc"],
                "auto_rate_val_manual": val_manual_pb["auto_rate"],
                "manual_rate_val_manual": val_manual_pb["manual_rate"],
                "precision_reject_val_manual": val_manual_pb["precision_reject_auto"],
                "recall_default_val_manual_on_total": val_manual_pb["recall_default_auto_on_total"],
                "auto_rate_test_raw_manual": test_raw_manual_pb["auto_rate"],
                "manual_rate_test_raw_manual": test_raw_manual_pb["manual_rate"],
                "precision_reject_test_raw_manual": test_raw_manual_pb["precision_reject_auto"],
                "recall_default_test_raw_manual_on_total": test_raw_manual_pb["recall_default_auto_on_total"],
            }
        )
    if args.threshold_policy == "fixed_reject_rate":
        summary_df = pd.DataFrame(summary_rows).sort_values(
            [
                "precision_reject_val_at_target_rr",
                "recall_default_val_at_target_rr",
                "reject_rate_val_at_target_rr",
            ],
            ascending=[False, False, True],
        )
    elif args.enable_manual_review:
        summary_df = pd.DataFrame(summary_rows).sort_values(
            [
                "recall_default_val_manual_on_total",
                "precision_reject_val_manual",
                "auto_rate_val_manual",
            ],
            ascending=[False, False, False],
        )
    else:
        if args.threshold_objective == "precision_at_reject_rate_bounds":
            summary_df = pd.DataFrame(summary_rows).sort_values(
                ["precision_reject_val", "recall_default_val", "reject_rate_val"],
                ascending=[False, False, True],
            )
        else:
            summary_df = pd.DataFrame(summary_rows).sort_values(
                ["recall_at_precision_floor_val", "precision_reject_val", "reject_rate_val"],
                ascending=[False, False, True],
            )
    summary_df.to_csv(model_dir / "model_summary.csv", index=False)

    gains_lift_artifacts: dict[str, str] = {}
    for result in all_results:
        curve_rows = result.get("gains_lift_curve_rows", [])
        curve_df = pd.DataFrame(curve_rows)
        curve_path = model_dir / f"{result['model_name']}_gains_lift_curve.csv"
        curve_df.to_csv(curve_path, index=False)
        gains_lift_artifacts[str(result["model_name"])] = str(curve_path)

    report = {
        "input_csv": str(input_csv),
        "rows_used": int(len(df)),
        "split_mode": "stratified",
        "split_summary": split_summary,
        "threshold_policy": str(args.threshold_policy),
        "target_reject_rate": float(args.target_reject_rate),
        "precision_floor": float(args.precision_floor),
        "recall_floor": float(args.recall_floor),
        "threshold_objective": str(args.threshold_objective),
        "threshold_min_reject_rate": float(args.min_reject_rate),
        "threshold_max_reject_rate": float(args.max_reject_rate),
        "manual_review_enabled": bool(args.enable_manual_review),
        "max_manual_rate": float(args.max_manual_rate),
        "min_auto_rate": float(args.min_auto_rate),
        "train_balance_enabled": bool(args.balance_train),
        "train_balance_summary": train_balance_summary,
        "validation_source": "train_raw_distribution",
        "validation_rows": int(len(val_df)),
        "validation_positive_rate": float(y_val.mean()),
        "test_balance_enabled": bool(args.balance_test),
        "test_balance_summary": test_balance_summary,
        "test_eval_rows": int(len(test_eval_df)),
        "test_eval_positive_rate": float(y_test_eval.mean()),
        "test_raw_rows": int(len(test_df_raw)),
        "test_raw_positive_rate": float(y_test_raw.mean()),
        "feature_selection": selection_report,
        "eda_report": eda_report,
        "models": all_results,
        "best_model_name": best_name,
        "artifacts": {
            "data_dir": str(data_dir),
            "feature_selection_json": str(analysis_dir / "feature_selection.json"),
            "eda_report_json": str(analysis_dir / "eda_report.json"),
            "model_summary_csv": str(model_dir / "model_summary.csv"),
            "predictions_csv": str(model_dir / "model_predictions.csv"),
            "best_model_joblib": str(model_dir / "best_model.joblib"),
            "gains_lift_curves": gains_lift_artifacts,
        },
    }
    save_json(output_dir / "run_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone ML pipeline: EDA + preprocessing + model selection.")
    parser.add_argument("--input", default="data/shared/shared_subset.csv")
    parser.add_argument("--output-dir", default="output/ml_standalone")

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
    parser.add_argument(
        "--threshold-policy",
        type=str,
        default="fixed_reject_rate",
        choices=["fixed_reject_rate", "business_search"],
    )
    parser.add_argument("--target-reject-rate", type=float, default=0.35)
    parser.add_argument(
        "--threshold-objective",
        type=str,
        default="precision_at_reject_rate_bounds",
        choices=["recall_at_precision_floor", "precision_at_reject_rate_bounds"],
    )
    parser.add_argument("--precision-floor", type=float, default=0.5)
    parser.add_argument("--recall-floor", type=float, default=0.0)
    parser.add_argument("--min-reject-rate", type=float, default=0.05)
    parser.add_argument("--max-reject-rate", type=float, default=0.60)
    parser.add_argument("--enable-manual-review", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-manual-rate", type=float, default=0.25)
    parser.add_argument("--min-auto-rate", type=float, default=0.60)
    parser.add_argument("--balance-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-pos-cap", type=int, default=None)
    parser.add_argument("--balance-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test-pos-cap", type=int, default=None)
    parser.add_argument("--max-missing-ratio", type=float, default=0.30)
    parser.add_argument("--max-numeric-corr", type=float, default=0.98)
    parser.add_argument("--text-max-features", type=int, default=30000)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_pipeline(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


