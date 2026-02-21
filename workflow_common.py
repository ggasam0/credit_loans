from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COL = "target"
RAW_TEXT_COL = "desc"
TEXT_COL = "desc_clean"
STATUS_COL = "loan_status"
ISSUE_COL = "issue_d"
ISSUE_YEAR_COL = "issue_year"
SHARED_ROW_ID_COL = "_shared_row_id"

ALLOWED_STATUS = {"Fully Paid", "Charged Off"}

NUMERIC_CANDIDATES = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "revol_bal",
    "revol_util",
    "open_acc",
    "pub_rec",
    "total_acc",
]

CATEGORICAL_CANDIDATES = [
    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
]

BASE_USECOLS = sorted(set(NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES + [RAW_TEXT_COL, STATUS_COL, ISSUE_COL]))

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
EMP_LEN_RE = re.compile(r"(\d+)")
YEAR_RE = re.compile(r"(19|20)\d{2}")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    if not t or t.lower() == "nan":
        return ""
    t = html.unescape(t)
    t = TAG_RE.sub(" ", t)
    t = SPACE_RE.sub(" ", t)
    return t.strip().lower()


def parse_emp_length(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    if "< 1" in text:
        return 0.5
    match = EMP_LEN_RE.search(text)
    if not match:
        return np.nan
    return float(match.group(1))


def to_term_months(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    match = EMP_LEN_RE.search(text)
    if not match:
        return np.nan
    return float(match.group(1))


def parse_issue_year(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    match = YEAR_RE.search(text)
    if not match:
        return np.nan
    return float(match.group(0))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def compute_business_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = pd.Series(y_true).astype(int).to_numpy()
    p = np.asarray(y_pred).astype(int).reshape(-1)
    if y.size == 0 or p.size == 0 or y.size != p.size:
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
    tp = float(((y == 1) & (p == 1)).sum())
    fp = float(((y == 0) & (p == 1)).sum())
    tn = float(((y == 0) & (p == 0)).sum())
    fn = float(((y == 1) & (p == 0)).sum())
    n = float(max(y.size, 1))
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


def optimize_binary_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "balanced_accuracy",
    precision_floor: float | None = None,
) -> dict[str, Any]:
    y = pd.Series(y_true).astype(int).to_numpy()
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    if y.size == 0 or p.size == 0 or y.size != p.size:
        return {
            "best_threshold": 0.5,
            "best_score": float("nan"),
            "metric": metric,
            "precision_floor": precision_floor,
            "constraint_satisfied": False,
            "fallback_metric": None,
        }

    grid = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_score = float("-inf")
    best_precision = float("-inf")
    constraint_satisfied = False
    floor = float(precision_floor) if precision_floor is not None else None

    for threshold in grid:
        pred = (p >= threshold).astype(int)
        precision = float(precision_score(y, pred, zero_division=0))
        recall = float(recall_score(y, pred, zero_division=0))
        if metric == "recall_at_precision_floor":
            if floor is not None and precision < floor:
                continue
            score = recall
            constraint_satisfied = True
            better = (
                score > best_score
                or (np.isclose(score, best_score) and precision > best_precision)
                or (np.isclose(score, best_score) and np.isclose(precision, best_precision) and threshold > best_threshold)
            )
        elif metric == "f1":
            score = float(f1_score(y, pred, zero_division=0))
            better = score > best_score
        elif metric == "precision":
            score = precision
            better = score > best_score
        elif metric == "recall":
            score = recall
            better = score > best_score
        else:
            score = float(balanced_accuracy_score(y, pred))
            better = score > best_score
        if better:
            best_score = score
            best_threshold = float(threshold)
            best_precision = precision

    fallback_metric: str | None = None
    if metric == "recall_at_precision_floor" and not constraint_satisfied:
        fallback_metric = "f1"
        best_threshold = 0.5
        best_score = float("-inf")
        for threshold in grid:
            pred = (p >= threshold).astype(int)
            score = float(f1_score(y, pred, zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

    return {
        "best_threshold": float(best_threshold),
        "best_score": float(best_score),
        "metric": metric,
        "precision_floor": floor,
        "constraint_satisfied": bool(constraint_satisfied),
        "fallback_metric": fallback_metric,
    }


def optimize_binary_threshold_business(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold_objective: str = "precision_at_reject_rate_bounds",
    precision_floor: float = 0.0,
    recall_floor: float = 0.0,
    min_reject_rate: float = 0.0,
    max_reject_rate: float = 1.0,
    target_reject_rate: float | None = None,
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
    floor_p = float(max(0.0, precision_floor))
    floor_r = float(max(0.0, recall_floor))
    rr_min = float(min(min_reject_rate, max_reject_rate))
    rr_max = float(max(min_reject_rate, max_reject_rate))
    rr_min = max(0.0, min(rr_min, 1.0))
    rr_max = max(0.0, min(rr_max, 1.0))
    rr_target = (
        float(max(rr_min, min(rr_max, target_reject_rate)))
        if target_reject_rate is not None
        else float((rr_min + rr_max) / 2.0)
    )

    if y.size == 0 or p.size == 0 or y.size != p.size:
        return {
            "best_threshold": 0.5,
            "best_score": float("nan"),
            "metric": metric_name,
            "precision_floor": floor_p,
            "recall_floor": floor_r,
            "min_reject_rate": rr_min,
            "max_reject_rate": rr_max,
            "target_reject_rate": rr_target,
            "constraint_satisfied": False,
            "fallback_metric": "f1",
            "validation_precision": 0.0,
            "validation_recall": 0.0,
            "validation_reject_rate": 0.0,
        }

    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 197),
                np.quantile(p, np.linspace(0.01, 0.99, 99)),
            ]
        )
    )

    best_tuple: tuple[float, float, float, float, float] | None = None
    best_item: dict[str, float] | None = None

    for threshold in grid:
        pred = (p >= threshold).astype(int)
        biz = compute_business_metrics(y, pred)
        precision = float(biz["bad_rate_in_reject_precision"])
        recall = float(biz["default_capture_recall"])
        reject_rate = float(biz["reject_rate"])
        if reject_rate < rr_min or reject_rate > rr_max:
            continue
        if precision < floor_p or recall < floor_r:
            continue

        primary = precision if threshold_objective == "precision_at_reject_rate_bounds" else recall
        secondary = recall if threshold_objective == "precision_at_reject_rate_bounds" else precision
        distance_to_target = abs(reject_rate - rr_target)
        item_tuple = (
            primary,
            secondary,
            -distance_to_target,
            -reject_rate,
            float(threshold),
        )
        if best_tuple is None or item_tuple > best_tuple:
            best_tuple = item_tuple
            best_item = {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "reject_rate": reject_rate,
            }

    if best_item is not None:
        return {
            "best_threshold": float(best_item["threshold"]),
            "best_score": float(
                best_item["precision"]
                if threshold_objective == "precision_at_reject_rate_bounds"
                else best_item["recall"]
            ),
            "metric": metric_name,
            "precision_floor": floor_p,
            "recall_floor": floor_r,
            "min_reject_rate": rr_min,
            "max_reject_rate": rr_max,
            "target_reject_rate": rr_target,
            "constraint_satisfied": True,
            "fallback_metric": None,
            "validation_precision": float(best_item["precision"]),
            "validation_recall": float(best_item["recall"]),
            "validation_reject_rate": float(best_item["reject_rate"]),
        }

    # Fallback 1: keep reject-rate bounds, relax precision floor.
    best_tuple = None
    best_item = None
    for threshold in grid:
        pred = (p >= threshold).astype(int)
        biz = compute_business_metrics(y, pred)
        precision = float(biz["bad_rate_in_reject_precision"])
        recall = float(biz["default_capture_recall"])
        reject_rate = float(biz["reject_rate"])
        if reject_rate < rr_min or reject_rate > rr_max:
            continue
        if recall < floor_r:
            continue

        primary = precision if threshold_objective == "precision_at_reject_rate_bounds" else recall
        secondary = recall if threshold_objective == "precision_at_reject_rate_bounds" else precision
        distance_to_target = abs(reject_rate - rr_target)
        item_tuple = (
            primary,
            secondary,
            -distance_to_target,
            -reject_rate,
            float(threshold),
        )
        if best_tuple is None or item_tuple > best_tuple:
            best_tuple = item_tuple
            best_item = {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "reject_rate": reject_rate,
            }

    if best_item is not None:
        return {
            "best_threshold": float(best_item["threshold"]),
            "best_score": float(
                best_item["precision"]
                if threshold_objective == "precision_at_reject_rate_bounds"
                else best_item["recall"]
            ),
            "metric": f"{metric_name}_fallback_without_precision_floor",
            "precision_floor": floor_p,
            "recall_floor": floor_r,
            "min_reject_rate": rr_min,
            "max_reject_rate": rr_max,
            "target_reject_rate": rr_target,
            "constraint_satisfied": False,
            "fallback_metric": "drop_precision_floor_keep_reject_bounds",
            "validation_precision": float(best_item["precision"]),
            "validation_recall": float(best_item["recall"]),
            "validation_reject_rate": float(best_item["reject_rate"]),
        }

    # Fallback 2: keep reject-rate bounds, relax both floors.
    best_tuple = None
    best_item = None
    for threshold in grid:
        pred = (p >= threshold).astype(int)
        biz = compute_business_metrics(y, pred)
        precision = float(biz["bad_rate_in_reject_precision"])
        recall = float(biz["default_capture_recall"])
        reject_rate = float(biz["reject_rate"])
        if reject_rate < rr_min or reject_rate > rr_max:
            continue

        primary = precision if threshold_objective == "precision_at_reject_rate_bounds" else recall
        secondary = recall if threshold_objective == "precision_at_reject_rate_bounds" else precision
        distance_to_target = abs(reject_rate - rr_target)
        item_tuple = (
            primary,
            secondary,
            -distance_to_target,
            -reject_rate,
            float(threshold),
        )
        if best_tuple is None or item_tuple > best_tuple:
            best_tuple = item_tuple
            best_item = {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "reject_rate": reject_rate,
            }

    if best_item is not None:
        return {
            "best_threshold": float(best_item["threshold"]),
            "best_score": float(
                best_item["precision"]
                if threshold_objective == "precision_at_reject_rate_bounds"
                else best_item["recall"]
            ),
            "metric": f"{metric_name}_fallback_without_precision_or_recall_floor",
            "precision_floor": floor_p,
            "recall_floor": floor_r,
            "min_reject_rate": rr_min,
            "max_reject_rate": rr_max,
            "target_reject_rate": rr_target,
            "constraint_satisfied": False,
            "fallback_metric": "drop_precision_and_recall_floor_keep_reject_bounds",
            "validation_precision": float(best_item["precision"]),
            "validation_recall": float(best_item["recall"]),
            "validation_reject_rate": float(best_item["reject_rate"]),
        }

    fallback_metric = (
        "recall_at_precision_floor" if threshold_objective == "recall_at_precision_floor" else "precision"
    )
    fallback = optimize_binary_threshold(y, p, metric=fallback_metric, precision_floor=floor_p)
    return {
        "best_threshold": float(fallback["best_threshold"]),
        "best_score": float(fallback["best_score"]),
        "metric": (
            "recall_at_precision_floor_fallback_no_reject_bound_solution"
            if threshold_objective == "recall_at_precision_floor"
            else "precision_fallback_no_reject_bound_solution"
        ),
        "precision_floor": floor_p,
        "recall_floor": floor_r,
        "min_reject_rate": rr_min,
        "max_reject_rate": rr_max,
        "target_reject_rate": rr_target,
        "constraint_satisfied": False,
        "fallback_metric": fallback_metric,
        "validation_precision": float("nan"),
        "validation_recall": float("nan"),
        "validation_reject_rate": float("nan"),
    }


def ensure_prob_range(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if np.nanmin(arr) < 0.0 or np.nanmax(arr) > 1.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    return arr


def stratified_train_val_test_split(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    max_attempts: int = 200,
    rate_tolerance: float = 0.01,
    strict_rate_match: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if not 0.0 < float(test_size) < 0.5:
        raise ValueError("test_size must be in (0, 0.5).")
    if not 0.0 < float(val_size) < 0.5:
        raise ValueError("val_size must be in (0, 0.5).")
    if int(max_attempts) <= 0:
        raise ValueError("max_attempts must be a positive integer.")
    if float(rate_tolerance) < 0.0:
        raise ValueError("rate_tolerance must be >= 0.")

    y = df[target_col].astype(int)
    global_rate = float(y.mean()) if len(y) > 0 else float("nan")
    tol = float(rate_tolerance)

    best_item: dict[str, Any] | None = None
    for attempt in range(int(max_attempts)):
        seed = int(random_state) + attempt
        train_val_df, test_df = train_test_split(
            df,
            test_size=float(test_size),
            stratify=y,
            random_state=seed,
        )
        train_fit_df, val_df = train_test_split(
            train_val_df,
            test_size=float(val_size),
            stratify=train_val_df[target_col].astype(int),
            random_state=seed,
        )

        train_rate = float(train_fit_df[target_col].astype(int).mean())
        val_rate = float(val_df[target_col].astype(int).mean())
        test_rate = float(test_df[target_col].astype(int).mean())
        max_gap = float(max(abs(train_rate - global_rate), abs(val_rate - global_rate), abs(test_rate - global_rate)))
        pair_gap = float(max(abs(train_rate - val_rate), abs(train_rate - test_rate), abs(val_rate - test_rate)))

        score = (max_gap, pair_gap)
        if best_item is None or score < best_item["score"]:
            best_item = {
                "train_fit_df": train_fit_df,
                "val_df": val_df,
                "test_df": test_df,
                "seed": seed,
                "attempt": attempt + 1,
                "score": score,
                "train_rate": train_rate,
                "val_rate": val_rate,
                "test_rate": test_rate,
                "max_gap": max_gap,
                "pair_gap": pair_gap,
            }

        if max_gap <= tol and pair_gap <= tol:
            break

    if best_item is None:
        raise RuntimeError("Failed to build stratified split.")

    train_fit_df = best_item["train_fit_df"]
    val_df = best_item["val_df"]
    test_df = best_item["test_df"]
    constraint_satisfied = bool(best_item["max_gap"] <= tol and best_item["pair_gap"] <= tol)
    if strict_rate_match and not constraint_satisfied:
        raise ValueError(
            f"Could not satisfy rate_tolerance={tol:.6f} after {int(max_attempts)} attempts; "
            f"best_max_gap={best_item['max_gap']:.6f}, best_pair_gap={best_item['pair_gap']:.6f}."
        )

    def _dist(frame: pd.DataFrame) -> dict[str, Any]:
        s = frame[target_col].astype(int)
        return {
            "rows": int(len(frame)),
            "positive_rate": float(s.mean()) if len(frame) > 0 else float("nan"),
            "class_counts": {str(k): int(v) for k, v in s.value_counts().to_dict().items()},
        }

    summary = {
        "mode": "stratified_random_split",
        "test_size": float(test_size),
        "val_size_within_train": float(val_size),
        "random_state": int(random_state),
        "max_attempts": int(max_attempts),
        "attempts_used": int(best_item["attempt"]),
        "selected_seed": int(best_item["seed"]),
        "rate_tolerance": tol,
        "rate_match_satisfied": constraint_satisfied,
        "best_max_gap_vs_full": float(best_item["max_gap"]),
        "best_max_pair_gap": float(best_item["pair_gap"]),
        "full": _dist(df),
        "train_fit": _dist(train_fit_df),
        "validation": _dist(val_df),
        "test": _dist(test_df),
    }
    return (
        train_fit_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        summary,
    )


def materialize_or_load_stratified_split(
    df: pd.DataFrame,
    *,
    split_dir: Path,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train_fit.csv"
    val_path = split_dir / "validation.csv"
    test_path = split_dir / "test.csv"
    summary_path = split_dir / "split_summary.json"

    can_load = (
        (not force_rebuild)
        and train_path.exists()
        and val_path.exists()
        and test_path.exists()
        and summary_path.exists()
    )
    if can_load:
        train_df = pd.read_csv(train_path, low_memory=False)
        val_df = pd.read_csv(val_path, low_memory=False)
        test_df = pd.read_csv(test_path, low_memory=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if SHARED_ROW_ID_COL not in train_df.columns or SHARED_ROW_ID_COL not in val_df.columns or SHARED_ROW_ID_COL not in test_df.columns:
            # Legacy split files without row-id cannot be reliably shared across pipelines.
            can_load = False
        else:
            return train_df, val_df, test_df, summary

    base_df = df.copy().reset_index(drop=True)
    if SHARED_ROW_ID_COL not in base_df.columns:
        base_df[SHARED_ROW_ID_COL] = np.arange(len(base_df), dtype=int)

    train_df, val_df, test_df, summary = stratified_train_val_test_split(
        base_df,
        target_col=target_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    summary["shared_split_dir"] = str(split_dir)
    summary["shared_row_id_col"] = SHARED_ROW_ID_COL

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    save_json(summary_path, summary)
    return train_df, val_df, test_df, summary


def threshold_by_target_reject_rate(
    y_prob: np.ndarray,
    *,
    target_reject_rate: float,
) -> dict[str, float]:
    p = ensure_prob_range(y_prob)
    rr = float(max(0.0, min(1.0, target_reject_rate)))
    if p.size == 0:
        return {"threshold": 0.5, "target_reject_rate": rr, "actual_reject_rate": 0.0}
    if rr <= 0.0:
        threshold = float(np.max(p) + 1e-12)
        pred = np.zeros_like(p, dtype=int)
    elif rr >= 1.0:
        threshold = float(np.min(p) - 1e-12)
        pred = np.ones_like(p, dtype=int)
    else:
        threshold = float(np.quantile(p, 1.0 - rr))
        pred = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "target_reject_rate": rr,
        "actual_reject_rate": float(pred.mean()),
    }


def evaluate_fixed_reject_rate(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    *,
    target_reject_rate: float,
    threshold: float | None = None,
) -> dict[str, Any]:
    y = pd.Series(y_true).astype(int)
    p = ensure_prob_range(y_prob)
    if threshold is None:
        th_info = threshold_by_target_reject_rate(p, target_reject_rate=target_reject_rate)
        used_threshold = float(th_info["threshold"])
    else:
        used_threshold = float(threshold)
    pred = (p >= used_threshold).astype(int)
    cls_metrics = compute_metrics(y, pred, p)
    biz_metrics = compute_business_metrics(y, pred)
    rr_target = float(max(0.0, min(1.0, target_reject_rate)))
    precision = float(biz_metrics["bad_rate_in_reject_precision"])
    return {
        "threshold": used_threshold,
        "target_reject_rate": rr_target,
        "actual_reject_rate": float(biz_metrics["reject_rate"]),
        "precision_at_target_reject_rate": precision,
        "recall_at_target_reject_rate": float(biz_metrics["default_capture_recall"]),
        "approval_bad_rate": float(biz_metrics["bad_rate_in_approve"]),
        "lift_at_target_reject_rate": (precision / rr_target) if rr_target > 0 else float("nan"),
        "classification_metrics": cls_metrics,
        "business_metrics": biz_metrics,
    }


def build_validation_to_test_reject_curve(
    *,
    y_val: pd.Series | np.ndarray,
    p_val: np.ndarray,
    y_test: pd.Series | np.ndarray,
    p_test: np.ndarray,
    reject_rates: list[float] | None = None,
) -> pd.DataFrame:
    rates = reject_rates or [i / 100.0 for i in range(5, 51, 5)]
    rows: list[dict[str, float]] = []
    for rr in rates:
        th_info = threshold_by_target_reject_rate(p_val, target_reject_rate=float(rr))
        threshold = float(th_info["threshold"])
        val_eval = evaluate_fixed_reject_rate(
            y_val,
            p_val,
            target_reject_rate=float(rr),
            threshold=threshold,
        )
        test_eval = evaluate_fixed_reject_rate(
            y_test,
            p_test,
            target_reject_rate=float(rr),
            threshold=threshold,
        )
        rows.append(
            {
                "target_reject_rate": float(rr),
                "threshold_from_validation": threshold,
                "validation_reject_rate": float(val_eval["actual_reject_rate"]),
                "validation_precision": float(val_eval["precision_at_target_reject_rate"]),
                "validation_recall": float(val_eval["recall_at_target_reject_rate"]),
                "test_reject_rate": float(test_eval["actual_reject_rate"]),
                "test_precision": float(test_eval["precision_at_target_reject_rate"]),
                "test_recall": float(test_eval["recall_at_target_reject_rate"]),
                "test_approval_bad_rate": float(test_eval["approval_bad_rate"]),
                "test_lift": float(test_eval["lift_at_target_reject_rate"]),
            }
        )
    return pd.DataFrame(rows)


def downsample_binary_equal(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    random_state: int = 42,
    pos_cap: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    pos_df = df[df[target_col].astype(int) == 1]
    neg_df = df[df[target_col].astype(int) == 0]
    if pos_df.empty or neg_df.empty:
        raise ValueError("Both positive and negative samples are required for equal downsample.")

    n = min(len(pos_df), len(neg_df))
    if pos_cap is not None and int(pos_cap) > 0:
        n = min(n, int(pos_cap))

    pos_s = pos_df.sample(n=n, random_state=random_state, replace=False)
    neg_s = neg_df.sample(n=n, random_state=random_state, replace=False)
    out = (
        pd.concat([pos_s, neg_s], axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    summary = {
        "before_total": int(len(df)),
        "before_pos": int(len(pos_df)),
        "before_neg": int(len(neg_df)),
        "after_total": int(len(out)),
        "after_pos": int((out[target_col].astype(int) == 1).sum()),
        "after_neg": int((out[target_col].astype(int) == 0).sum()),
    }
    return out, summary


def _non_missing_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        text = series.astype(str).str.strip()
        return series.notna() & (text != "") & (text.str.lower() != "nan")
    return series.notna()


def apply_dataset_cleaning(
    df: pd.DataFrame,
    *,
    required_non_null_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
    drop_constant_cols: bool = True,
    max_col_missing_ratio: float | None = None,
    protect_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    protect = set(protect_cols or [])
    report: dict[str, Any] = {
        "rows_before": int(len(out)),
        "rows_after": int(len(out)),
        "rows_dropped_by_required_non_null": 0,
        "required_non_null_cols_used": [],
        "dropped_columns_manual": [],
        "dropped_columns_missing_ratio": [],
        "dropped_columns_constant": [],
    }

    required_cols = [c for c in (required_non_null_cols or []) if c in out.columns]
    if required_cols:
        mask = pd.Series(True, index=out.index)
        for col in required_cols:
            mask &= _non_missing_mask(out[col])
        before = len(out)
        out = out[mask].copy()
        report["rows_dropped_by_required_non_null"] = int(before - len(out))
        report["required_non_null_cols_used"] = required_cols

    manual_drop = [c for c in (drop_cols or []) if c in out.columns and c not in protect]
    if manual_drop:
        out = out.drop(columns=manual_drop, errors="ignore")
        report["dropped_columns_manual"] = sorted(manual_drop)

    if max_col_missing_ratio is not None and 0.0 <= float(max_col_missing_ratio) <= 1.0 and len(out) > 0:
        miss_ratio = out.isna().mean()
        miss_drop = [
            c
            for c in out.columns
            if c not in protect and float(miss_ratio.get(c, 0.0)) > float(max_col_missing_ratio)
        ]
        if miss_drop:
            out = out.drop(columns=miss_drop, errors="ignore")
            report["dropped_columns_missing_ratio"] = sorted(miss_drop)

    if drop_constant_cols and len(out) > 0:
        const_cols: list[str] = []
        for col in out.columns:
            if col in protect:
                continue
            if out[col].nunique(dropna=False) <= 1:
                const_cols.append(col)
        if const_cols:
            out = out.drop(columns=const_cols, errors="ignore")
            report["dropped_columns_constant"] = sorted(const_cols)

    out = out.reset_index(drop=True)
    report["rows_after"] = int(len(out))
    return out, report


def base_filter_chunk(chunk: pd.DataFrame, require_text: bool) -> pd.DataFrame:
    mask = chunk[STATUS_COL].isin(ALLOWED_STATUS)
    if require_text:
        mask = mask & chunk[RAW_TEXT_COL].notna() & (chunk[RAW_TEXT_COL].astype(str).str.strip() != "")
    filtered = chunk[mask].copy()
    if filtered.empty:
        return filtered
    filtered[TARGET_COL] = (filtered[STATUS_COL] == "Charged Off").astype(int)
    return filtered


def sample_prepared_df(
    input_csv: Path,
    sample_size: int | None,
    random_state: int,
    *,
    strategy: str = "preserve_ratio",
) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    if TARGET_COL not in df.columns or TEXT_COL not in df.columns:
        raise ValueError(f"{input_csv} must include {TARGET_COL} and {TEXT_COL}.")

    if strategy == "equal_class_downsample":
        pos = df[df[TARGET_COL].astype(int) == 1]
        neg = df[df[TARGET_COL].astype(int) == 0]
        if pos.empty or neg.empty:
            return df.reset_index(drop=True)
        base_n = min(len(pos), len(neg))
        if sample_size is not None and sample_size > 0:
            base_n = min(base_n, int(sample_size))
        pos_sample = pos.sample(n=base_n, random_state=random_state, replace=False)
        neg_sample = neg.sample(n=base_n, random_state=random_state, replace=False)
        df = (
            pd.concat([pos_sample, neg_sample], axis=0)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )
        return df

    if strategy != "preserve_ratio":
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        pos = df[df[TARGET_COL] == 1]
        neg = df[df[TARGET_COL] == 0]
        pos_n = max(1, int(sample_size * (len(pos) / len(df))))
        neg_n = max(1, sample_size - pos_n)
        pos_sample = pos.sample(n=min(pos_n, len(pos)), random_state=random_state)
        neg_sample = neg.sample(n=min(neg_n, len(neg)), random_state=random_state)
        df = pd.concat([pos_sample, neg_sample], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def parse_risk_level(text: str) -> str:
    t = text.lower()
    if '"risk":"high"' in t or '"risk": "high"' in t or "risk: high" in t or "high risk" in t:
        return "high"
    if '"risk":"low"' in t or '"risk": "low"' in t or "risk: low" in t or "low risk" in t:
        return "low"
    return "unknown"


def split_by_issue_year(
    df: pd.DataFrame,
    *,
    train_year_start: int = 2007,
    train_year_end: int = 2013,
    test_year_start: int = 2014,
    test_year_end: int = 2016,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ISSUE_YEAR_COL not in df.columns:
        raise ValueError(f"Missing required column for time split: {ISSUE_YEAR_COL}")

    year = pd.to_numeric(df[ISSUE_YEAR_COL], errors="coerce")
    train_df = df[(year >= train_year_start) & (year <= train_year_end)].copy()
    test_df = df[(year >= test_year_start) & (year <= test_year_end)].copy()
    return train_df, test_df


def make_time_split_with_train_neg_downsample(
    df: pd.DataFrame,
    *,
    random_state: int,
    train_year_start: int = 2007,
    train_year_end: int = 2013,
    test_year_start: int = 2014,
    test_year_end: int = 2016,
    target_col: str = TARGET_COL,
    train_pos_cap: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_df, test_df = split_by_issue_year(
        df,
        train_year_start=train_year_start,
        train_year_end=train_year_end,
        test_year_start=test_year_start,
        test_year_end=test_year_end,
    )
    if train_df.empty:
        raise ValueError("Train split is empty after year filtering.")
    if test_df.empty:
        raise ValueError("Test split is empty after year filtering.")
    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    train_pos = train_df[train_df[target_col].astype(int) == 1]
    train_neg = train_df[train_df[target_col].astype(int) == 0]

    if train_pos.empty or train_neg.empty:
        raise ValueError("Train split must include both positive and negative samples.")

    n_pos = int(len(train_pos))
    if train_pos_cap is not None and train_pos_cap > 0:
        n_pos = min(n_pos, int(train_pos_cap))

    if len(train_neg) < n_pos:
        raise ValueError(
            f"Not enough negatives in train split: need {n_pos}, got {len(train_neg)}."
        )

    if n_pos < len(train_pos):
        train_pos = train_pos.sample(n=n_pos, random_state=random_state, replace=False)
    train_neg = train_neg.sample(n=n_pos, random_state=random_state, replace=False)

    train_balanced = (
        pd.concat([train_pos, train_neg], axis=0)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    summary = {
        "train_total_before": int(len(train_df)),
        "train_pos_before": int((train_df[target_col].astype(int) == 1).sum()),
        "train_neg_before": int((train_df[target_col].astype(int) == 0).sum()),
        "train_pos_after": int((train_balanced[target_col].astype(int) == 1).sum()),
        "train_neg_after": int((train_balanced[target_col].astype(int) == 0).sum()),
        "test_total": int(len(test_df)),
        "test_pos": int((test_df[target_col].astype(int) == 1).sum()),
        "test_neg": int((test_df[target_col].astype(int) == 0).sum()),
    }
    return train_balanced, test_df, summary


def balance_train_binary_equal_neg(
    train_df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    neg_df = train_df[train_df[target_col].astype(int) == 0]
    pos_df = train_df[train_df[target_col].astype(int) == 1]

    before = {
        "neg_before": int(len(neg_df)),
        "pos_before": int(len(pos_df)),
    }

    if len(neg_df) == 0 or len(pos_df) == 0:
        after = {
            "neg_after": int(len(neg_df)),
            "pos_after": int(len(pos_df)),
        }
        return train_df.copy(), {**before, **after}

    # Keep all negatives, oversample positives to the same count as negatives.
    pos_up = pos_df.sample(n=len(neg_df), replace=True, random_state=random_state)
    balanced = (
        pd.concat([neg_df, pos_up], axis=0)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    after = {
        "neg_after": int((balanced[target_col].astype(int) == 0).sum()),
        "pos_after": int((balanced[target_col].astype(int) == 1).sum()),
    }
    return balanced, {**before, **after}
