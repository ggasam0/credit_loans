from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from workflow_common import (
    BASE_USECOLS,
    CATEGORICAL_CANDIDATES,
    ISSUE_COL,
    ISSUE_YEAR_COL,
    NUMERIC_CANDIDATES,
    RAW_TEXT_COL,
    STATUS_COL,
    TARGET_COL,
    TEXT_COL,
    apply_dataset_cleaning,
    base_filter_chunk,
    clean_text,
    compute_metrics,
    downsample_binary_equal,
    optimize_binary_threshold,
    parse_issue_year,
    parse_emp_length,
    save_json,
    make_time_split_with_train_neg_downsample,
    to_term_months,
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


def prepare_ml_dataset(
    input_csv: Path,
    raw_output_csv: Path,
    processed_output_csv: Path,
    *,
    chunksize: int,
    max_rows: int | None,
    max_chunks: int | None,
) -> dict[str, Any]:
    raw_output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_output_csv.parent.mkdir(parents=True, exist_ok=True)
    if raw_output_csv.exists():
        raw_output_csv.unlink()
    if processed_output_csv.exists():
        processed_output_csv.unlink()

    seen = 0
    kept = 0
    chunk_id = 0

    reader = pd.read_csv(
        input_csv,
        usecols=lambda c: c in BASE_USECOLS,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in reader:
        chunk_id += 1
        seen += len(chunk)
        filtered = base_filter_chunk(chunk, require_text=True)
        if filtered.empty:
            if max_chunks is not None and chunk_id >= max_chunks:
                break
            continue

        if max_rows is not None:
            remain = max_rows - kept
            if remain <= 0:
                break
            filtered = filtered.iloc[:remain].copy()

        raw_chunk = filtered.copy()
        raw_chunk.to_csv(raw_output_csv, mode="a", index=False, header=not raw_output_csv.exists() or kept == 0)

        processed = filtered.copy()
        processed[TEXT_COL] = processed[RAW_TEXT_COL].map(clean_text)
        processed["emp_length_years"] = processed.get("emp_length", pd.Series(dtype=object)).map(parse_emp_length)
        processed["term_months"] = processed.get("term", pd.Series(dtype=object)).map(to_term_months)
        processed[ISSUE_YEAR_COL] = processed.get(ISSUE_COL, pd.Series(dtype=object)).map(parse_issue_year)
        processed["fico_mean"] = (
            pd.to_numeric(processed.get("fico_range_low"), errors="coerce")
            + pd.to_numeric(processed.get("fico_range_high"), errors="coerce")
        ) / 2.0
        processed.to_csv(
            processed_output_csv,
            mode="a",
            index=False,
            header=not processed_output_csv.exists() or kept == 0,
        )

        kept += len(filtered)
        if max_rows is not None and kept >= max_rows:
            break
        if max_chunks is not None and chunk_id >= max_chunks:
            break

    summary = {
        "input_csv": str(input_csv),
        "raw_output_csv": str(raw_output_csv),
        "processed_output_csv": str(processed_output_csv),
        "rows_seen": seen,
        "rows_kept_before_cleaning": kept,
    }

    if processed_output_csv.exists() and kept > 0:
        processed_all = pd.read_csv(processed_output_csv, low_memory=False)
        cleaned, cleaning_report = apply_dataset_cleaning(
            processed_all,
            required_non_null_cols=[
                TARGET_COL,
                TEXT_COL,
                ISSUE_YEAR_COL,
                "loan_amnt",
                "int_rate",
                "annual_inc",
                "dti",
                "fico_mean",
                "revol_util",
            ],
            drop_cols=[
                RAW_TEXT_COL,
                STATUS_COL,
                ISSUE_COL,
                "application_type",
                "sub_grade",
                "term",
                "emp_length",
                "fico_range_low",
                "fico_range_high",
            ],
            drop_constant_cols=True,
            max_col_missing_ratio=0.4,
            protect_cols=[TARGET_COL, TEXT_COL, ISSUE_YEAR_COL],
        )
        cleaned.to_csv(processed_output_csv, index=False)
        summary["rows_after_cleaning"] = int(len(cleaned))
        summary["cleaning_report"] = cleaning_report
    else:
        summary["rows_after_cleaning"] = int(kept)
        summary["cleaning_report"] = {}

    save_json(processed_output_csv.with_suffix(".summary.json"), summary)
    return summary


def build_preprocessor(columns: list[str], include_text: bool, text_max_features: int) -> ColumnTransformer:
    numeric_cols = [c for c in (NUMERIC_CANDIDATES + ["emp_length_years", "term_months", "fico_mean", "bert_score"]) if c in columns]
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in columns]

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical_cols,
            )
        )

    if include_text and TEXT_COL in columns:
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
        raise ValueError("No usable feature columns found for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_estimator(model_name: str, *, random_state: int, n_jobs: int, scale_pos_weight: float) -> Any:
    if model_name == "xgboost" and HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=260,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
    return LogisticRegression(max_iter=1600, class_weight="balanced", n_jobs=n_jobs, random_state=random_state)


def _fit_and_score(
    *,
    model_name: str,
    preprocessor: ColumnTransformer,
    estimator: Any,
    x_train_for_cv: pd.DataFrame,
    y_train_for_cv: pd.Series,
    x_train_fit: pd.DataFrame,
    y_train_fit: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int,
    n_jobs: int,
    precision_floor: float,
) -> tuple[Pipeline, dict[str, Any]]:
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, x_train_for_cv, y_train_for_cv, cv=cv, scoring="balanced_accuracy", n_jobs=n_jobs)

    pipe.fit(x_train_fit, y_train_fit)

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        val_prob = pipe.predict_proba(x_val)[:, 1]
    else:
        val_decision = pipe.decision_function(x_val)
        val_prob = 1 / (1 + np.exp(-val_decision))

    threshold_info = optimize_binary_threshold(
        y_val,
        val_prob,
        metric="recall_at_precision_floor",
        precision_floor=precision_floor,
    )
    best_threshold = float(threshold_info["best_threshold"])
    val_pred = (val_prob >= best_threshold).astype(int)
    val_metrics = compute_metrics(y_val, val_pred, val_prob)

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        prob = pipe.predict_proba(x_test)[:, 1]
    else:
        test_decision = pipe.decision_function(x_test)
        prob = 1 / (1 + np.exp(-test_decision))
    pred = (prob >= best_threshold).astype(int)

    return pipe, {
        "model_name": model_name,
        "cv_balanced_accuracy_mean": float(np.mean(cv_scores)),
        "cv_balanced_accuracy_std": float(np.std(cv_scores)),
        "threshold_tuning": {
            "method": "validation_search",
            "metric": threshold_info["metric"],
            "precision_floor": threshold_info.get("precision_floor"),
            "constraint_satisfied": bool(threshold_info.get("constraint_satisfied", False)),
            "fallback_metric": threshold_info.get("fallback_metric"),
            "best_threshold": best_threshold,
            "best_score": float(threshold_info["best_score"]),
            "validation_metrics_at_best_threshold": val_metrics,
        },
        "test_metrics": compute_metrics(y_test, pred, prob),
    }


def _build_text_score_proxy(
    train_text: pd.Series,
    y_train: pd.Series,
    val_text: pd.Series,
    test_text: pd.Series,
    *,
    cv_folds: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_train = train_text.fillna("").astype(str)
    text_val = val_text.fillna("").astype(str)
    text_test = test_text.fillna("").astype(str)

    proxy_pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3)),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", n_jobs=n_jobs, random_state=42)),
        ]
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(text_train), dtype=float)
    for train_idx, valid_idx in cv.split(text_train, y_train):
        proxy_pipe.fit(text_train.iloc[train_idx], y_train.iloc[train_idx])
        oof[valid_idx] = proxy_pipe.predict_proba(text_train.iloc[valid_idx])[:, 1]

    proxy_pipe.fit(text_train, y_train)
    val_score = proxy_pipe.predict_proba(text_val)[:, 1]
    test_score = proxy_pipe.predict_proba(text_test)[:, 1]
    return oof, val_score, test_score


def train_ml_models(
    input_csv: Path,
    output_dir: Path,
    *,
    sample_size: int,
    test_size: float,
    precision_floor: float,
    balance_test: bool,
    test_pos_cap: int | None,
    cv_folds: int,
    random_state: int,
    text_max_features: int,
    model_name: str,
    n_jobs: int,
    train_year_start: int,
    train_year_end: int,
    test_year_start: int,
    test_year_end: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv, low_memory=False)
    if TARGET_COL not in df.columns or TEXT_COL not in df.columns:
        raise ValueError(f"{input_csv} must include {TARGET_COL} and {TEXT_COL}.")

    train_pos_cap = sample_size if sample_size is not None and sample_size > 0 else None
    sampled_class_counts = (
        df[TARGET_COL].astype(int).value_counts(dropna=False).to_dict()
        if TARGET_COL in df.columns
        else {}
    )

    train_df, test_df, split_summary = make_time_split_with_train_neg_downsample(
        df,
        random_state=random_state,
        train_year_start=train_year_start,
        train_year_end=train_year_end,
        test_year_start=test_year_start,
        test_year_end=test_year_end,
        target_col=TARGET_COL,
        train_pos_cap=train_pos_cap,
    )
    test_balance_summary: dict[str, int] | None = None
    if balance_test:
        test_df, test_balance_summary = downsample_binary_equal(
            test_df,
            target_col=TARGET_COL,
            random_state=random_state,
            pos_cap=test_pos_cap,
        )
    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)
    x_train = train_df.drop(columns=[TARGET_COL, STATUS_COL, ISSUE_COL, ISSUE_YEAR_COL], errors="ignore")
    x_test = test_df.drop(columns=[TARGET_COL, STATUS_COL, ISSUE_COL, ISSUE_YEAR_COL], errors="ignore")
    split_mode = (
        f"time_split_train_{train_year_start}_{train_year_end}"
        f"_test_{test_year_start}_{test_year_end}"
    )

    val_size = float(test_size) if 0.0 < float(test_size) < 0.5 else 0.2
    x_train_fit, x_val, y_train_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state,
    )

    pos_rate = float(y_train_fit.mean())
    scale_pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-8)

    prep1 = build_preprocessor(columns=x_train_fit.columns.tolist(), include_text=False, text_max_features=text_max_features)
    est1 = make_estimator(model_name, random_state=random_state, n_jobs=n_jobs, scale_pos_weight=scale_pos_weight)
    m1, r1 = _fit_and_score(
        model_name=f"{model_name}_tabular",
        preprocessor=prep1,
        estimator=est1,
        x_train_for_cv=x_train_fit,
        y_train_for_cv=y_train_fit,
        x_train_fit=x_train_fit,
        y_train_fit=y_train_fit,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        precision_floor=precision_floor,
    )
    joblib.dump(m1, output_dir / "model_tabular.joblib")

    prep2 = build_preprocessor(columns=x_train_fit.columns.tolist(), include_text=True, text_max_features=text_max_features)
    est2 = make_estimator(model_name, random_state=random_state, n_jobs=n_jobs, scale_pos_weight=scale_pos_weight)
    m2, r2 = _fit_and_score(
        model_name=f"{model_name}_text_fusion",
        preprocessor=prep2,
        estimator=est2,
        x_train_for_cv=x_train_fit,
        y_train_for_cv=y_train_fit,
        x_train_fit=x_train_fit,
        y_train_fit=y_train_fit,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        precision_floor=precision_floor,
    )
    joblib.dump(m2, output_dir / "model_text_fusion.joblib")

    oof_score, val_score, test_score = _build_text_score_proxy(
        x_train_fit[TEXT_COL],
        y_train_fit,
        x_val[TEXT_COL],
        x_test[TEXT_COL],
        cv_folds=cv_folds,
        n_jobs=n_jobs,
    )
    x_train_fusion = x_train_fit.copy()
    x_val_fusion = x_val.copy()
    x_test_fusion = x_test.copy()
    x_train_fusion["bert_score"] = oof_score
    x_val_fusion["bert_score"] = val_score
    x_test_fusion["bert_score"] = test_score

    prep3 = build_preprocessor(columns=x_train_fusion.columns.tolist(), include_text=False, text_max_features=text_max_features)
    est3 = make_estimator(model_name, random_state=random_state, n_jobs=n_jobs, scale_pos_weight=scale_pos_weight)
    m3, r3 = _fit_and_score(
        model_name=f"{model_name}_fusion_text_score",
        preprocessor=prep3,
        estimator=est3,
        x_train_for_cv=x_train_fusion,
        y_train_for_cv=y_train_fit,
        x_train_fit=x_train_fusion,
        y_train_fit=y_train_fit,
        x_val=x_val_fusion,
        y_val=y_val,
        x_test=x_test_fusion,
        y_test=y_test,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        precision_floor=precision_floor,
    )
    joblib.dump(m3, output_dir / "model_fusion_text_score.joblib")

    rows = []
    for result in (r1, r2, r3):
        row = {
            "model_name": result["model_name"],
            "cv_bacc_mean": result["cv_balanced_accuracy_mean"],
            "cv_bacc_std": result["cv_balanced_accuracy_std"],
            "threshold": float(result["threshold_tuning"]["best_threshold"]),
        }
        row.update(result["test_metrics"])
        rows.append(row)
    summary_df = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    payload = {
        "input_csv": str(input_csv),
        "rows_used": int(len(df)),
        "sampling_strategy": "time_split_train_neg_downsample",
        "sampled_class_counts": {str(k): int(v) for k, v in sampled_class_counts.items()},
        "train_rows": int(len(x_train)),
        "train_fit_rows": int(len(x_train_fit)),
        "validation_rows": int(len(x_val)),
        "test_rows": int(len(x_test)),
        "precision_floor": float(precision_floor),
        "test_balance_enabled": bool(balance_test),
        "test_balance_summary": test_balance_summary,
        "train_sampling_summary": split_summary,
        "target_positive_rate_train": float(y_train_fit.mean()),
        "target_positive_rate_test": float(y_test.mean()),
        "split_mode": split_mode,
        "model_results": [r1, r2, r3],
        "metrics_summary_csv": str(output_dir / "metrics_summary.csv"),
    }
    save_json(output_dir / "training_report.json", payload)
    return payload
