from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from workflow_common import (
    BASE_USECOLS,
    CATEGORICAL_CANDIDATES,
    apply_dataset_cleaning,
    build_validation_to_test_reject_curve,
    compute_business_metrics,
    downsample_binary_equal,
    evaluate_fixed_reject_rate,
    ISSUE_COL,
    ISSUE_YEAR_COL,
    NUMERIC_CANDIDATES,
    optimize_binary_threshold_business,
    RAW_TEXT_COL,
    STATUS_COL,
    TARGET_COL,
    TEXT_COL,
    base_filter_chunk,
    clean_text,
    compute_metrics,
    make_time_split_with_train_neg_downsample,
    materialize_or_load_stratified_split,
    parse_issue_year,
    SHARED_ROW_ID_COL,
    save_json,
    split_by_issue_year,
    stratified_train_val_test_split,
    threshold_by_target_reject_rate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BCE_EMBED_PATH = str(PROJECT_ROOT / "models" / "bce-embedding-base_v1")
BERT_INPUT_COL = "bert_input"


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value).strip()
    return text if text else "missing"


def _build_bert_input_column(frame: pd.DataFrame) -> pd.Series:
    def _row_to_text(row: pd.Series) -> str:
        desc_raw = row.get(TEXT_COL)
        desc = "" if pd.isna(desc_raw) else str(desc_raw).strip()
        if not desc or desc.lower() == "nan":
            desc = "missing"
        return (
            "[Loan Application Profile]\n"
            f"Loan amount: {_format_value(row.get('loan_amnt'))}\n"
            f"Term: {_format_value(row.get('term'))}\n"
            f"Interest rate: {_format_value(row.get('int_rate'))}\n"
            f"Installment: {_format_value(row.get('installment'))}\n"
            f"Purpose: {_format_value(row.get('purpose'))}\n"
            f"Application type: {_format_value(row.get('application_type'))}\n"
            "\n"
            "[Borrower Background]\n"
            f"Annual income: {_format_value(row.get('annual_inc'))}\n"
            f"Employment length: {_format_value(row.get('emp_length'))}\n"
            f"Home ownership: {_format_value(row.get('home_ownership'))}\n"
            f"Income verification: {_format_value(row.get('verification_status'))}\n"
            f"State: {_format_value(row.get('addr_state'))}\n"
            "\n"
            "[Credit And Liability Snapshot]\n"
            f"DTI: {_format_value(row.get('dti'))}\n"
            f"FICO low: {_format_value(row.get('fico_range_low'))}\n"
            f"FICO high: {_format_value(row.get('fico_range_high'))}\n"
            f"Revolving balance: {_format_value(row.get('revol_bal'))}\n"
            f"Revolving utilization: {_format_value(row.get('revol_util'))}\n"
            f"Open accounts: {_format_value(row.get('open_acc'))}\n"
            f"Public records: {_format_value(row.get('pub_rec'))}\n"
            f"Total accounts: {_format_value(row.get('total_acc'))}\n"
            f"Grade: {_format_value(row.get('grade'))}\n"
            f"Sub grade: {_format_value(row.get('sub_grade'))}\n"
            "\n"
            "[Borrower Description]\n"
            f"{desc}\n"
        )

    return frame.apply(_row_to_text, axis=1)


def _resolve_effective_max_length(requested_max_length: int, tokenizer_max_length: Any) -> int:
    requested = int(requested_max_length)
    # Keep token-based truncation and cap to tokenizer/model limit when it is finite.
    # Some tokenizers expose a very large sentinel (for example 1e30); ignore those.
    if tokenizer_max_length is None:
        return requested
    try:
        tok_max = int(tokenizer_max_length)
    except Exception:
        return requested
    if tok_max <= 0 or tok_max > 1_000_000:
        return requested
    return min(requested, tok_max)


def _realign_split_from_base(base_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    if SHARED_ROW_ID_COL not in split_df.columns:
        return split_df.copy()
    if SHARED_ROW_ID_COL not in base_df.columns:
        raise ValueError(
            f"Shared split contains {SHARED_ROW_ID_COL} but base dataframe does not; cannot realign columns."
        )
    base = base_df.drop_duplicates(subset=[SHARED_ROW_ID_COL], keep="first").set_index(SHARED_ROW_ID_COL, drop=False)
    ids = split_df[SHARED_ROW_ID_COL].tolist()
    missing_ids = [rid for rid in ids if rid not in base.index]
    if missing_ids:
        raise ValueError(
            f"Shared split row ids not found in current base dataframe: missing={len(missing_ids)}."
        )
    aligned = base.loc[ids].reset_index(drop=True).copy()
    return aligned


def prepare_bert_dataset(
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

        filtered.to_csv(raw_output_csv, mode="a", index=False, header=not raw_output_csv.exists() or kept == 0)

        keep_cols = [c for c in [RAW_TEXT_COL, TARGET_COL, ISSUE_COL] if c in filtered.columns]
        keep_cols += [c for c in (NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES) if c in filtered.columns]
        keep_cols = list(dict.fromkeys(keep_cols))
        processed = filtered[keep_cols].copy()
        processed[TEXT_COL] = processed[RAW_TEXT_COL].map(clean_text)
        processed[ISSUE_YEAR_COL] = processed.get(ISSUE_COL, pd.Series(dtype=object)).map(parse_issue_year)
        processed = processed[processed[TEXT_COL].str.len() > 0].copy()
        processed[BERT_INPUT_COL] = _build_bert_input_column(processed)
        processed.to_csv(processed_output_csv, mode="a", index=False, header=not processed_output_csv.exists() or kept == 0)

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
            required_non_null_cols=[TARGET_COL, TEXT_COL, ISSUE_YEAR_COL, BERT_INPUT_COL],
            drop_cols=[STATUS_COL, ISSUE_COL],
            drop_constant_cols=True,
            max_col_missing_ratio=0.98,
            protect_cols=[TARGET_COL, RAW_TEXT_COL, TEXT_COL, ISSUE_YEAR_COL, BERT_INPUT_COL],
        )
        cleaned.to_csv(processed_output_csv, index=False)
        summary["rows_after_cleaning"] = int(len(cleaned))
        summary["cleaning_report"] = cleaning_report
    else:
        summary["rows_after_cleaning"] = int(kept)
        summary["cleaning_report"] = {}

    save_json(processed_output_csv.with_suffix(".summary.json"), summary)
    return summary


def evaluate_bert_embedding_local(
    input_csv: Path,
    output_dir: Path,
    *,
    embedding_model_path: Path,
    sample_size: int,
    test_size: float,
    precision_floor: float,
    balance_test: bool,
    test_pos_cap: int | None,
    random_state: int,
    device: str,
    max_length: int,
    train_year_start: int = 2007,
    train_year_end: int = 2013,
    test_year_start: int = 2014,
    test_year_end: int = 2016,
    split_mode: str = "stratified",
    threshold_policy: str = "fixed_reject_rate",
    threshold_objective: str = "precision_at_reject_rate_bounds",
    recall_floor: float = 0.03,
    min_reject_rate: float = 0.03,
    max_reject_rate: float = 0.20,
    target_reject_rate: float = 0.35,
    rebuild_bert_input: bool = True,
    downsample_train_neg: bool = True,
    shared_split_dir: Path | None = None,
    force_rebuild_shared_split: bool = False,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

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
    if rebuild_bert_input or BERT_INPUT_COL not in df.columns:
        # Rebuild every run by default; standalone workflows can opt out and provide their own BERT input.
        df[BERT_INPUT_COL] = _build_bert_input_column(df)
    input_col = BERT_INPUT_COL if BERT_INPUT_COL in df.columns else TEXT_COL

    val_size = float(test_size) if 0.0 < float(test_size) < 0.5 else 0.2
    if split_mode == "stratified":
        if shared_split_dir is not None and str(shared_split_dir).strip():
            train_fit_df, val_df, test_df, split_summary = materialize_or_load_stratified_split(
                df,
                split_dir=Path(shared_split_dir),
                target_col=TARGET_COL,
                test_size=val_size,
                val_size=val_size,
                random_state=random_state,
                force_rebuild=bool(force_rebuild_shared_split),
            )
            train_fit_df = _realign_split_from_base(df, train_fit_df)
            val_df = _realign_split_from_base(df, val_df)
            test_df = _realign_split_from_base(df, test_df)
        else:
            train_fit_df, val_df, test_df, split_summary = stratified_train_val_test_split(
                df,
                target_col=TARGET_COL,
                test_size=val_size,
                val_size=val_size,
                random_state=random_state,
            )
        train_df = pd.concat([train_fit_df, val_df], axis=0).reset_index(drop=True)
        sampling_strategy = "stratified_random_split"
    elif downsample_train_neg:
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
        sampling_strategy = "time_split_train_neg_downsample"
    else:
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
        split_summary = {
            "train_total_before": int(len(train_df)),
            "train_pos_before": int((train_df[TARGET_COL].astype(int) == 1).sum()),
            "train_neg_before": int((train_df[TARGET_COL].astype(int) == 0).sum()),
            "train_pos_after": int((train_df[TARGET_COL].astype(int) == 1).sum()),
            "train_neg_after": int((train_df[TARGET_COL].astype(int) == 0).sum()),
            "test_total": int(len(test_df)),
            "test_pos": int((test_df[TARGET_COL].astype(int) == 1).sum()),
            "test_neg": int((test_df[TARGET_COL].astype(int) == 0).sum()),
        }
        sampling_strategy = "time_split_no_train_downsample"
    test_balance_summary: dict[str, int] | None = None
    if balance_test:
        test_df, test_balance_summary = downsample_binary_equal(
            test_df,
            target_col=TARGET_COL,
            random_state=random_state,
            pos_cap=test_pos_cap,
        )
    x_train = train_df[[input_col]]
    x_test = test_df[[input_col]]
    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)
    split_mode_name = (
        "stratified_random_split"
        if split_mode == "stratified"
        else (
            f"time_split_train_{train_year_start}_{train_year_end}"
            f"_test_{test_year_start}_{test_year_end}"
        )
    )

    embed_model = SentenceTransformer(str(embedding_model_path), device=device)
    tokenizer_max_length = getattr(getattr(embed_model, "tokenizer", None), "model_max_length", None)
    effective_max_length = _resolve_effective_max_length(max_length, tokenizer_max_length)
    embed_model.max_seq_length = int(effective_max_length)
    if split_mode == "stratified":
        x_train_fit = train_fit_df[[input_col]]
        x_val = val_df[[input_col]]
        y_train_fit = train_fit_df[TARGET_COL].astype(int)
        y_val = val_df[TARGET_COL].astype(int)
    else:
        x_train_fit, x_val, y_train_fit, y_val = train_test_split(
            x_train,
            y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=random_state,
        )

    emb_train = embed_model.encode(
        x_train_fit[input_col].astype(str).tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb_val = embed_model.encode(
        x_val[input_col].astype(str).tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb_test = embed_model.encode(
        x_test[input_col].astype(str).tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    clf.fit(emb_train, y_train_fit.values)

    val_prob = clf.predict_proba(emb_val)[:, 1]
    if threshold_policy == "fixed_reject_rate":
        fixed_threshold = threshold_by_target_reject_rate(
            val_prob,
            target_reject_rate=target_reject_rate,
        )
        threshold_info = {
            "metric": "fixed_reject_rate_quantile",
            "best_threshold": float(fixed_threshold["threshold"]),
            "best_score": float("nan"),
            "precision_floor": None,
            "recall_floor": None,
            "min_reject_rate": None,
            "max_reject_rate": None,
            "target_reject_rate": float(target_reject_rate),
            "constraint_satisfied": True,
            "fallback_metric": None,
            "validation_reject_rate": float(fixed_threshold["actual_reject_rate"]),
        }
    else:
        threshold_info = optimize_binary_threshold_business(
            y_val,
            val_prob,
            threshold_objective=threshold_objective,
            precision_floor=precision_floor,
            recall_floor=recall_floor,
            min_reject_rate=min_reject_rate,
            max_reject_rate=max_reject_rate,
            target_reject_rate=target_reject_rate,
        )
    best_threshold = float(threshold_info["best_threshold"])
    val_pred = (val_prob >= best_threshold).astype(int)
    val_metrics = compute_metrics(y_val, val_pred, val_prob)
    val_business = compute_business_metrics(y_val, val_pred)
    val_fixed = evaluate_fixed_reject_rate(
        y_val,
        val_prob,
        target_reject_rate=target_reject_rate,
        threshold=best_threshold,
    )

    prob = clf.predict_proba(emb_test)[:, 1]
    pred = (prob >= best_threshold).astype(int)
    metrics = compute_metrics(y_test, pred, prob)
    test_business = compute_business_metrics(y_test, pred)
    test_fixed = evaluate_fixed_reject_rate(
        y_test,
        prob,
        target_reject_rate=target_reject_rate,
        threshold=best_threshold,
    )
    gains_lift_curve = build_validation_to_test_reject_curve(
        y_val=y_val,
        p_val=val_prob,
        y_test=y_test,
        p_test=prob,
        reject_rates=[i / 100.0 for i in range(5, 51, 5)],
    )
    gains_lift_curve_csv = output_dir / "bert_embedding_gains_lift_curve.csv"
    gains_lift_curve.to_csv(gains_lift_curve_csv, index=False)

    pred_out = x_test.copy()
    pred_out[TARGET_COL] = y_test.values
    pred_out["embed_prob"] = prob
    pred_out["embed_pred"] = pred
    pred_out["threshold_used"] = best_threshold
    pred_csv = output_dir / "bert_embedding_eval_predictions.csv"
    pred_out.to_csv(pred_csv, index=False)

    report = {
        "input_csv": str(input_csv),
        "source_rows": int(len(df)),
        "train_pos_cap": None if train_pos_cap is None else int(train_pos_cap),
        "sampling_strategy": sampling_strategy,
        "sampled_class_counts": {str(k): int(v) for k, v in sampled_class_counts.items()},
        "train_rows": int(len(x_train)),
        "train_fit_rows": int(len(x_train_fit)),
        "validation_rows": int(len(x_val)),
        "test_rows": int(len(x_test)),
        "split_mode": split_mode_name,
        "train_sampling_summary": split_summary,
        "threshold_tuning": {
            "method": "validation_search",
            "threshold_policy": threshold_policy,
            "metric": threshold_info["metric"],
            "precision_floor": threshold_info.get("precision_floor"),
            "recall_floor": threshold_info.get("recall_floor"),
            "min_reject_rate": threshold_info.get("min_reject_rate"),
            "max_reject_rate": threshold_info.get("max_reject_rate"),
            "target_reject_rate": threshold_info.get("target_reject_rate"),
            "constraint_satisfied": bool(threshold_info.get("constraint_satisfied", False)),
            "fallback_metric": threshold_info.get("fallback_metric"),
            "best_threshold": best_threshold,
            "best_score": float(threshold_info["best_score"]),
            "validation_precision": float(threshold_info.get("validation_precision", val_business["bad_rate_in_reject_precision"])),
            "validation_recall": float(threshold_info.get("validation_recall", val_business["default_capture_recall"])),
            "validation_reject_rate": float(threshold_info.get("validation_reject_rate", val_business["reject_rate"])),
            "validation_metrics_at_best_threshold": val_metrics,
            "validation_business_metrics_at_best_threshold": val_business,
            "validation_fixed_metrics_at_target_reject_rate": val_fixed,
        },
        "test_balance_enabled": bool(balance_test),
        "test_balance_summary": test_balance_summary,
        "bert_input_column": input_col,
        "requested_max_length": int(max_length),
        "effective_max_length": int(effective_max_length),
        "tokenizer_max_length": None if tokenizer_max_length is None else int(tokenizer_max_length),
        "embedding_model_path": str(embedding_model_path),
        "embedding_metrics": metrics,
        "embedding_business_metrics": test_business,
        "embedding_fixed_metrics_at_target_reject_rate": test_fixed,
        "predictions_csv": str(pred_csv),
        "gains_lift_curve_csv": str(gains_lift_curve_csv),
        "note": "rerank is intentionally disabled per project policy.",
    }
    save_json(output_dir / "bert_embedding_eval_report.json", report)
    return report


def train_and_evaluate_bert_finetune(
    input_csv: Path,
    output_dir: Path,
    *,
    model_path: Path,
    sample_size: int,
    test_size: float,
    precision_floor: float,
    balance_test: bool,
    test_pos_cap: int | None,
    random_state: int,
    max_length: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    num_train_epochs: float,
    weight_decay: float,
    logging_steps: int,
    save_total_limit: int,
    fp16: bool,
    train_year_start: int = 2007,
    train_year_end: int = 2013,
    test_year_start: int = 2014,
    test_year_end: int = 2016,
    split_mode: str = "stratified",
    threshold_policy: str = "fixed_reject_rate",
    threshold_objective: str = "precision_at_reject_rate_bounds",
    recall_floor: float = 0.03,
    min_reject_rate: float = 0.03,
    max_reject_rate: float = 0.20,
    target_reject_rate: float = 0.35,
    rebuild_bert_input: bool = True,
    downsample_train_neg: bool = True,
    shared_split_dir: Path | None = None,
    force_rebuild_shared_split: bool = False,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

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
    if rebuild_bert_input or BERT_INPUT_COL not in df.columns:
        # Rebuild every run by default; standalone workflows can opt out and provide their own BERT input.
        df[BERT_INPUT_COL] = _build_bert_input_column(df)
    input_col = BERT_INPUT_COL if BERT_INPUT_COL in df.columns else TEXT_COL
    val_size = float(test_size) if 0.0 < float(test_size) < 0.5 else 0.2
    if split_mode == "stratified":
        if shared_split_dir is not None and str(shared_split_dir).strip():
            train_fit_df, val_df, test_df, split_summary = materialize_or_load_stratified_split(
                df,
                split_dir=Path(shared_split_dir),
                target_col=TARGET_COL,
                test_size=val_size,
                val_size=val_size,
                random_state=random_state,
                force_rebuild=bool(force_rebuild_shared_split),
            )
            train_fit_df = _realign_split_from_base(df, train_fit_df)
            val_df = _realign_split_from_base(df, val_df)
            test_df = _realign_split_from_base(df, test_df)
        else:
            train_fit_df, val_df, test_df, split_summary = stratified_train_val_test_split(
                df,
                target_col=TARGET_COL,
                test_size=val_size,
                val_size=val_size,
                random_state=random_state,
            )
        train_df = pd.concat([train_fit_df, val_df], axis=0).reset_index(drop=True)
        sampling_strategy = "stratified_random_split"
    elif downsample_train_neg:
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
        sampling_strategy = "time_split_train_neg_downsample"
    else:
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
        split_summary = {
            "train_total_before": int(len(train_df)),
            "train_pos_before": int((train_df[TARGET_COL].astype(int) == 1).sum()),
            "train_neg_before": int((train_df[TARGET_COL].astype(int) == 0).sum()),
            "train_pos_after": int((train_df[TARGET_COL].astype(int) == 1).sum()),
            "train_neg_after": int((train_df[TARGET_COL].astype(int) == 0).sum()),
            "test_total": int(len(test_df)),
            "test_pos": int((test_df[TARGET_COL].astype(int) == 1).sum()),
            "test_neg": int((test_df[TARGET_COL].astype(int) == 0).sum()),
        }
        sampling_strategy = "time_split_no_train_downsample"
    test_balance_summary: dict[str, int] | None = None
    if balance_test:
        test_df, test_balance_summary = downsample_binary_equal(
            test_df,
            target_col=TARGET_COL,
            random_state=random_state,
            pos_cap=test_pos_cap,
        )
    split_mode_name = (
        "stratified_random_split"
        if split_mode == "stratified"
        else (
            f"time_split_train_{train_year_start}_{train_year_end}"
            f"_test_{test_year_start}_{test_year_end}"
        )
    )
    if split_mode != "stratified":
        train_fit_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            stratify=train_df[TARGET_COL].astype(int),
            random_state=random_state,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    effective_max_length = _resolve_effective_max_length(max_length, tokenizer_max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=2,
        trust_remote_code=True,
    )

    def _to_dataset(frame: pd.DataFrame) -> Dataset:
        data = frame[[input_col, TARGET_COL]].copy()
        data = data.rename(columns={input_col: "text", TARGET_COL: "labels"})
        data["labels"] = data["labels"].astype(int)
        ds = Dataset.from_pandas(data, preserve_index=False)
        return ds

    train_ds = _to_dataset(train_fit_df)
    val_ds = _to_dataset(val_df)
    test_ds = _to_dataset(test_df)

    def tokenize_fn(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=effective_max_length,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    def _metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        prob = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].cpu().numpy()
        pred = np.argmax(logits, axis=-1)
        return compute_metrics(pd.Series(labels), pred, prob)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "hf_ckpt"),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        report_to=[],
        seed=random_state,
        fp16=fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=_metrics,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate(eval_dataset=val_ds)

    val_pred_result = trainer.predict(val_ds)
    val_logits = val_pred_result.predictions
    val_labels = val_pred_result.label_ids
    val_probs = torch.softmax(torch.tensor(val_logits), dim=-1)[:, 1].cpu().numpy()
    if threshold_policy == "fixed_reject_rate":
        fixed_threshold = threshold_by_target_reject_rate(
            val_probs,
            target_reject_rate=target_reject_rate,
        )
        threshold_info = {
            "metric": "fixed_reject_rate_quantile",
            "best_threshold": float(fixed_threshold["threshold"]),
            "best_score": float("nan"),
            "precision_floor": None,
            "recall_floor": None,
            "min_reject_rate": None,
            "max_reject_rate": None,
            "target_reject_rate": float(target_reject_rate),
            "constraint_satisfied": True,
            "fallback_metric": None,
            "validation_reject_rate": float(fixed_threshold["actual_reject_rate"]),
        }
    else:
        threshold_info = optimize_binary_threshold_business(
            pd.Series(val_labels),
            val_probs,
            threshold_objective=threshold_objective,
            precision_floor=precision_floor,
            recall_floor=recall_floor,
            min_reject_rate=min_reject_rate,
            max_reject_rate=max_reject_rate,
            target_reject_rate=target_reject_rate,
        )
    best_threshold = float(threshold_info["best_threshold"])
    val_preds_tuned = (val_probs >= best_threshold).astype(int)
    val_metrics = compute_metrics(pd.Series(val_labels), val_preds_tuned, val_probs)
    val_business = compute_business_metrics(pd.Series(val_labels), val_preds_tuned)
    val_fixed = evaluate_fixed_reject_rate(
        pd.Series(val_labels),
        val_probs,
        target_reject_rate=target_reject_rate,
        threshold=best_threshold,
    )

    pred_result = trainer.predict(test_ds)
    logits = pred_result.predictions
    labels = pred_result.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].cpu().numpy()
    preds = (probs >= best_threshold).astype(int)
    test_metrics = compute_metrics(pd.Series(labels), preds, probs)
    test_business = compute_business_metrics(pd.Series(labels), preds)
    test_fixed = evaluate_fixed_reject_rate(
        pd.Series(labels),
        probs,
        target_reject_rate=target_reject_rate,
        threshold=best_threshold,
    )
    gains_lift_curve = build_validation_to_test_reject_curve(
        y_val=pd.Series(val_labels),
        p_val=val_probs,
        y_test=pd.Series(labels),
        p_test=probs,
        reject_rates=[i / 100.0 for i in range(5, 51, 5)],
    )
    gains_lift_curve_csv = output_dir / "bert_finetune_gains_lift_curve.csv"
    gains_lift_curve.to_csv(gains_lift_curve_csv, index=False)

    pred_df = test_df[[input_col, TARGET_COL]].copy().reset_index(drop=True)
    pred_df["ft_prob"] = probs
    pred_df["ft_pred"] = preds
    pred_df["threshold_used"] = best_threshold
    pred_csv = output_dir / "bert_finetune_eval_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    trainer.save_model(str(output_dir / "finetuned_model"))
    tokenizer.save_pretrained(str(output_dir / "finetuned_model"))

    report = {
        "input_csv": str(input_csv),
        "source_rows": int(len(df)),
        "train_pos_cap": None if train_pos_cap is None else int(train_pos_cap),
        "sampling_strategy": sampling_strategy,
        "sampled_class_counts": {str(k): int(v) for k, v in sampled_class_counts.items()},
        "train_rows": int(len(train_df)),
        "train_fit_rows": int(len(train_fit_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "split_mode": split_mode_name,
        "train_sampling_summary": split_summary,
        "threshold_tuning": {
            "method": "validation_search",
            "threshold_policy": threshold_policy,
            "metric": threshold_info["metric"],
            "precision_floor": threshold_info.get("precision_floor"),
            "recall_floor": threshold_info.get("recall_floor"),
            "min_reject_rate": threshold_info.get("min_reject_rate"),
            "max_reject_rate": threshold_info.get("max_reject_rate"),
            "target_reject_rate": threshold_info.get("target_reject_rate"),
            "constraint_satisfied": bool(threshold_info.get("constraint_satisfied", False)),
            "fallback_metric": threshold_info.get("fallback_metric"),
            "best_threshold": best_threshold,
            "best_score": float(threshold_info["best_score"]),
            "validation_precision": float(threshold_info.get("validation_precision", val_business["bad_rate_in_reject_precision"])),
            "validation_recall": float(threshold_info.get("validation_recall", val_business["default_capture_recall"])),
            "validation_reject_rate": float(threshold_info.get("validation_reject_rate", val_business["reject_rate"])),
            "validation_metrics_at_best_threshold": val_metrics,
            "validation_business_metrics_at_best_threshold": val_business,
            "validation_fixed_metrics_at_target_reject_rate": val_fixed,
        },
        "test_balance_enabled": bool(balance_test),
        "test_balance_summary": test_balance_summary,
        "bert_input_column": input_col,
        "requested_max_length": int(max_length),
        "effective_max_length": int(effective_max_length),
        "tokenizer_max_length": None if tokenizer_max_length is None else int(tokenizer_max_length),
        "base_model_path": str(model_path),
        "finetuned_model_dir": str(output_dir / "finetuned_model"),
        "train_runtime_sec": float(train_result.metrics.get("train_runtime", 0.0)),
        "eval_metrics_trainer": {k: float(v) for k, v in eval_result.items() if isinstance(v, (int, float))},
        "test_metrics": test_metrics,
        "test_business_metrics": test_business,
        "test_fixed_metrics_at_target_reject_rate": test_fixed,
        "predictions_csv": str(pred_csv),
        "gains_lift_curve_csv": str(gains_lift_curve_csv),
    }
    save_json(output_dir / "bert_finetune_eval_report.json", report)
    return report


