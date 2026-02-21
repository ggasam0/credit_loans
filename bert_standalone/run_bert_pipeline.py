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

from bert_standalone.bert_workflow import (
    BERT_INPUT_COL,
    DEFAULT_BCE_EMBED_PATH,
    evaluate_bert_embedding_local,
    train_and_evaluate_bert_finetune,
)
from workflow_common import ISSUE_YEAR_COL, SHARED_ROW_ID_COL, TARGET_COL, TEXT_COL, save_json
from workflow_common import materialize_or_load_stratified_split, stratified_train_val_test_split


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value).strip()
    return text if text else "missing"


def _resolve_desc_column(df: pd.DataFrame, desc_candidates: list[str]) -> str | None:
    for col in desc_candidates:
        if col in df.columns:
            return col
    return None


def _load_feature_columns(
    df: pd.DataFrame,
    feature_selection_json: Path | None,
) -> list[str]:
    excluded = {TARGET_COL, ISSUE_YEAR_COL, SHARED_ROW_ID_COL, TEXT_COL, BERT_INPUT_COL, "desc", "desc_clean"}

    if feature_selection_json is not None and feature_selection_json.exists():
        payload = json.loads(feature_selection_json.read_text(encoding="utf-8"))
        numeric_cols = payload.get("selected_numeric_columns", []) or []
        categorical_cols = payload.get("selected_categorical_columns", []) or []
        cols = [c for c in [*numeric_cols, *categorical_cols] if c in df.columns and c not in excluded]
        if cols:
            return cols

    inferred: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            inferred.append(col)
    return inferred


def _build_subset_bert_input(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    desc_col: str | None,
    include_desc: bool,
) -> pd.Series:
    def _row_to_text(row: pd.Series) -> str:
        lines: list[str] = ["[Structured Risk Features]"]
        for col in feature_cols:
            lines.append(f"{col}: {_format_value(row.get(col))}")
        if include_desc:
            lines.append("")
            lines.append("[Borrower Description]")
            if desc_col is None:
                lines.append("missing")
            else:
                desc_raw = row.get(desc_col)
                desc = "" if pd.isna(desc_raw) else str(desc_raw).strip()
                lines.append(desc if desc else "missing")
        return "\n".join(lines)

    return frame.apply(_row_to_text, axis=1)


def _realign_split_from_prepared(prepared_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    if SHARED_ROW_ID_COL not in split_df.columns:
        return split_df.copy()
    if SHARED_ROW_ID_COL not in prepared_df.columns:
        raise ValueError(
            f"Shared split contains {SHARED_ROW_ID_COL} but prepared dataframe does not; cannot realign columns."
        )
    base = prepared_df.drop_duplicates(subset=[SHARED_ROW_ID_COL], keep="first").set_index(
        SHARED_ROW_ID_COL,
        drop=False,
    )
    ids = split_df[SHARED_ROW_ID_COL].tolist()
    missing_ids = [rid for rid in ids if rid not in base.index]
    if missing_ids:
        raise ValueError(f"Shared split row ids not found in prepared dataframe: missing={len(missing_ids)}.")
    return base.loc[ids].reset_index(drop=True).copy()


def _prepare_input(
    *,
    input_csv: Path,
    prepared_csv: Path,
    feature_selection_json: Path | None,
    desc_candidates: list[str],
    include_desc: bool,
) -> dict[str, Any]:
    df = pd.read_csv(input_csv, low_memory=False)
    missing_required = [c for c in [TARGET_COL, ISSUE_YEAR_COL] if c not in df.columns]
    if missing_required:
        raise ValueError(f"{input_csv} missing required columns: {missing_required}")

    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[ISSUE_YEAR_COL] = pd.to_numeric(df[ISSUE_YEAR_COL], errors="coerce")
    df = df[df[TARGET_COL].notna() & df[ISSUE_YEAR_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df[ISSUE_YEAR_COL] = df[ISSUE_YEAR_COL].astype(int)
    if SHARED_ROW_ID_COL not in df.columns:
        df[SHARED_ROW_ID_COL] = np.arange(len(df), dtype=int)

    desc_col = _resolve_desc_column(df, desc_candidates)
    if TEXT_COL not in df.columns:
        if desc_col is not None:
            df[TEXT_COL] = df[desc_col].fillna("").astype(str)
        else:
            df[TEXT_COL] = ""
    else:
        df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    feature_cols = _load_feature_columns(df, feature_selection_json)
    if not feature_cols:
        raise ValueError("No usable feature columns to build BERT input.")

    df[BERT_INPUT_COL] = _build_subset_bert_input(
        df,
        feature_cols=feature_cols,
        desc_col=desc_col,
        include_desc=include_desc,
    )

    prepared_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prepared_csv, index=False)

    return {
        "input_csv": str(input_csv),
        "prepared_csv": str(prepared_csv),
        "rows_after_basic_cleaning": int(len(df)),
        "feature_dimension_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "desc_column_used": desc_col,
        "include_desc": bool(include_desc),
        "feature_selection_json": None if feature_selection_json is None else str(feature_selection_json),
    }


def _save_pretraining_splits(
    *,
    prepared_csv: Path,
    data_dir: Path,
    test_size: float,
    val_size: float,
    random_state: int,
    shared_split_dir: Path | None,
    force_rebuild_shared_split: bool,
) -> dict[str, Any]:
    df = pd.read_csv(prepared_csv, low_memory=False)
    if shared_split_dir is not None and str(shared_split_dir).strip():
        train_fit_df, val_df, test_df, split_summary = materialize_or_load_stratified_split(
            df,
            split_dir=Path(shared_split_dir),
            target_col=TARGET_COL,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            force_rebuild=bool(force_rebuild_shared_split),
        )
        train_fit_df = _realign_split_from_prepared(df, train_fit_df)
        val_df = _realign_split_from_prepared(df, val_df)
        test_df = _realign_split_from_prepared(df, test_df)
    else:
        train_fit_df, val_df, test_df, split_summary = stratified_train_val_test_split(
            df,
            target_col=TARGET_COL,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "bert_processed.csv", index=False)
    train_fit_df.to_csv(data_dir / "train_fit.csv", index=False)
    val_df.to_csv(data_dir / "validation.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    save_json(
        data_dir / "split_summary.json",
        {
            "prepared_csv": str(prepared_csv),
            "split_mode": "stratified",
            "split_summary": split_summary,
        },
    )
    return split_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone BERT workflow on a fixed ML subset: prepare bert_input + embedding eval + finetune eval."
    )
    parser.add_argument("--input", default="data/shared/shared_subset.csv")
    parser.add_argument("--output-dir", default="output/bert_standalone")
    parser.add_argument("--prepared-csv", default="")
    parser.add_argument(
        "--feature-selection-json",
        default="output/ml_standalone/analysis/feature_selection.json",
        help="Use ML-selected feature list to keep the same feature dimension.",
    )
    parser.add_argument("--desc-columns", nargs="+", default=["desc_clean", "desc"])
    parser.add_argument("--include-desc", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--run-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-finetune", action=argparse.BooleanOptionalAction, default=True)

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

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional cap for positive-class train rows.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--precision-floor", type=float, default=0.4)
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
    parser.add_argument("--recall-floor", type=float, default=0.03)
    parser.add_argument("--min-reject-rate", type=float, default=0.03)
    parser.add_argument("--max-reject-rate", type=float, default=0.20)
    parser.add_argument("--target-reject-rate", type=float, default=0.35)
    parser.add_argument(
        "--downsample-train-neg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to downsample train negatives to match positives before BERT training/eval.",
    )
    parser.add_argument("--balance-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test-pos-cap", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=514)
    parser.add_argument("--embedding-model-path", default=DEFAULT_BCE_EMBED_PATH)

    parser.add_argument("--model-path", default=DEFAULT_BCE_EMBED_PATH)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)

    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_csv = (
        Path(args.prepared_csv)
        if str(args.prepared_csv).strip()
        else output_dir / "data" / "bert_subset_processed.csv"
    )
    feature_selection_json = Path(args.feature_selection_json) if str(args.feature_selection_json).strip() else None

    prep_report = _prepare_input(
        input_csv=Path(args.input),
        prepared_csv=prepared_csv,
        feature_selection_json=feature_selection_json,
        desc_candidates=args.desc_columns,
        include_desc=bool(args.include_desc),
    )
    split_summary = _save_pretraining_splits(
        prepared_csv=prepared_csv,
        data_dir=output_dir / "data",
        test_size=float(args.test_size),
        val_size=float(args.val_size),
        random_state=int(args.random_state),
        shared_split_dir=(Path(args.shared_split_dir) if str(args.shared_split_dir).strip() else None),
        force_rebuild_shared_split=bool(args.force_rebuild_shared_split),
    )

    embedding_report: dict[str, Any] | None = None
    if args.run_embedding:
        embedding_report = evaluate_bert_embedding_local(
            input_csv=prepared_csv,
            output_dir=output_dir / "embedding_eval",
            embedding_model_path=Path(args.embedding_model_path),
            sample_size=args.sample_size,
            test_size=args.test_size,
            precision_floor=args.precision_floor,
            split_mode="stratified",
            threshold_policy=args.threshold_policy,
            threshold_objective=args.threshold_objective,
            recall_floor=args.recall_floor,
            min_reject_rate=args.min_reject_rate,
            max_reject_rate=args.max_reject_rate,
            target_reject_rate=args.target_reject_rate,
            balance_test=args.balance_test,
            test_pos_cap=args.test_pos_cap,
            random_state=args.random_state,
            device=args.device,
            max_length=args.max_length,
            rebuild_bert_input=False,
            downsample_train_neg=bool(args.downsample_train_neg),
            shared_split_dir=(Path(args.shared_split_dir) if str(args.shared_split_dir).strip() else None),
        )

    finetune_report: dict[str, Any] | None = None
    if args.run_finetune:
        finetune_report = train_and_evaluate_bert_finetune(
            input_csv=prepared_csv,
            output_dir=output_dir / "finetune_eval",
            model_path=Path(args.model_path),
            sample_size=args.sample_size,
            test_size=args.test_size,
            precision_floor=args.precision_floor,
            split_mode="stratified",
            threshold_policy=args.threshold_policy,
            threshold_objective=args.threshold_objective,
            recall_floor=args.recall_floor,
            min_reject_rate=args.min_reject_rate,
            max_reject_rate=args.max_reject_rate,
            target_reject_rate=args.target_reject_rate,
            balance_test=args.balance_test,
            test_pos_cap=args.test_pos_cap,
            random_state=args.random_state,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            fp16=args.fp16,
            rebuild_bert_input=False,
            downsample_train_neg=bool(args.downsample_train_neg),
            shared_split_dir=(Path(args.shared_split_dir) if str(args.shared_split_dir).strip() else None),
        )

    report = {
        "input_csv": str(args.input),
        "output_dir": str(output_dir),
        "preparation": prep_report,
        "split_summary": split_summary,
        "embedding_eval": embedding_report,
        "finetune_eval": finetune_report,
    }
    save_json(output_dir / "run_report.json", report)
    return report


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


