from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml_standalone.ml_workflow import prepare_ml_dataset
from workflow_common import ISSUE_YEAR_COL, TARGET_COL, save_json


def _series_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _remove_file_if_exists(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        # best-effort cleanup
        pass


def build_shared_subset_from_raw(
    *,
    input_csv: Path,
    raw_output_csv: Path,
    processed_output_csv: Path,
    shared_output_csv: Path,
    ml_subset_output_csv: Path,
    chunksize: int,
    max_rows: int | None,
    max_chunks: int | None,
    allowed_grades: list[str],
    min_int_rate: float,
    max_annual_inc: float,
    keep_ml_intermediate: bool = True,
    write_ml_subset_alias: bool = False,
) -> dict[str, Any]:
    """
    Rebuild a unified shared subset from raw accepted data.

    Pipeline:
    1) Re-run ML preprocessing on the raw accepted CSV.
    2) Apply deterministic subset rules on ml_processed.csv.
    3) Persist shared subset output.
    4) Optionally keep ML intermediate files and/or write compatibility alias.
    """
    prep_summary = prepare_ml_dataset(
        input_csv=input_csv,
        raw_output_csv=raw_output_csv,
        processed_output_csv=processed_output_csv,
        chunksize=chunksize,
        max_rows=max_rows,
        max_chunks=max_chunks,
    )

    if not processed_output_csv.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_output_csv}")

    df = pd.read_csv(processed_output_csv, low_memory=False)
    required = {TARGET_COL, ISSUE_YEAR_COL, "grade", "int_rate", "annual_inc"}
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns in processed data: {missing}")

    df = df.copy()
    df["grade"] = df["grade"].astype(str).str.strip().str.upper()
    df["int_rate"] = _series_to_float(df["int_rate"])
    df["annual_inc"] = _series_to_float(df["annual_inc"])
    df[ISSUE_YEAR_COL] = _series_to_float(df[ISSUE_YEAR_COL])
    df[TARGET_COL] = _series_to_float(df[TARGET_COL]).astype("Int64")

    mask = (
        df["grade"].isin([g.strip().upper() for g in allowed_grades])
        & (df["int_rate"] >= float(min_int_rate))
        & (df["annual_inc"] <= float(max_annual_inc))
        & df[TARGET_COL].notna()
        & df[ISSUE_YEAR_COL].notna()
    )
    subset = df[mask].copy()
    if subset.empty:
        raise ValueError("Shared subset is empty. Please relax subset rules.")

    # Stable row order for reproducibility.
    order_cols: list[str] = [ISSUE_YEAR_COL, TARGET_COL]
    for c in ["grade", "int_rate", "annual_inc", "loan_amnt"]:
        if c in subset.columns:
            order_cols.append(c)
    subset = subset.sort_values(order_cols, kind="mergesort").reset_index(drop=True)

    shared_output_csv.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(shared_output_csv, index=False)

    if write_ml_subset_alias:
        ml_subset_output_csv.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(ml_subset_output_csv, index=False)
    else:
        _remove_file_if_exists(ml_subset_output_csv)

    by_year = (
        subset.groupby(ISSUE_YEAR_COL)[TARGET_COL]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "default_rate"})
    )
    by_grade = (
        subset.groupby("grade")[TARGET_COL]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "default_rate"})
        .sort_values("grade")
    )

    payload = {
        "input_csv": str(input_csv),
        "prepare_ml_summary": prep_summary,
        "subset_rules": {
            "allowed_grades": [g.strip().upper() for g in allowed_grades],
            "min_int_rate": float(min_int_rate),
            "max_annual_inc": float(max_annual_inc),
        },
        "build_options": {
            "keep_ml_intermediate": bool(keep_ml_intermediate),
            "write_ml_subset_alias": bool(write_ml_subset_alias),
        },
        "artifacts": {
            "ml_raw_csv": str(raw_output_csv) if keep_ml_intermediate else None,
            "ml_processed_csv": str(processed_output_csv) if keep_ml_intermediate else None,
            "shared_subset_csv": str(shared_output_csv),
            "ml_subset_compat_csv": str(ml_subset_output_csv) if write_ml_subset_alias else None,
        },
        "subset_summary": {
            "rows": int(len(subset)),
            "target_default_rate": float(subset[TARGET_COL].astype(int).mean()),
            "int_rate_min": float(subset["int_rate"].min()),
            "int_rate_max": float(subset["int_rate"].max()),
            "annual_inc_median": float(subset["annual_inc"].median()),
            "year_min": int(subset[ISSUE_YEAR_COL].min()),
            "year_max": int(subset[ISSUE_YEAR_COL].max()),
            "grade_counts": {
                str(k): int(v)
                for k, v in subset["grade"].value_counts(dropna=False).sort_index().to_dict().items()
            },
            "target_counts": {
                str(k): int(v)
                for k, v in subset[TARGET_COL].astype(int).value_counts(dropna=False).sort_index().to_dict().items()
            },
            "target_by_year": by_year.to_dict(orient="records"),
            "target_by_grade": by_grade.to_dict(orient="records"),
        },
    }
    save_json(shared_output_csv.with_suffix(".summary.json"), payload)

    if not keep_ml_intermediate:
        _remove_file_if_exists(raw_output_csv)
        _remove_file_if_exists(processed_output_csv)
        _remove_file_if_exists(processed_output_csv.with_suffix(".summary.json"))

    return payload
