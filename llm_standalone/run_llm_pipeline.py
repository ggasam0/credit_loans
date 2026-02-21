from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm_standalone.llm_workflow import (  # noqa: E402
    DEFAULT_QWEN_06B_PATH,
    _generate_with_local_model_batch,
    write_llamafactory_qlora_assets,
)
from workflow_common import (  # noqa: E402
    SHARED_ROW_ID_COL,
    TARGET_COL,
    compute_business_metrics,
    compute_metrics,
    materialize_or_load_stratified_split,
    save_json,
    stratified_train_val_test_split,
)


STUDENT_SYSTEM_PROMPT = (
    "You are a bank credit risk reviewer. "
    "Read the input and return strict JSON only with keys: "
    '{"reason":"...","action":"reject|approve"}. '
    "Do not output markdown, do not output extra keys."
)


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    normalized = text.strip()
    normalized = re.sub(r"```json\\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("```", "")
    end = normalized.rfind("}")
    if end < 0:
        return None
    start = normalized.rfind("{", 0, end + 1)
    while start >= 0:
        candidate = normalized[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
            return None
        except Exception:
            start = normalized.rfind("{", 0, start)
    return None


def _label_to_action(label: int) -> str:
    return "reject" if int(label) == 1 else "approve"


def _action_to_label(action: str) -> int | None:
    text = str(action).strip().lower()
    if text == "reject":
        return 1
    if text == "approve":
        return 0
    return None


def _extract_action_fallback(raw_text: str) -> str | None:
    text = str(raw_text).lower()
    if "reject" in text:
        return "reject"
    if "approve" in text:
        return "approve"
    return None


def _teacher_system_prompt() -> str:
    return (
        "You are a senior bank credit risk reviewer. "
        "You already know the true action label. "
        "Use only the provided input text as evidence. "
        "Return strict JSON only with keys: reason, action. "
        "action must be exactly reject or approve."
    )


def _teacher_user_prompt(bert_input: str, action: str) -> str:
    return (
        "[Known Label]\n"
        f"action={action}\n\n"
        "[Input]\n"
        f"{bert_input}\n\n"
        'Return strict JSON: {"reason":"...","action":"reject|approve"}'
    )


def _sample_ratio_by_target(
    df: pd.DataFrame,
    *,
    ratio: float,
    target_col: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    ratio = float(max(0.0, min(1.0, ratio)))
    if ratio <= 0.0:
        raise ValueError("sample ratio must be > 0.")

    parts: list[pd.DataFrame] = []
    class_before: dict[str, int] = {}
    class_after: dict[str, int] = {}

    for label in [0, 1]:
        group = df[df[target_col].astype(int) == label]
        class_before[str(label)] = int(len(group))
        if group.empty:
            class_after[str(label)] = 0
            continue
        n = int(round(len(group) * ratio))
        n = max(1, min(n, len(group)))
        sampled = group.sample(n=n, random_state=random_state, replace=False)
        class_after[str(label)] = int(len(sampled))
        parts.append(sampled)

    if not parts:
        raise ValueError("No rows available after class grouping.")

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    summary = {
        "ratio": float(ratio),
        "rows_before": int(len(df)),
        "rows_after": int(len(out)),
        "class_before": class_before,
        "class_after": class_after,
    }
    return out, summary


def _ratio_file_tag(ratio: float) -> str:
    ratio = float(max(0.0, min(1.0, ratio)))
    if abs(ratio - 1.0) < 1e-9:
        return "full"
    pct = ratio * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}pct"
    pct_text = f"{pct:.1f}".rstrip("0").rstrip(".").replace(".", "_")
    return f"{pct_text}pct"


def _realign_split_from_prepared(prepared_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    if SHARED_ROW_ID_COL not in split_df.columns:
        return split_df.copy()
    if SHARED_ROW_ID_COL not in prepared_df.columns:
        raise ValueError(
            f"Shared split contains {SHARED_ROW_ID_COL} but input dataframe does not; cannot realign columns."
        )
    base = prepared_df.drop_duplicates(subset=[SHARED_ROW_ID_COL], keep="first").set_index(
        SHARED_ROW_ID_COL,
        drop=False,
    )
    ids = split_df[SHARED_ROW_ID_COL].tolist()
    missing_ids = [rid for rid in ids if rid not in base.index]
    if missing_ids:
        raise ValueError(f"Shared split row ids not found in input dataframe: missing={len(missing_ids)}.")
    return base.loc[ids].reset_index(drop=True).copy()


def prepare_subset_data(
    *,
    input_csv: Path,
    output_dir: Path,
    ratio: float,
    random_state: int,
    shared_split_dir: Path | None,
    force_rebuild_shared_split: bool,
    test_size: float,
    val_size: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    required_cols = [TARGET_COL, "bert_input"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} missing required columns: {missing}")

    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    if SHARED_ROW_ID_COL not in df.columns:
        df[SHARED_ROW_ID_COL] = range(len(df))
    df["bert_input"] = df["bert_input"].fillna("").astype(str).str.strip()
    empty_bert_input_count = int((df["bert_input"].str.len() == 0).sum())
    if empty_bert_input_count > 0:
        df.loc[df["bert_input"].str.len() == 0, "bert_input"] = "missing"

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
    train_df = pd.concat([train_fit_df, val_df], axis=0).reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Train split is empty.")
    if test_df.empty:
        raise ValueError("Test split is empty.")

    train_sub, train_summary = _sample_ratio_by_target(
        train_df,
        ratio=ratio,
        target_col=TARGET_COL,
        random_state=random_state,
    )
    test_sub, test_summary = _sample_ratio_by_target(
        test_df,
        ratio=ratio,
        target_col=TARGET_COL,
        random_state=random_state,
    )

    file_tag = _ratio_file_tag(ratio)
    train_csv = data_dir / f"train_{file_tag}.csv"
    test_csv = data_dir / f"test_{file_tag}.csv"
    all_csv = data_dir / f"all_{file_tag}.csv"
    train_sub.to_csv(train_csv, index=False)
    test_sub.to_csv(test_csv, index=False)
    pd.concat([train_sub, test_sub], axis=0).reset_index(drop=True).to_csv(all_csv, index=False)

    legacy_removed: list[str] = []
    legacy_remove_skipped: list[str] = []
    legacy_names = ["train_5pct.csv", "test_5pct.csv", "all_5pct.csv"]
    for legacy_name in legacy_names:
        legacy_path = data_dir / legacy_name
        if legacy_path in {train_csv, test_csv, all_csv}:
            continue
        if legacy_path.exists():
            try:
                legacy_path.unlink()
                legacy_removed.append(str(legacy_path))
            except PermissionError:
                legacy_remove_skipped.append(f"{legacy_path} (locked)")

    payload = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "file_tag": file_tag,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "all_csv": str(all_csv),
        "split_mode": "stratified",
        "ratio": float(ratio),
        "empty_bert_input_filled": int(empty_bert_input_count),
        "shared_split_dir": None if shared_split_dir is None else str(shared_split_dir),
        "force_rebuild_shared_split": bool(force_rebuild_shared_split),
        "split_summary": split_summary,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "legacy_removed": legacy_removed,
        "legacy_remove_skipped": legacy_remove_skipped,
    }
    save_json(output_dir / "subset_report.json", payload)
    return payload


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null", "nan"}:
        return None
    text = text.replace("%", "").replace(",", "")
    try:
        return float(text)
    except Exception:
        return None


def _parse_bert_input(bert_input: str) -> tuple[dict[str, str], str]:
    features: dict[str, str] = {}
    desc_lines: list[str] = []
    in_structured = False
    in_desc = False
    for raw_line in str(bert_input).splitlines():
        line = raw_line.strip()
        if line == "[Structured Risk Features]":
            in_structured = True
            in_desc = False
            continue
        if line == "[Borrower Description]":
            in_desc = True
            in_structured = False
            continue
        if in_structured and ":" in line:
            key, value = line.split(":", 1)
            features[key.strip()] = value.strip()
            continue
        if in_desc:
            desc_lines.append(line)
    desc = " ".join([x for x in desc_lines if x and x.lower() != "null"]).strip()
    return features, desc


def _format_pct(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"{x:.1f}%"


def _format_num(x: float | None) -> str:
    if x is None:
        return "N/A"
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.2f}"


def _build_teacher_reason(bert_input: str, target: int) -> tuple[str, dict[str, Any]]:
    features, desc = _parse_bert_input(bert_input)
    grade = str(features.get("grade", "")).strip().upper()
    purpose = str(features.get("purpose", "")).strip().lower()
    home_ownership = str(features.get("home_ownership", "")).strip().upper()
    verification_status = str(features.get("verification_status", "")).strip().lower()

    int_rate = _safe_float(features.get("int_rate"))
    dti = _safe_float(features.get("dti"))
    revol_util = _safe_float(features.get("revol_util"))
    fico = _safe_float(features.get("fico_mean"))
    annual_inc = _safe_float(features.get("annual_inc"))
    term_months = _safe_float(features.get("term_months"))
    emp_length = _safe_float(features.get("emp_length_years"))

    risk_signals: list[tuple[int, str]] = []
    strength_signals: list[tuple[int, str]] = []

    if int_rate is not None:
        if int_rate >= 24:
            risk_signals.append((4, f"a very high borrowing cost ({_format_pct(int_rate)})"))
        elif int_rate >= 18:
            risk_signals.append((3, f"an elevated interest rate ({_format_pct(int_rate)})"))
        elif int_rate <= 12:
            strength_signals.append((2, f"a relatively low pricing level ({_format_pct(int_rate)})"))

    if dti is not None:
        if dti >= 30:
            risk_signals.append((4, f"heavy leverage pressure (DTI {_format_num(dti)})"))
        elif dti >= 22:
            risk_signals.append((2, f"above-comfort debt burden (DTI {_format_num(dti)})"))
        elif dti <= 12:
            strength_signals.append((3, f"conservative leverage (DTI {_format_num(dti)})"))
        elif dti <= 18:
            strength_signals.append((2, f"manageable leverage (DTI {_format_num(dti)})"))

    if revol_util is not None:
        if revol_util >= 95:
            risk_signals.append((4, f"near-saturation revolving utilization ({_format_pct(revol_util)})"))
        elif revol_util >= 80:
            risk_signals.append((3, f"high revolving utilization ({_format_pct(revol_util)})"))
        elif revol_util <= 55:
            strength_signals.append((2, f"controlled revolving utilization ({_format_pct(revol_util)})"))

    if fico is not None:
        if fico < 660:
            risk_signals.append((4, f"a weak credit score level (FICO {_format_num(fico)})"))
        elif fico < 690:
            risk_signals.append((2, f"a below-prime score band (FICO {_format_num(fico)})"))
        elif fico >= 720:
            strength_signals.append((4, f"a strong credit score (FICO {_format_num(fico)})"))
        elif fico >= 700:
            strength_signals.append((3, f"a solid score profile (FICO {_format_num(fico)})"))

    if grade in {"E", "F", "G"}:
        risk_signals.append((3, f"a high-risk grade bucket ({grade})"))
    elif grade in {"A", "B"}:
        strength_signals.append((3, f"a favorable grade bucket ({grade})"))
    elif grade == "C":
        strength_signals.append((1, "a mid-risk grade with acceptable baseline quality"))

    if annual_inc is not None:
        if annual_inc < 45000:
            risk_signals.append((3, f"limited income buffer (${_format_num(annual_inc)})"))
        elif annual_inc < 65000:
            risk_signals.append((1, f"moderate income capacity (${_format_num(annual_inc)})"))
        elif annual_inc >= 100000:
            strength_signals.append((3, f"a strong income base (${_format_num(annual_inc)})"))
        elif annual_inc >= 75000:
            strength_signals.append((2, f"a stable income base (${_format_num(annual_inc)})"))

    if term_months is not None:
        if term_months >= 60:
            risk_signals.append((1, f"a longer repayment horizon ({_format_num(term_months)} months)"))
        elif term_months <= 36:
            strength_signals.append((1, f"a shorter repayment horizon ({_format_num(term_months)} months)"))

    if emp_length is not None:
        if emp_length < 1:
            risk_signals.append((2, f"very short employment tenure ({_format_num(emp_length)} years)"))
        elif emp_length >= 7:
            strength_signals.append((1, f"longer employment tenure ({_format_num(emp_length)} years)"))

    if home_ownership == "RENT":
        risk_signals.append((1, "rent status with less balance-sheet cushion"))
    elif home_ownership in {"MORTGAGE", "OWN"}:
        strength_signals.append((1, f"{home_ownership.lower()} housing status"))

    if verification_status == "not verified":
        risk_signals.append((1, "unverified income information"))
    elif verification_status in {"verified", "source verified"}:
        strength_signals.append((1, "income verification completed"))

    if purpose in {"small_business", "medical", "moving", "vacation", "wedding", "renewable_energy"}:
        risk_signals.append((1, f"use-case uncertainty in purpose ({purpose})"))
    elif purpose in {"credit_card", "debt_consolidation"}:
        strength_signals.append((1, f"a common refinancing purpose ({purpose})"))

    if desc:
        if any(token in desc.lower() for token in ["stable", "steady", "income", "payoff", "consolidat"]):
            strength_signals.append((1, "borrower narrative indicates a repayment plan"))
        if any(token in desc.lower() for token in ["urgent", "behind", "late", "hardship", "medical"]):
            risk_signals.append((1, "borrower narrative reflects potential stress context"))

    risk_signals = sorted(risk_signals, key=lambda x: x[0], reverse=True)
    strength_signals = sorted(strength_signals, key=lambda x: x[0], reverse=True)

    if int(target) == 1:
        top_risks = [txt for _, txt in risk_signals[:3]]
        if not top_risks:
            top_risks = ["multiple risk factors cluster at less favorable levels"]
        top_strength = [txt for _, txt in strength_signals[:1]]
        reason = "Key concerns include " + "; ".join(top_risks) + "."
        if top_strength:
            reason += " Some supportive signals exist, but they are not strong enough to offset the downside concentration."
        reason += " Overall, the profile sits in a fragile risk zone."
    else:
        top_strength = [txt for _, txt in strength_signals[:3]]
        if not top_strength:
            top_strength = ["core affordability and credit quality indicators remain within an acceptable range"]
        top_risks = [txt for _, txt in risk_signals[:1]]
        reason = "Positive signals include " + "; ".join(top_strength) + "."
        if top_risks:
            reason += " There is still caution around " + top_risks[0] + ", but it appears manageable in context."
        reason += " Overall, repayment capacity looks comparatively stable."

    meta = {
        "risk_signal_count": int(len(risk_signals)),
        "strength_signal_count": int(len(strength_signals)),
        "grade": grade,
        "purpose": purpose,
    }
    return reason, meta


def generate_teacher_sft_data(
    *,
    train_csv: Path,
    output_jsonl: Path,
    audit_csv: Path,
    teacher_model_path: Path | None,
    device_map: str | None,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    random_state: int,
    sleep_ms: int,
) -> dict[str, Any]:
    df = pd.read_csv(train_csv, low_memory=False)
    required_cols = [TARGET_COL, "bert_input"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{train_csv} missing required columns: {missing}")

    df = df.copy().reset_index(drop=True)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df["bert_input"] = df["bert_input"].fillna("").astype(str)
    df = df[df["bert_input"].str.len() > 0].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows available for teacher SFT generation.")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        output_jsonl.unlink()
    if audit_csv.exists():
        audit_csv.unlink()

    kept = 0
    dropped = 0
    total = int(len(df))

    with output_jsonl.open("w", encoding="utf-8") as writer, audit_csv.open("w", encoding="utf-8", newline="") as audit_file:
        fieldnames = [
            "row_id",
            "target",
            "expected_action",
            "status",
            "reason",
            "reason_source",
            "risk_signal_count",
            "strength_signal_count",
            "normalized_json",
        ]
        audit_writer = csv.DictWriter(audit_file, fieldnames=fieldnames)
        audit_writer.writeheader()
        audit_file.flush()

        for idx, row in df.iterrows():
            row_id = int(idx)
            target = int(row[TARGET_COL])
            expected_action = _label_to_action(target)
            bert_input = str(row["bert_input"])

            reason, meta = _build_teacher_reason(bert_input=bert_input, target=target)
            normalized_json = json.dumps({"reason": reason, "action": expected_action}, ensure_ascii=False)
            sample = {
                "system": STUDENT_SYSTEM_PROMPT,
                "conversations": [
                    {"from": "human", "value": bert_input},
                    {"from": "gpt", "value": normalized_json},
                ],
            }
            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")
            writer.flush()
            kept += 1

            audit_writer.writerow(
                {
                    "row_id": row_id,
                    "target": target,
                    "expected_action": expected_action,
                    "status": "kept",
                    "reason": "ok",
                    "reason_source": "codex_teacher_rules",
                    "risk_signal_count": int(meta.get("risk_signal_count", 0)),
                    "strength_signal_count": int(meta.get("strength_signal_count", 0)),
                    "normalized_json": normalized_json,
                }
            )
            audit_file.flush()
            processed = row_id + 1
            print(f"[teacher-sft-json] processed {processed}/{total} (kept={kept}, dropped={dropped}, mode=codex_teacher)")
            if sleep_ms > 0:
                time.sleep(float(sleep_ms) / 1000.0)

    payload = {
        "train_csv": str(train_csv),
        "output_jsonl": str(output_jsonl),
        "audit_csv": str(audit_csv),
        "teacher_mode": "codex_teacher_rules",
        "teacher_model_path": str(teacher_model_path) if teacher_model_path else None,
        "device_map": device_map,
        "batch_size": int(max(1, int(batch_size))),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "random_state": int(random_state),
        "total_rows": total,
        "kept_rows": int(kept),
        "dropped_rows": int(dropped),
    }
    save_json(output_jsonl.with_suffix(".summary.json"), payload)
    return payload


def write_lf_train_assets(
    *,
    sft_jsonl: Path,
    output_dir: Path,
    student_model_path: Path,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    cutoff_len: int,
    max_samples: int | None,
) -> dict[str, Any]:
    data_dir = output_dir / "lf_data"
    dataset_name = "credit_teacher_sft_5pct_json"
    train_yaml = data_dir / "qwen3_06b_lora_sft_5pct.yaml"
    train_cmd = data_dir / "run_qwen3_06b_lora_sft_5pct.ps1"
    train_out_dir = output_dir / "qwen3_06b_lora_sft"

    payload = write_llamafactory_qlora_assets(
        sft_jsonl=sft_jsonl,
        dataset_dir=data_dir,
        dataset_name=dataset_name,
        student_model_path=student_model_path,
        output_yaml=train_yaml,
        output_cmd=train_cmd,
        output_dir=train_out_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        max_samples=max_samples,
    )
    return payload


def infer_with_finetuned_model(
    *,
    test_csv: Path,
    output_csv: Path,
    model_path: Path,
    adapter_path: Path,
    device_map: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    sleep_ms: int,
) -> dict[str, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pd.read_csv(test_csv, low_memory=False)
    required_cols = [TARGET_COL, "bert_input"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{test_csv} missing required columns: {missing}")

    df = df.copy().reset_index(drop=True)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df["bert_input"] = df["bert_input"].fillna("").astype(str)
    df = df[df["bert_input"].str.len() > 0].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows available for inference.")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    total = int(len(df))
    batch_n = max(1, int(batch_size))
    responses: list[str] = []
    parsed_actions: list[str] = []
    parsed_reasons: list[str] = []
    parse_status: list[str] = []
    pred_labels: list[int] = []

    for start in range(0, total, batch_n):
        end = min(start + batch_n, total)
        batch = df.iloc[start:end]
        batch_messages: list[list[dict[str, str]]] = []
        for _, row in batch.iterrows():
            batch_messages.append(
                [
                    {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                    {"role": "user", "content": str(row["bert_input"])},
                ]
            )

        raw_outputs = _generate_with_local_model_batch(
            batch_messages,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=False,
        )

        for raw in raw_outputs:
            obj = _extract_last_json_object(raw)
            action = None
            reason = ""
            status = "ok_json"
            if obj is not None:
                action = str(obj.get("action", "")).strip().lower()
                reason = str(obj.get("reason", "")).strip()
                if action not in {"reject", "approve"}:
                    action = None
                    status = "bad_action_in_json"
            else:
                status = "bad_json"

            if action is None:
                fallback = _extract_action_fallback(raw)
                if fallback is not None:
                    action = fallback
                    if status == "bad_json":
                        status = "fallback_from_text_no_json"
                    else:
                        status = "fallback_from_text_bad_json_action"

            if action is None:
                action = "approve"
                status = "fallback_default_approve"

            label = _action_to_label(action)
            if label is None:
                label = 0
                status = "fallback_invalid_action_to_approve"

            responses.append(raw)
            parsed_actions.append(action)
            parsed_reasons.append(reason)
            parse_status.append(status)
            pred_labels.append(int(label))
            if sleep_ms > 0:
                time.sleep(float(sleep_ms) / 1000.0)

        print(f"[llm-ft-infer] processed {end}/{total}")

    out = df.copy()
    out["pred_label"] = pred_labels
    out["pred_action"] = parsed_actions
    out["pred_reason"] = parsed_reasons
    out["parse_status"] = parse_status
    out["llm_response"] = responses

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    y_true = out[TARGET_COL].astype(int)
    y_pred = out["pred_label"].astype(int).values
    y_prob = out["pred_label"].astype(float).values
    metrics = compute_metrics(y_true, y_pred, y_prob)
    business = compute_business_metrics(y_true, y_pred)

    status_counts = out["parse_status"].value_counts(dropna=False).to_dict()
    payload = {
        "test_csv": str(test_csv),
        "output_csv": str(output_csv),
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "device_map": device_map,
        "batch_size": int(batch_n),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "rows": int(len(out)),
        "metrics": metrics,
        "business_metrics": business,
        "parse_status_counts": {str(k): int(v) for k, v in status_counts.items()},
    }
    save_json(output_csv.with_suffix(".json"), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone LLM pipeline on BERT-consistent input with 5%% subset: "
            "teacher SFT data generation, LlamaFactory config, and finetuned inference."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare-data", help="Build stratified train/test subset from BERT-consistent dataset.")
    p_prepare.add_argument("--input-csv", default="output/bert_standalone/data/bert_subset_processed.csv")
    p_prepare.add_argument("--output-dir", default="output/llm_standalone_subset_5pct")
    p_prepare.add_argument("--sample-ratio", type=float, default=0.05)
    p_prepare.add_argument("--random-state", type=int, default=42)
    p_prepare.add_argument("--shared-split-dir", default="data/shared/splits/stratified")
    p_prepare.add_argument("--force-rebuild-shared-split", action=argparse.BooleanOptionalAction, default=False)
    p_prepare.add_argument("--test-size", type=float, default=0.2)
    p_prepare.add_argument("--val-size", type=float, default=0.2)

    p_teacher = sub.add_parser("generate-teacher-sft", help="Generate teacher-labeled JSON SFT data on 5%% train subset.")
    p_teacher.add_argument("--train-csv", default="output/llm_standalone_subset_5pct/data/train_5pct.csv")
    p_teacher.add_argument("--output-jsonl", default="output/llm_standalone_subset_5pct/data/sft/credit_teacher_sft_5pct.jsonl")
    p_teacher.add_argument("--audit-csv", default="output/llm_standalone_subset_5pct/data/sft/credit_teacher_sft_5pct_audit.csv")
    p_teacher.add_argument("--teacher-model-path", default=DEFAULT_QWEN_06B_PATH)
    p_teacher.add_argument("--device-map", default="auto")
    p_teacher.add_argument("--batch-size", type=int, default=4)
    p_teacher.add_argument("--max-new-tokens", type=int, default=256)
    p_teacher.add_argument("--temperature", type=float, default=0.0)
    p_teacher.add_argument("--sleep-ms", type=int, default=0)
    p_teacher.add_argument("--random-state", type=int, default=42)

    p_lf = sub.add_parser("write-lf-config", help="Write LlamaFactory QLoRA train yaml and command files.")
    p_lf.add_argument("--sft-jsonl", default="output/llm_standalone_subset_5pct/data/sft/credit_teacher_sft_5pct.jsonl")
    p_lf.add_argument("--output-dir", default="output/llm_standalone_subset_5pct")
    p_lf.add_argument("--student-model-path", default=DEFAULT_QWEN_06B_PATH)
    p_lf.add_argument("--num-train-epochs", type=float, default=3.0)
    p_lf.add_argument("--per-device-train-batch-size", type=int, default=1)
    p_lf.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p_lf.add_argument("--learning-rate", type=float, default=1e-4)
    p_lf.add_argument("--cutoff-len", type=int, default=1024)
    p_lf.add_argument("--max-samples", type=int, default=None)

    p_infer = sub.add_parser("infer-finetuned", help="Run inference with finetuned 0.6B adapter on 5%% test subset.")
    p_infer.add_argument("--test-csv", default="output/llm_standalone_subset_5pct/data/test_5pct.csv")
    p_infer.add_argument("--output-csv", default="output/llm_standalone_subset_5pct/infer/llm_finetuned_predictions_5pct.csv")
    p_infer.add_argument("--model-path", default=DEFAULT_QWEN_06B_PATH)
    p_infer.add_argument("--adapter-path", default="output/llm_standalone_subset_5pct/qwen3_06b_lora_sft")
    p_infer.add_argument("--device-map", default="auto")
    p_infer.add_argument("--batch-size", type=int, default=4)
    p_infer.add_argument("--max-new-tokens", type=int, default=256)
    p_infer.add_argument("--temperature", type=float, default=0.0)
    p_infer.add_argument("--sleep-ms", type=int, default=0)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "prepare-data":
        report = prepare_subset_data(
            input_csv=Path(args.input_csv),
            output_dir=Path(args.output_dir),
            ratio=args.sample_ratio,
            random_state=args.random_state,
            shared_split_dir=(Path(args.shared_split_dir) if str(args.shared_split_dir).strip() else None),
            force_rebuild_shared_split=bool(args.force_rebuild_shared_split),
            test_size=float(args.test_size),
            val_size=float(args.val_size),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "generate-teacher-sft":
        report = generate_teacher_sft_data(
            train_csv=Path(args.train_csv),
            output_jsonl=Path(args.output_jsonl),
            audit_csv=Path(args.audit_csv),
            teacher_model_path=Path(args.teacher_model_path),
            device_map=args.device_map,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            random_state=args.random_state,
            sleep_ms=args.sleep_ms,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "write-lf-config":
        report = write_lf_train_assets(
            sft_jsonl=Path(args.sft_jsonl),
            output_dir=Path(args.output_dir),
            student_model_path=Path(args.student_model_path),
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            cutoff_len=args.cutoff_len,
            max_samples=args.max_samples,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "infer-finetuned":
        report = infer_with_finetuned_model(
            test_csv=Path(args.test_csv),
            output_csv=Path(args.output_csv),
            model_path=Path(args.model_path),
            adapter_path=Path(args.adapter_path),
            device_map=args.device_map,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            sleep_ms=args.sleep_ms,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
