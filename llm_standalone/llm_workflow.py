from __future__ import annotations

import csv
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

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
    make_time_split_with_train_neg_downsample,
    parse_issue_year,
    parse_risk_level,
    save_json,
    split_by_issue_year,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QWEN_17B_PATH = r".\models\Qwen3-1.7B"
DEFAULT_QWEN_06B_PATH = r".\models\Qwen3-0.6B"
DEFAULT_QWEN_LOCAL_PATH = DEFAULT_QWEN_06B_PATH

# Use absolute paths internally so training scripts can `cd LlamaFactory`
# without breaking model resolution.
DEFAULT_QWEN_17B_PATH = str((PROJECT_ROOT / "models" / "Qwen3-1.7B").resolve())
DEFAULT_QWEN_06B_PATH = str((PROJECT_ROOT / "models" / "Qwen3-0.6B").resolve())
DEFAULT_QWEN_LOCAL_PATH = DEFAULT_QWEN_06B_PATH


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: int
    temperature: float


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "missing"
    return text


def _build_case_text(row: pd.Series) -> str:
    desc = _format_value(row.get(TEXT_COL))
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


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    normalized = text.strip()
    normalized = re.sub(r"```json\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("```", "")

    end = normalized.rfind("}")
    if end < 0:
        return None
    start = normalized.rfind("{", 0, end + 1)
    while start >= 0:
        candidate = normalized[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception:
            start = normalized.rfind("{", 0, start)
    return None


def _extract_think_block(text: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    content = match.group(1).strip()
    return content or None


def _decision_to_label(raw_text: str, obj: dict[str, Any] | None) -> int:
    if obj is not None:
        decision = str(obj.get("decision", "")).strip().lower()
        if decision == "reject":
            return 1
        if decision == "approve":
            return 0

        risk_level = str(obj.get("risk_level", "")).strip().lower()
        if risk_level == "high":
            return 1
        if risk_level == "low":
            return 0

        risk = str(obj.get("risk", "")).strip().lower()
        if risk == "high":
            return 1
        if risk == "low":
            return 0

    risk_text = parse_risk_level(raw_text)
    if risk_text == "high":
        return 1
    return 0


def _render_chat_text(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    enable_thinking: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_with_local_model_batch(
    batch_messages: list[list[dict[str, str]]],
    tokenizer: Any,
    model: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> list[str]:
    import torch

    if not batch_messages:
        return []

    texts = [_render_chat_text(messages, tokenizer, enable_thinking=enable_thinking) for messages in batch_messages]
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    )
    if hasattr(model, "device"):
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    do_sample = temperature > 0
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )

    prompt_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
    outputs: list[str] = []
    for i, prompt_len in enumerate(prompt_lengths):
        output_ids = generated_ids[i][int(prompt_len) :]
        outputs.append(tokenizer.decode(output_ids, skip_special_tokens=True).strip())
    return outputs


def _generate_with_local_model(
    messages: list[dict[str, str]],
    tokenizer: Any,
    model: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> str:
    outputs = _generate_with_local_model_batch(
        [messages],
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
    return outputs[0] if outputs else ""


def _ensure_issue_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if ISSUE_YEAR_COL not in out.columns and ISSUE_COL in out.columns:
        out[ISSUE_YEAR_COL] = out[ISSUE_COL].map(parse_issue_year)
    return out


def _sample_binary_by_cap(
    df: pd.DataFrame,
    *,
    target_col: str,
    pos_cap: int | None,
    neg_cap: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    pos_df = df[df[target_col].astype(int) == 1]
    neg_df = df[df[target_col].astype(int) == 0]

    pos_keep = len(pos_df) if pos_cap is None or int(pos_cap) <= 0 else min(int(pos_cap), len(pos_df))
    neg_keep = len(neg_df) if neg_cap is None or int(neg_cap) <= 0 else min(int(neg_cap), len(neg_df))

    if pos_keep < len(pos_df):
        pos_df = pos_df.sample(n=pos_keep, random_state=random_state, replace=False)
    if neg_keep < len(neg_df):
        neg_df = neg_df.sample(n=neg_keep, random_state=random_state, replace=False)

    sampled = (
        pd.concat([pos_df, neg_df], axis=0)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    summary = {
        "total_before": int(len(df)),
        "pos_before": int((df[target_col].astype(int) == 1).sum()),
        "neg_before": int((df[target_col].astype(int) == 0).sum()),
        "pos_after": int((sampled[target_col].astype(int) == 1).sum()),
        "neg_after": int((sampled[target_col].astype(int) == 0).sum()),
    }
    return sampled, summary


def prepare_llm_dataset(
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
        processed["case_text"] = processed.apply(_build_case_text, axis=1)
        processed["llm_input"] = processed["case_text"]

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
            required_non_null_cols=[TARGET_COL, TEXT_COL, ISSUE_YEAR_COL, "llm_input"],
            drop_cols=[STATUS_COL, ISSUE_COL],
            drop_constant_cols=True,
            max_col_missing_ratio=0.98,
            protect_cols=[TARGET_COL, RAW_TEXT_COL, TEXT_COL, ISSUE_YEAR_COL, "llm_input", "case_text"],
        )
        cleaned.to_csv(processed_output_csv, index=False)
        summary["rows_after_cleaning"] = int(len(cleaned))
        summary["cleaning_report"] = cleaning_report
    else:
        summary["rows_after_cleaning"] = int(kept)
        summary["cleaning_report"] = {}

    save_json(processed_output_csv.with_suffix(".summary.json"), summary)
    return summary


def read_llm_config(timeout: int, temperature: float) -> LLMConfig:
    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not base_url or not api_key or not model:
        raise RuntimeError("Missing LLM_BASE_URL / LLM_API_KEY / LLM_MODEL environment variables.")
    return LLMConfig(base_url=base_url, api_key=api_key, model=model, timeout=timeout, temperature=temperature)


def infer_with_remote_llm(llm_input: str, cfg: LLMConfig) -> dict[str, Any]:
    prompt = (
        "You are a credit risk analyst. "
        "Return strict JSON only: "
        '{"decision":"approve|reject","risk_level":"low|medium|high","reasons":["..."]}.\n'
        f"Input:\n{llm_input}"
    )
    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": "You are a precise credit risk assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    resp = requests.post(f"{cfg.base_url}/chat/completions", headers=headers, json=payload, timeout=cfg.timeout)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    obj = _extract_last_json_object(text)
    pred = _decision_to_label(text, obj)
    return {"pred": pred, "raw_response": text, "parsed_json": obj}


def evaluate_llm_remote(
    input_csv: Path,
    output_csv: Path,
    *,
    sample_size: int,
    timeout: int,
    temperature: float,
    sleep_ms: int,
) -> dict[str, Any]:
    cfg = read_llm_config(timeout=timeout, temperature=temperature)
    df = pd.read_csv(input_csv, low_memory=False)
    required = {TARGET_COL, TEXT_COL}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{input_csv} must include columns: {sorted(required)}")
    if "llm_input" not in df.columns:
        if "case_text" in df.columns:
            df["llm_input"] = df["case_text"]
        else:
            df["llm_input"] = df.apply(_build_case_text, axis=1)

    if sample_size and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    preds: list[int] = []
    reasons: list[str] = []

    for idx, row in df.iterrows():
        result = infer_with_remote_llm(str(row["llm_input"]), cfg)
        preds.append(int(result["pred"]))
        reasons.append(result["raw_response"])
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
        if (idx + 1) % 10 == 0:
            print(f"[llm-eval] processed {idx + 1}/{len(df)}")

    out = df[[TEXT_COL, TARGET_COL, "llm_input"]].copy()
    out["pred_label"] = preds
    out["llm_response"] = reasons
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    metrics = compute_metrics(
        y_true=out[TARGET_COL].astype(int),
        y_pred=out["pred_label"].astype(int).values,
        y_prob=out["pred_label"].astype(float).values,
    )
    payload = {"input_csv": str(input_csv), "output_csv": str(output_csv), "sample_size": int(len(out)), "metrics": metrics}
    save_json(output_csv.with_suffix(".json"), payload)
    return payload


def infer_with_local_llm(
    llm_input: str,
    tokenizer: Any,
    model: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> str:
    prompt = _build_local_infer_prompt(llm_input)
    messages = [{"role": "user", "content": prompt}]
    return _generate_with_local_model(
        messages,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )


def _build_local_infer_prompt(llm_input: str) -> str:
    prompt = (
        "You are a credit risk analyst.\n"
        "Return strict JSON only with keys: decision, risk_level, reasons.\n"
        f"Input:\n{llm_input}"
    )
    return prompt


def evaluate_llm_local(
    input_csv: Path,
    output_csv: Path,
    *,
    model_path: Path,
    sample_size: int,
    eval_pos_cap: int | None,
    eval_neg_cap: int | None,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    device_map: str,
    enable_thinking: bool,
    sleep_ms: int,
    random_state: int,
    adapter_path: Path | None,
    train_year_start: int,
    train_year_end: int,
    test_year_start: int,
    test_year_end: int,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pd.read_csv(input_csv, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{input_csv} must include {TARGET_COL}.")
    if "llm_input" not in df.columns:
        if "case_text" in df.columns:
            df["llm_input"] = df["case_text"]
        elif TEXT_COL in df.columns:
            df["llm_input"] = df.apply(_build_case_text, axis=1)
        else:
            raise ValueError(f"{input_csv} must include llm_input or case_text.")

    df = _ensure_issue_year(df)
    try:
        _, eval_df = split_by_issue_year(
            df,
            train_year_start=train_year_start,
            train_year_end=train_year_end,
            test_year_start=test_year_start,
            test_year_end=test_year_end,
        )
        split_mode = f"time_split_test_{test_year_start}_{test_year_end}"
    except Exception:
        eval_df = df.copy()
        split_mode = "full_dataset_fallback"

    eval_df, eval_sampling_summary = _sample_binary_by_cap(
        eval_df,
        target_col=TARGET_COL,
        pos_cap=eval_pos_cap,
        neg_cap=eval_neg_cap,
        random_state=random_state,
    )
    if sample_size and sample_size > 0 and sample_size < len(eval_df):
        eval_df = eval_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

    preds: list[int] = []
    responses: list[str] = []
    batch_n = max(1, int(batch_size))
    total = int(len(eval_df))

    for start in range(0, total, batch_n):
        end = min(start + batch_n, total)
        batch = eval_df.iloc[start:end]
        batch_messages: list[list[dict[str, str]]] = []
        for _, row in batch.iterrows():
            prompt = _build_local_infer_prompt(str(row["llm_input"]))
            batch_messages.append([{"role": "user", "content": prompt}])

        texts = _generate_with_local_model_batch(
            batch_messages,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        for text in texts:
            obj = _extract_last_json_object(text)
            pred = _decision_to_label(text, obj)
            preds.append(pred)
            responses.append(text)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        print(f"[local-llm-eval] processed {end}/{total}")

    out = eval_df[[TARGET_COL, "llm_input"]].copy()
    if TEXT_COL in eval_df.columns:
        out[TEXT_COL] = eval_df[TEXT_COL]
    out["pred_label"] = preds
    out["llm_response"] = responses
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    metrics = compute_metrics(
        y_true=out[TARGET_COL].astype(int),
        y_pred=out["pred_label"].astype(int).values,
        y_prob=out["pred_label"].astype(float).values,
    )
    payload = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "sample_size": int(len(out)),
        "model_path": str(model_path),
        "adapter_path": None if adapter_path is None else str(adapter_path),
        "device_map": device_map,
        "batch_size": int(batch_n),
        "split_mode": split_mode,
        "eval_sampling_summary": eval_sampling_summary,
        "metrics": metrics,
    }
    save_json(output_csv.with_suffix(".json"), payload)
    return payload


def _teacher_system_prompt() -> str:
    return (
        "You are a senior bank credit risk reviewer. "
        "Use only provided fields, no fabrication. "
        "You are given the ground-truth label y in advance. "
        "Explain the evidence supporting that label. "
        "First output <think>...</think> with explicit evidence, then output strict JSON with keys: "
        "decision, risk_level, reasons, action. "
        "decision must align with label mapping: y=1 => reject, y=0 => approve."
    )


def _teacher_user_prompt(case_text: str, label: int) -> str:
    return (
        "[Task]\n"
        "You already know the true label y. Analyze why this case matches y, then output decision aligned with y.\n\n"
        f"[Label]\ny={int(label)}\n\n"
        f"[Case]\n{case_text}\n"
    )


def _student_prompt(case_text: str) -> str:
    return (
        f"{case_text}\n"
        "Analyze risk step by step in <think> and then output strict JSON "
        'with keys: {"decision":"approve|reject","risk_level":"low|medium|high","reasons":[...],"action":"..."}'
    )


def _validate_teacher_output(raw: str, label: int) -> tuple[bool, str, str]:
    think = _extract_think_block(raw)
    if think is None:
        return False, "missing_think", ""

    obj = _extract_last_json_object(raw)
    if obj is None:
        return False, "missing_or_bad_json", ""

    decision = str(obj.get("decision", "")).strip().lower()
    if decision not in {"approve", "reject"}:
        return False, "bad_decision", ""

    if int(label) == 1 and decision != "reject":
        return False, "label_mismatch", ""
    if int(label) == 0 and decision != "approve":
        return False, "label_mismatch", ""

    reasons = obj.get("reasons", [])
    if not isinstance(reasons, list) or len(reasons) == 0:
        return False, "missing_reasons", ""

    normalized = f"<think>\n{think}\n</think>\n\n{json.dumps(obj, ensure_ascii=False)}"
    return True, "ok", normalized


def generate_teacher_reason_sft(
    input_csv: Path,
    output_jsonl: Path,
    audit_csv: Path,
    *,
    teacher_model_path: Path,
    sample_size: int | None,
    random_state: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    device_map: str,
    enable_thinking: bool,
    train_year_start: int,
    train_year_end: int,
    test_year_start: int,
    test_year_end: int,
    train_pos_cap: int | None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pd.read_csv(input_csv, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{input_csv} must include {TARGET_COL}.")
    if TEXT_COL not in df.columns and RAW_TEXT_COL in df.columns:
        df[TEXT_COL] = df[RAW_TEXT_COL].map(clean_text)

    df = _ensure_issue_year(df)
    if "case_text" not in df.columns:
        df["case_text"] = df.apply(_build_case_text, axis=1)

    train_df, _, split_summary = make_time_split_with_train_neg_downsample(
        df,
        random_state=random_state,
        train_year_start=train_year_start,
        train_year_end=train_year_end,
        test_year_start=test_year_start,
        test_year_end=test_year_end,
        target_col=TARGET_COL,
        train_pos_cap=train_pos_cap,
    )
    if sample_size is not None and sample_size > 0 and sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(str(teacher_model_path), trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(teacher_model_path),
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    system_prompt = _teacher_system_prompt()

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        output_jsonl.unlink()
    if audit_csv.exists():
        audit_csv.unlink()

    kept = 0
    failed = 0
    total = int(len(train_df))
    batch_n = max(1, int(batch_size))
    audit_fields = ["row_index", "target", "status", "reason", "raw_response"]
    with output_jsonl.open("w", encoding="utf-8") as writer, audit_csv.open("w", encoding="utf-8", newline="") as audit_writer:
        audit_csv_writer = csv.DictWriter(audit_writer, fieldnames=audit_fields)
        audit_csv_writer.writeheader()
        audit_writer.flush()

        for start in range(0, total, batch_n):
            end = min(start + batch_n, total)
            batch = train_df.iloc[start:end]
            batch_records: list[tuple[int, int, str]] = []
            batch_messages: list[list[dict[str, str]]] = []

            for idx, row in batch.iterrows():
                label = int(row[TARGET_COL])
                case_text = str(row["case_text"])
                user_prompt = _teacher_user_prompt(case_text, label=label)
                batch_records.append((int(idx), label, case_text))
                batch_messages.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )

            batch_raw = _generate_with_local_model_batch(
                batch_messages,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )

            for offset, raw in enumerate(batch_raw):
                row_index, label, case_text = batch_records[offset]
                ok, reason, normalized = _validate_teacher_output(raw, label=label)
                if ok:
                    sample = {
                        "system": "You are a bank credit risk reviewer. Analyze evidence first, then return JSON decision.",
                        "conversations": [
                            {"from": "human", "value": _student_prompt(case_text)},
                            {"from": "gpt", "value": normalized},
                        ],
                    }
                    writer.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    writer.flush()
                    kept += 1
                else:
                    failed += 1

                audit_row = {
                    "row_index": int(row_index),
                    "target": label,
                    "status": "kept" if ok else "dropped",
                    "reason": reason,
                    "raw_response": raw,
                }
                audit_csv_writer.writerow(audit_row)
                audit_writer.flush()

                processed = start + offset + 1
                print(
                    f"[teacher-sft] processed {processed}/{total} "
                    f"(kept={kept}, dropped={failed}, last={reason})"
                )

    payload = {
        "input_csv": str(input_csv),
        "output_jsonl": str(output_jsonl),
        "audit_csv": str(audit_csv),
        "teacher_model_path": str(teacher_model_path),
        "batch_size": int(batch_n),
        "train_rows_after_split_sampling": int(len(train_df)),
        "kept_rows": int(kept),
        "dropped_rows": int(failed),
        "train_sampling_summary": split_summary,
    }
    save_json(output_jsonl.with_suffix(".summary.json"), payload)
    return payload


def write_llamafactory_qlora_assets(
    *,
    sft_jsonl: Path,
    dataset_dir: Path,
    dataset_name: str,
    student_model_path: Path,
    output_yaml: Path,
    output_cmd: Path,
    output_dir: Path,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    cutoff_len: int,
    max_samples: int | None,
    template: str = "qwen3_nothink",
) -> dict[str, Any]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_cmd.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    student_model_path = Path(student_model_path).expanduser()
    if not student_model_path.is_absolute():
        student_model_path = (PROJECT_ROOT / student_model_path).resolve()
    else:
        student_model_path = student_model_path.resolve()
    if not student_model_path.exists():
        raise FileNotFoundError(f"Student model path does not exist: {student_model_path}")

    dataset_file = f"{dataset_name}.jsonl"
    copied_dataset = dataset_dir / dataset_file
    shutil.copy2(sft_jsonl, copied_dataset)

    dataset_info_path = dataset_dir / "dataset_info.json"
    dataset_info: dict[str, Any] = {}
    if dataset_info_path.exists():
        try:
            dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        except Exception:
            dataset_info = {}

    dataset_info[dataset_name] = {
        "file_name": dataset_file,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
        },
    }
    dataset_info_path.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False), encoding="utf-8")

    max_samples_line = f"max_samples: {int(max_samples)}" if max_samples is not None and max_samples > 0 else "# max_samples: null"
    dataset_dir_abs = dataset_dir.resolve()
    output_dir_abs = output_dir.resolve()

    yaml_text = (
        "### model\n"
        f"model_name_or_path: {student_model_path}\n"
        "trust_remote_code: true\n"
        "quantization_bit: 4\n\n"
        "### method\n"
        "stage: sft\n"
        "do_train: true\n"
        "finetuning_type: lora\n"
        "lora_target: all\n"
        "lora_rank: 8\n\n"
        "### dataset\n"
        f"dataset_dir: {dataset_dir_abs}\n"
        f"dataset: {dataset_name}\n"
        f"template: {template}\n"
        f"cutoff_len: {int(cutoff_len)}\n"
        f"{max_samples_line}\n"
        "preprocessing_num_workers: 4\n"
        "dataloader_num_workers: 0\n\n"
        "### output\n"
        f"output_dir: {output_dir_abs}\n"
        "logging_steps: 10\n"
        "save_steps: 200\n"
        "plot_loss: true\n"
        "overwrite_output_dir: true\n"
        "save_only_model: false\n"
        "report_to: none\n\n"
        "### train\n"
        f"per_device_train_batch_size: {int(per_device_train_batch_size)}\n"
        f"gradient_accumulation_steps: {int(gradient_accumulation_steps)}\n"
        f"learning_rate: {learning_rate:.6g}\n"
        f"num_train_epochs: {float(num_train_epochs)}\n"
        "lr_scheduler_type: cosine\n"
        "warmup_ratio: 0.1\n"
        "fp16: true\n"
        "ddp_timeout: 180000000\n"
    )
    output_yaml.write_text(yaml_text, encoding="utf-8")

    cmd_text = (
        "# Run this manually. Do not run while other heavy training is in progress.\n"
        "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path\n"
        "$ProjectRoot = Resolve-Path (Join-Path $ScriptDir \"..\\..\\..\")\n"
        "$LlamaFactoryDir = Join-Path $ProjectRoot \"LlamaFactory\"\n"
        "$PythonExe = Join-Path $ProjectRoot \".venv\\Scripts\\python.exe\"\n"
        "$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)\n"
        "$OutputEncoding = $Utf8NoBom\n"
        "[Console]::InputEncoding = $Utf8NoBom\n"
        "[Console]::OutputEncoding = $Utf8NoBom\n"
        "$env:PYTHONUTF8 = \"1\"\n"
        "$env:PYTHONIOENCODING = \"utf-8\"\n"
        f"$YamlPath = Join-Path $ScriptDir \"{output_yaml.name}\"\n"
        "Set-Location $LlamaFactoryDir\n"
        "if (Test-Path $PythonExe) {\n"
        "  & $PythonExe -X utf8 -m llamafactory.cli train $YamlPath\n"
        "} else {\n"
        "  python -X utf8 -m llamafactory.cli train $YamlPath\n"
        "}\n"
    )
    output_cmd.write_text(cmd_text, encoding="utf-8-sig")

    payload = {
        "sft_jsonl": str(sft_jsonl),
        "dataset_dir": str(dataset_dir_abs),
        "dataset_name": dataset_name,
        "dataset_info_path": str(dataset_info_path),
        "copied_dataset_path": str(copied_dataset),
        "train_yaml": str(output_yaml),
        "train_cmd_file": str(output_cmd),
        "student_model_path": str(student_model_path),
        "output_dir": str(output_dir_abs),
        "template": template,
    }
    save_json(output_yaml.with_suffix(".summary.json"), payload)
    return payload
