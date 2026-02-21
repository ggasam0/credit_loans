from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm_standalone.run_llm_pipeline import (  # noqa: E402
    _build_teacher_reason,
    _extract_last_json_object,
    _label_to_action,
    prepare_subset_data,
)
from llm_standalone.llm_workflow import (  # noqa: E402
    DEFAULT_QWEN_17B_PATH,
    DEFAULT_QWEN_06B_PATH,
    _generate_with_local_model_batch,
    write_llamafactory_qlora_assets,
)
from workflow_common import (  # noqa: E402
    TARGET_COL,
    compute_business_metrics,
    compute_metrics,
    evaluate_fixed_reject_rate,
    save_json,
    threshold_by_target_reject_rate,
)


STAGE1_SYSTEM_PROMPT = (
    "You are a bank credit risk reviewer. "
    "Given one loan application input, provide one concise evidence-based reason paragraph. "
    "Do not output action labels. Do not output markdown."
)

STAGE2_SYSTEM_PROMPT = (
    "You are a bank credit risk reviewer. "
    "Given application input and a reason paragraph, decide the final action token. "
    "Output one token only."
)

MODEL_PRESETS: dict[str, dict[str, str]] = {
    "qwen3_0.6b": {
        "model_path": DEFAULT_QWEN_06B_PATH,
        "template": "qwen3_nothink",
        "model_tag": "qwen3_06b",
    },
    "qwen3_1.7b": {
        "model_path": DEFAULT_QWEN_17B_PATH,
        "template": "qwen3_nothink",
        "model_tag": "qwen3_17b",
    },
}


def _slugify_model_tag(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return cleaned or "custom_model"


def _resolve_model_settings(
    *,
    model_key: str | None,
    model_path: str | Path | None,
    template: str | None,
    model_tag: str | None,
) -> tuple[Path, str, str]:
    preset = MODEL_PRESETS.get(str(model_key)) if model_key else None

    if model_path is not None and str(model_path).strip():
        resolved_model_path = Path(model_path).expanduser()
    elif preset is not None:
        resolved_model_path = Path(preset["model_path"]).expanduser()
    else:
        resolved_model_path = Path(DEFAULT_QWEN_06B_PATH).expanduser()

    if not resolved_model_path.is_absolute():
        resolved_model_path = (ROOT_DIR / resolved_model_path).resolve()
    else:
        resolved_model_path = resolved_model_path.resolve()

    resolved_template = (
        str(template).strip()
        if template is not None and str(template).strip()
        else (preset["template"] if preset is not None else "qwen3_nothink")
    )

    resolved_model_tag = (
        str(model_tag).strip()
        if model_tag is not None and str(model_tag).strip()
        else (preset["model_tag"] if preset is not None else _slugify_model_tag(resolved_model_path.name))
    )

    return resolved_model_path, resolved_template, resolved_model_tag


def _stage2_user_prompt(bert_input: str, reason: str) -> str:
    return (
        "[Input]\n"
        f"{bert_input}\n\n"
        "[Reason]\n"
        f"{reason}\n\n"
        "Output action token only. Allowed values are reject and approve. No explanation."
    )


def _clean_reason_text(raw: str) -> str:
    text = str(raw).strip()
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    obj = _extract_last_json_object(text)
    if obj is not None and isinstance(obj, dict):
        reason = str(obj.get("reason", "")).strip()
        if reason:
            return reason
    return text


def _extract_action(raw: str) -> tuple[str, str]:
    text = str(raw).strip()
    obj = _extract_last_json_object(text)
    if obj is not None and isinstance(obj, dict):
        action = str(obj.get("action", "")).strip().lower()
        if action in {"reject", "approve"}:
            return action, "json_action"

    low = text.lower()
    reject_pos = low.find("reject")
    approve_pos = low.find("approve")
    if reject_pos >= 0 and approve_pos >= 0:
        if reject_pos <= approve_pos:
            return "reject", "text_both_first_reject"
        return "approve", "text_both_first_approve"
    if reject_pos >= 0:
        return "reject", "text_reject"
    if approve_pos >= 0:
        return "approve", "text_approve"
    return "approve", "fallback_default_approve"


def _score_action_by_logprob(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    candidate_action: str,
) -> float:
    import torch

    # Score P(candidate_action | prompt) under causal LM.
    prompt_encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    # `apply_chat_template` may return a plain token-id list, a BatchEncoding,
    # or a tensor depending on tokenizer/version.
    prompt_ids: Any = prompt_encoded
    if hasattr(prompt_encoded, "get"):
        prompt_ids = prompt_encoded.get("input_ids", prompt_encoded)
    elif isinstance(prompt_encoded, dict):
        prompt_ids = prompt_encoded.get("input_ids", prompt_encoded)

    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()
    if isinstance(prompt_ids, list) and prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]

    if not isinstance(prompt_ids, list) or len(prompt_ids) == 0:
        raise ValueError("Failed to build prompt token ids for stage2 scoring.")

    cand_ids = tokenizer(candidate_action, add_special_tokens=False)["input_ids"]
    if not isinstance(cand_ids, list) or len(cand_ids) == 0:
        raise ValueError(f"Failed to tokenize candidate action: {candidate_action}")

    all_ids = prompt_ids + cand_ids
    device = next(model.parameters()).device
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

    prompt_len = len(prompt_ids)
    score = 0.0
    for j, tok in enumerate(cand_ids):
        pos = prompt_len + j - 1
        score += float(log_probs[pos, tok].item())
    return score


def _action_to_label(action: str) -> int:
    return 1 if str(action).strip().lower() == "reject" else 0


def generate_two_stage_sft(
    *,
    train_csv: Path,
    reason_jsonl: Path,
    action_jsonl: Path,
    audit_csv: Path,
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
        raise ValueError("No rows available for two-stage SFT generation.")

    reason_jsonl.parent.mkdir(parents=True, exist_ok=True)
    action_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    for p in [reason_jsonl, action_jsonl, audit_csv]:
        if p.exists():
            p.unlink()

    total = int(len(df))
    kept = 0

    with (
        reason_jsonl.open("w", encoding="utf-8") as reason_writer,
        action_jsonl.open("w", encoding="utf-8") as action_writer,
        audit_csv.open("w", encoding="utf-8", newline="") as audit_writer_file,
    ):
        fieldnames = [
            "row_id",
            "target",
            "expected_action",
            "status",
            "reason_source",
            "risk_signal_count",
            "strength_signal_count",
            "reason_text",
        ]
        audit_writer = csv.DictWriter(audit_writer_file, fieldnames=fieldnames)
        audit_writer.writeheader()

        for row_id, row in df.iterrows():
            target = int(row[TARGET_COL])
            bert_input = str(row["bert_input"])
            expected_action = _label_to_action(target)
            reason, meta = _build_teacher_reason(bert_input=bert_input, target=target)

            reason_sample = {
                "system": STAGE1_SYSTEM_PROMPT,
                "conversations": [
                    {"from": "human", "value": bert_input},
                    {"from": "gpt", "value": reason},
                ],
            }
            action_sample = {
                "system": STAGE2_SYSTEM_PROMPT,
                "conversations": [
                    {"from": "human", "value": _stage2_user_prompt(bert_input, reason)},
                    {"from": "gpt", "value": expected_action},
                ],
            }
            reason_writer.write(json.dumps(reason_sample, ensure_ascii=False) + "\n")
            action_writer.write(json.dumps(action_sample, ensure_ascii=False) + "\n")
            audit_writer.writerow(
                {
                    "row_id": int(row_id),
                    "target": target,
                    "expected_action": expected_action,
                    "status": "kept",
                    "reason_source": "codex_teacher_rules",
                    "risk_signal_count": int(meta.get("risk_signal_count", 0)),
                    "strength_signal_count": int(meta.get("strength_signal_count", 0)),
                    "reason_text": reason,
                }
            )
            kept += 1
            print(f"[two-stage-sft] processed {kept}/{total}")

    payload = {
        "train_csv": str(train_csv),
        "reason_jsonl": str(reason_jsonl),
        "action_jsonl": str(action_jsonl),
        "audit_csv": str(audit_csv),
        "total_rows": total,
        "kept_rows": kept,
        "dropped_rows": int(total - kept),
    }
    save_json(reason_jsonl.with_suffix(".summary.json"), payload)
    return payload


def write_two_stage_lf_configs(
    *,
    reason_jsonl: Path,
    action_jsonl: Path,
    output_dir: Path,
    student_model_path: Path,
    template: str,
    model_tag: str,
    reason_num_train_epochs: float,
    action_num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    cutoff_len: int,
    max_samples: int | None,
) -> dict[str, Any]:
    lf_dir = output_dir / "lf_data"
    lf_dir.mkdir(parents=True, exist_ok=True)
    reason_run_name = f"{model_tag}_lora_reason"
    action_run_name = f"{model_tag}_lora_action"

    reason_payload = write_llamafactory_qlora_assets(
        sft_jsonl=reason_jsonl,
        dataset_dir=lf_dir,
        dataset_name="credit_teacher_reason_json",
        student_model_path=student_model_path,
        output_yaml=lf_dir / f"{reason_run_name}.yaml",
        output_cmd=lf_dir / f"run_{reason_run_name}.ps1",
        output_dir=output_dir / reason_run_name,
        num_train_epochs=reason_num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        max_samples=max_samples,
        template=template,
    )
    action_payload = write_llamafactory_qlora_assets(
        sft_jsonl=action_jsonl,
        dataset_dir=lf_dir,
        dataset_name="credit_teacher_action_json",
        student_model_path=student_model_path,
        output_yaml=lf_dir / f"{action_run_name}.yaml",
        output_cmd=lf_dir / f"run_{action_run_name}.ps1",
        output_dir=output_dir / action_run_name,
        num_train_epochs=action_num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        max_samples=max_samples,
        template=template,
    )
    payload = {
        "model_tag": model_tag,
        "student_model_path": str(student_model_path),
        "template": template,
        "reason_num_train_epochs": float(reason_num_train_epochs),
        "action_num_train_epochs": float(action_num_train_epochs),
        "reason": reason_payload,
        "action": action_payload,
    }
    save_json(output_dir / "lf_data" / "two_stage_lf_config.summary.json", payload)
    return payload


def infer_reason(
    *,
    test_csv: Path,
    output_csv: Path,
    model_path: Path,
    adapter_path: Path,
    device_map: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
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
        raise ValueError("No rows available for reason inference.")

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
    pred_reason: list[str] = []
    raw_output: list[str] = []

    for start in range(0, total, batch_n):
        end = min(start + batch_n, total)
        batch = df.iloc[start:end]
        batch_messages: list[list[dict[str, str]]] = []
        for _, row in batch.iterrows():
            batch_messages.append(
                [
                    {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                    {"role": "user", "content": str(row["bert_input"])},
                ]
            )
        outs = _generate_with_local_model_batch(
            batch_messages,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=False,
        )
        for raw in outs:
            raw_output.append(raw)
            pred_reason.append(_clean_reason_text(raw))
        print(f"[stage1-reason-infer] processed {end}/{total}")

    out = df.copy()
    out["pred_reason"] = pred_reason
    out["raw_reason_output"] = raw_output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    payload = {
        "test_csv": str(test_csv),
        "output_csv": str(output_csv),
        "rows": int(len(out)),
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
    }
    save_json(output_csv.with_suffix(".json"), payload)
    return payload


def infer_action(
    *,
    reason_pred_csv: Path,
    output_csv: Path,
    model_path: Path,
    adapter_path: Path,
    device_map: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    reject_delta_threshold: float = 0.0,
    target_reject_rate: float | None = None,
) -> dict[str, Any]:
    import numpy as np

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pd.read_csv(reason_pred_csv, low_memory=False)
    required_cols = [TARGET_COL, "bert_input", "pred_reason"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{reason_pred_csv} missing required columns: {missing}")

    df = df.copy().reset_index(drop=True)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df["bert_input"] = df["bert_input"].fillna("").astype(str)
    df["pred_reason"] = df["pred_reason"].fillna("").astype(str)
    df = df[(df["bert_input"].str.len() > 0) & (df["pred_reason"].str.len() > 0)].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows available for action inference.")

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
    pred_action: list[str] = []
    pred_label: list[int] = []
    parse_status: list[str] = []
    raw_output: list[str] = []
    reject_scores: list[float] = []
    approve_scores: list[float] = []
    delta_scores: list[float] = []

    for start in range(0, total, batch_n):
        end = min(start + batch_n, total)
        batch = df.iloc[start:end]
        for _, row in batch.iterrows():
            user_prompt = _stage2_user_prompt(str(row["bert_input"]), str(row["pred_reason"]))
            messages = [
                {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            s_reject = _score_action_by_logprob(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                candidate_action="reject",
            )
            s_approve = _score_action_by_logprob(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                candidate_action="approve",
            )
            delta = float(s_reject - s_approve)
            action = "reject" if delta >= float(reject_delta_threshold) else "approve"
            pred_action.append(action)
            pred_label.append(_action_to_label(action))
            parse_status.append("logprob_candidate_compare")
            raw_output.append(
                "logprob(reject)="
                + f"{s_reject:.6f}; "
                + "logprob(approve)="
                + f"{s_approve:.6f}; "
                + "delta="
                + f"{delta:.6f}; "
                + "threshold="
                + f"{float(reject_delta_threshold):.6f}"
            )
            reject_scores.append(float(s_reject))
            approve_scores.append(float(s_approve))
            delta_scores.append(delta)
        print(f"[stage2-action-infer] processed {end}/{total}")

    out = df.copy()
    out["pred_action"] = pred_action
    out["pred_label"] = pred_label
    out["parse_status"] = parse_status
    out["raw_action_output"] = raw_output
    out["logprob_reject"] = reject_scores
    out["logprob_approve"] = approve_scores
    out["logprob_delta"] = delta_scores
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    y_true = out[TARGET_COL].astype(int)
    y_pred = out["pred_label"].astype(int).values
    # Convert logprob delta to a calibrated-like score in [0, 1].
    y_prob = 1.0 / (1.0 + np.exp(-out["logprob_delta"].astype(float).values))
    metrics = compute_metrics(y_true, y_pred, y_prob)
    business = compute_business_metrics(y_true, y_pred)
    status_counts = out["parse_status"].value_counts(dropna=False).to_dict()

    payload = {
        "reason_pred_csv": str(reason_pred_csv),
        "output_csv": str(output_csv),
        "rows": int(len(out)),
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "metrics": metrics,
        "business_metrics": business,
        "parse_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "inference_mode": "logprob_candidate_compare",
        "reject_delta_threshold": float(reject_delta_threshold),
    }

    if target_reject_rate is not None:
        rr = float(max(0.0, min(1.0, target_reject_rate)))
        th_info = threshold_by_target_reject_rate(y_prob, target_reject_rate=rr)
        rr_threshold = float(th_info["threshold"])
        fixed_pred = (y_prob >= rr_threshold).astype(int)
        fixed_out = out.copy()
        fixed_out["pred_label_fixed_rr"] = fixed_pred
        fixed_out["pred_action_fixed_rr"] = fixed_out["pred_label_fixed_rr"].map({1: "reject", 0: "approve"})
        fixed_csv = output_csv.with_name(output_csv.stem + f"_fixed_rr_{int(round(rr * 100))}pct.csv")
        fixed_out.to_csv(fixed_csv, index=False)
        payload["fixed_reject_rate_eval"] = evaluate_fixed_reject_rate(
            y_true,
            y_prob,
            target_reject_rate=rr,
            threshold=rr_threshold,
        )
        payload["fixed_reject_rate_pred_csv"] = str(fixed_csv)

    save_json(output_csv.with_suffix(".json"), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-stage LLM pipeline: stage1 input->reason, stage2 input+reason->action."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare-data", help="Build subset data for two-stage pipeline.")
    p_prepare.add_argument("--input-csv", default="output/bert_standalone/data/bert_subset_processed.csv")
    p_prepare.add_argument("--output-dir", default="output/llm_two_stage_subset_5pct")
    p_prepare.add_argument("--sample-ratio", type=float, default=0.05)
    p_prepare.add_argument("--random-state", type=int, default=42)
    p_prepare.add_argument("--shared-split-dir", default="data/shared/splits/stratified")
    p_prepare.add_argument("--force-rebuild-shared-split", action=argparse.BooleanOptionalAction, default=False)
    p_prepare.add_argument("--test-size", type=float, default=0.2)
    p_prepare.add_argument("--val-size", type=float, default=0.2)

    p_sft = sub.add_parser("generate-two-stage-sft", help="Generate two-stage teacher SFT data.")
    p_sft.add_argument("--train-csv", default="output/llm_two_stage_subset_5pct/data/train_5pct.csv")
    p_sft.add_argument("--reason-jsonl", default="output/llm_two_stage_subset_5pct/data/sft/credit_teacher_reason_5pct.jsonl")
    p_sft.add_argument("--action-jsonl", default="output/llm_two_stage_subset_5pct/data/sft/credit_teacher_action_5pct.jsonl")
    p_sft.add_argument("--audit-csv", default="output/llm_two_stage_subset_5pct/data/sft/credit_teacher_two_stage_5pct_audit.csv")

    p_cfg = sub.add_parser("write-two-stage-lf-config", help="Write two-stage LlamaFactory train configs.")
    p_cfg.add_argument("--reason-jsonl", default="output/llm_two_stage_subset_5pct/data/sft/credit_teacher_reason_5pct.jsonl")
    p_cfg.add_argument("--action-jsonl", default="output/llm_two_stage_subset_5pct/data/sft/credit_teacher_action_5pct.jsonl")
    p_cfg.add_argument("--output-dir", default="output/llm_two_stage_subset_5pct")
    p_cfg.add_argument("--student-model-key", choices=sorted(MODEL_PRESETS.keys()), default="qwen3_0.6b")
    p_cfg.add_argument("--student-model-path", default=None)
    p_cfg.add_argument("--template", default=None, help="Override tokenizer template, e.g. qwen3_nothink.")
    p_cfg.add_argument("--model-tag", default=None, help="Output tag prefix, e.g. qwen3_06b.")
    p_cfg.add_argument("--num-train-epochs", type=float, default=3.0)
    p_cfg.add_argument("--reason-num-train-epochs", type=float, default=None)
    p_cfg.add_argument("--action-num-train-epochs", type=float, default=None)
    p_cfg.add_argument("--per-device-train-batch-size", type=int, default=1)
    p_cfg.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p_cfg.add_argument("--learning-rate", type=float, default=1e-4)
    p_cfg.add_argument("--cutoff-len", type=int, default=2048)
    p_cfg.add_argument("--max-samples", type=int, default=None)

    p_rinfer = sub.add_parser("infer-reason", help="Stage1 inference: input->reason.")
    p_rinfer.add_argument("--test-csv", default="output/llm_two_stage_subset_5pct/data/test_5pct.csv")
    p_rinfer.add_argument("--output-csv", default="output/llm_two_stage_subset_5pct/infer/stage1_reason_predictions_5pct.csv")
    p_rinfer.add_argument("--model-key", choices=sorted(MODEL_PRESETS.keys()), default="qwen3_0.6b")
    p_rinfer.add_argument("--model-path", default=None)
    p_rinfer.add_argument("--adapter-path", default="output/llm_two_stage_subset_5pct/qwen3_06b_lora_reason")
    p_rinfer.add_argument("--device-map", default="auto")
    p_rinfer.add_argument("--batch-size", type=int, default=4)
    p_rinfer.add_argument("--max-new-tokens", type=int, default=256)
    p_rinfer.add_argument("--temperature", type=float, default=0.0)

    p_ainfer = sub.add_parser("infer-action", help="Stage2 inference: input+reason->action.")
    p_ainfer.add_argument("--reason-pred-csv", default="output/llm_two_stage_subset_5pct/infer/stage1_reason_predictions_5pct.csv")
    p_ainfer.add_argument("--output-csv", default="output/llm_two_stage_subset_5pct/infer/stage2_action_predictions_5pct.csv")
    p_ainfer.add_argument("--model-key", choices=sorted(MODEL_PRESETS.keys()), default="qwen3_0.6b")
    p_ainfer.add_argument("--model-path", default=None)
    p_ainfer.add_argument("--adapter-path", default="output/llm_two_stage_subset_5pct/qwen3_06b_lora_action")
    p_ainfer.add_argument("--device-map", default="auto")
    p_ainfer.add_argument("--batch-size", type=int, default=4)
    p_ainfer.add_argument("--max-new-tokens", type=int, default=16)
    p_ainfer.add_argument("--temperature", type=float, default=0.0)
    p_ainfer.add_argument("--reject-delta-threshold", type=float, default=0.0)
    p_ainfer.add_argument("--target-reject-rate", type=float, default=None)

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

    if args.command == "generate-two-stage-sft":
        report = generate_two_stage_sft(
            train_csv=Path(args.train_csv),
            reason_jsonl=Path(args.reason_jsonl),
            action_jsonl=Path(args.action_jsonl),
            audit_csv=Path(args.audit_csv),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "write-two-stage-lf-config":
        student_model_path, template, model_tag = _resolve_model_settings(
            model_key=args.student_model_key,
            model_path=args.student_model_path,
            template=args.template,
            model_tag=args.model_tag,
        )
        report = write_two_stage_lf_configs(
            reason_jsonl=Path(args.reason_jsonl),
            action_jsonl=Path(args.action_jsonl),
            output_dir=Path(args.output_dir),
            student_model_path=student_model_path,
            template=template,
            model_tag=model_tag,
            reason_num_train_epochs=(args.reason_num_train_epochs if args.reason_num_train_epochs is not None else args.num_train_epochs),
            action_num_train_epochs=(args.action_num_train_epochs if args.action_num_train_epochs is not None else args.num_train_epochs),
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            cutoff_len=args.cutoff_len,
            max_samples=args.max_samples,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "infer-reason":
        model_path, _, _ = _resolve_model_settings(
            model_key=args.model_key,
            model_path=args.model_path,
            template=None,
            model_tag=None,
        )
        report = infer_reason(
            test_csv=Path(args.test_csv),
            output_csv=Path(args.output_csv),
            model_path=model_path,
            adapter_path=Path(args.adapter_path),
            device_map=args.device_map,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "infer-action":
        model_path, _, _ = _resolve_model_settings(
            model_key=args.model_key,
            model_path=args.model_path,
            template=None,
            model_tag=None,
        )
        report = infer_action(
            reason_pred_csv=Path(args.reason_pred_csv),
            output_csv=Path(args.output_csv),
            model_path=model_path,
            adapter_path=Path(args.adapter_path),
            device_map=args.device_map,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            reject_delta_threshold=args.reject_delta_threshold,
            target_reject_rate=args.target_reject_rate,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())





