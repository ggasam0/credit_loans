from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from workflow_common import (  # noqa: E402
    SHARED_ROW_ID_COL,
    TARGET_COL,
    compute_business_metrics,
    compute_metrics,
    evaluate_fixed_reject_rate,
    materialize_or_load_stratified_split,
    save_json,
    stratified_train_val_test_split,
    threshold_by_target_reject_rate,
)

DEFAULT_QWEN_17B_PATH = str((ROOT_DIR / "models" / "Qwen3-1.7B").resolve())
DEFAULT_QWEN_06B_PATH = str((ROOT_DIR / "models" / "Qwen3-0.6B").resolve())

ONE_STAGE_SYSTEM_PROMPT = (
    "You are a bank credit risk reviewer. "
    "Given one loan application input, decide the final action token. "
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


def _label_to_action(label: int) -> str:
    return "reject" if int(label) == 1 else "approve"


def _action_to_label(action: str) -> int:
    return 1 if str(action).strip().lower() == "reject" else 0


def _one_stage_user_prompt(bert_input: str) -> str:
    return (
        "[Input]\n"
        f"{bert_input}\n\n"
        "Output action token only. Allowed values are reject and approve. No explanation."
    )


def _score_action_by_logprob(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    candidate_action: str,
) -> float:
    import torch

    prompt_encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
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
        raise ValueError("Failed to build prompt token ids for scoring.")

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
    }
    save_json(output_dir / "subset_report.json", payload)
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
        student_model_path = (ROOT_DIR / student_model_path).resolve()
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


def generate_one_stage_sft(
    *,
    train_csv: Path,
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
        raise ValueError("No rows available for one-stage SFT generation.")

    action_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    for p in [action_jsonl, audit_csv]:
        if p.exists():
            p.unlink()

    total = int(len(df))
    kept = 0
    with (
        action_jsonl.open("w", encoding="utf-8") as action_writer,
        audit_csv.open("w", encoding="utf-8", newline="") as audit_writer_file,
    ):
        fieldnames = ["row_id", "target", "expected_action", "status"]
        audit_writer = csv.DictWriter(audit_writer_file, fieldnames=fieldnames)
        audit_writer.writeheader()

        for row_id, row in df.iterrows():
            target = int(row[TARGET_COL])
            bert_input = str(row["bert_input"])
            expected_action = _label_to_action(target)

            action_sample = {
                "system": ONE_STAGE_SYSTEM_PROMPT,
                "conversations": [
                    {"from": "human", "value": _one_stage_user_prompt(bert_input)},
                    {"from": "gpt", "value": expected_action},
                ],
            }
            action_writer.write(json.dumps(action_sample, ensure_ascii=False) + "\n")
            audit_writer.writerow(
                {
                    "row_id": int(row_id),
                    "target": target,
                    "expected_action": expected_action,
                    "status": "kept",
                }
            )
            kept += 1
            print(f"[one-stage-sft] processed {kept}/{total}")

    payload = {
        "train_csv": str(train_csv),
        "action_jsonl": str(action_jsonl),
        "audit_csv": str(audit_csv),
        "total_rows": total,
        "kept_rows": kept,
        "dropped_rows": int(total - kept),
    }
    save_json(action_jsonl.with_suffix(".summary.json"), payload)
    return payload


def write_one_stage_lf_config(
    *,
    action_jsonl: Path,
    output_dir: Path,
    student_model_path: Path,
    template: str,
    model_tag: str,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    cutoff_len: int,
    max_samples: int | None,
) -> dict[str, Any]:
    lf_dir = output_dir / "lf_data"
    lf_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_tag}_lora_action_direct"

    action_payload = write_llamafactory_qlora_assets(
        sft_jsonl=action_jsonl,
        dataset_dir=lf_dir,
        dataset_name="credit_teacher_action_direct_json",
        student_model_path=student_model_path,
        output_yaml=lf_dir / f"{run_name}.yaml",
        output_cmd=lf_dir / f"run_{run_name}.ps1",
        output_dir=output_dir / run_name,
        num_train_epochs=num_train_epochs,
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
        "num_train_epochs": float(num_train_epochs),
        "action": action_payload,
    }
    save_json(output_dir / "lf_data" / "one_stage_lf_config.summary.json", payload)
    return payload


def infer_one_stage_action(
    *,
    test_csv: Path,
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
        raise ValueError("No rows available for one-stage action inference.")

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
            user_prompt = _one_stage_user_prompt(str(row["bert_input"]))
            messages = [
                {"role": "system", "content": ONE_STAGE_SYSTEM_PROMPT},
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
                + f"{float(reject_delta_threshold):.6f}; "
                + "max_new_tokens="
                + f"{int(max_new_tokens)}; "
                + "temperature="
                + f"{float(temperature):.6f}"
            )
            reject_scores.append(float(s_reject))
            approve_scores.append(float(s_approve))
            delta_scores.append(delta)
        print(f"[one-stage-action-infer] processed {end}/{total}")

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
    y_prob = 1.0 / (1.0 + np.exp(-out["logprob_delta"].astype(float).values))
    metrics = compute_metrics(y_true, y_pred, y_prob)
    business = compute_business_metrics(y_true, y_pred)
    status_counts = out["parse_status"].value_counts(dropna=False).to_dict()

    payload = {
        "test_csv": str(test_csv),
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
        description="One-stage LLM pipeline: input->action (approve/reject)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare-data", help="Build subset data for one-stage pipeline.")
    p_prepare.add_argument("--input-csv", default="output/bert_standalone/data/bert_subset_processed.csv")
    p_prepare.add_argument("--output-dir", default="output/llm_one_stage_subset_full_06b")
    p_prepare.add_argument("--sample-ratio", type=float, default=1.0)
    p_prepare.add_argument("--random-state", type=int, default=42)
    p_prepare.add_argument("--shared-split-dir", default="data/shared/splits/stratified")
    p_prepare.add_argument("--force-rebuild-shared-split", action=argparse.BooleanOptionalAction, default=False)
    p_prepare.add_argument("--test-size", type=float, default=0.2)
    p_prepare.add_argument("--val-size", type=float, default=0.2)

    p_sft = sub.add_parser("generate-one-stage-sft", help="Generate one-stage teacher SFT data.")
    p_sft.add_argument("--train-csv", default="output/llm_one_stage_subset_full_06b/data/train_full.csv")
    p_sft.add_argument("--action-jsonl", default="output/llm_one_stage_subset_full_06b/data/sft/credit_teacher_action_direct_full.jsonl")
    p_sft.add_argument("--audit-csv", default="output/llm_one_stage_subset_full_06b/data/sft/credit_teacher_action_direct_full_audit.csv")

    p_cfg = sub.add_parser("write-one-stage-lf-config", help="Write one-stage LlamaFactory train config.")
    p_cfg.add_argument("--action-jsonl", default="output/llm_one_stage_subset_full_06b/data/sft/credit_teacher_action_direct_full.jsonl")
    p_cfg.add_argument("--output-dir", default="output/llm_one_stage_subset_full_06b")
    p_cfg.add_argument("--student-model-key", choices=sorted(MODEL_PRESETS.keys()), default="qwen3_0.6b")
    p_cfg.add_argument("--student-model-path", default=None)
    p_cfg.add_argument("--template", default=None, help="Override tokenizer template, e.g. qwen3_nothink.")
    p_cfg.add_argument("--model-tag", default=None, help="Output tag prefix, e.g. qwen3_06b.")
    p_cfg.add_argument("--num-train-epochs", type=float, default=4.0)
    p_cfg.add_argument("--per-device-train-batch-size", type=int, default=1)
    p_cfg.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p_cfg.add_argument("--learning-rate", type=float, default=1e-4)
    p_cfg.add_argument("--cutoff-len", type=int, default=2048)
    p_cfg.add_argument("--max-samples", type=int, default=None)

    p_ainfer = sub.add_parser("infer-one-stage-action", help="One-stage inference: input->action.")
    p_ainfer.add_argument("--test-csv", default="output/llm_one_stage_subset_full_06b/data/test_full.csv")
    p_ainfer.add_argument("--output-csv", default="output/llm_one_stage_subset_full_06b/infer/stage_action_direct_predictions_full.csv")
    p_ainfer.add_argument("--model-key", choices=sorted(MODEL_PRESETS.keys()), default="qwen3_0.6b")
    p_ainfer.add_argument("--model-path", default=None)
    p_ainfer.add_argument("--adapter-path", default="output/llm_one_stage_subset_full_06b/qwen3_06b_lora_action_direct")
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

    if args.command == "generate-one-stage-sft":
        report = generate_one_stage_sft(
            train_csv=Path(args.train_csv),
            action_jsonl=Path(args.action_jsonl),
            audit_csv=Path(args.audit_csv),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "write-one-stage-lf-config":
        student_model_path, template, model_tag = _resolve_model_settings(
            model_key=args.student_model_key,
            model_path=args.student_model_path,
            template=args.template,
            model_tag=args.model_tag,
        )
        report = write_one_stage_lf_config(
            action_jsonl=Path(args.action_jsonl),
            output_dir=Path(args.output_dir),
            student_model_path=student_model_path,
            template=template,
            model_tag=model_tag,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            cutoff_len=args.cutoff_len,
            max_samples=args.max_samples,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "infer-one-stage-action":
        model_path, _, _ = _resolve_model_settings(
            model_key=args.model_key,
            model_path=args.model_path,
            template=None,
            model_tag=None,
        )
        report = infer_one_stage_action(
            test_csv=Path(args.test_csv),
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
