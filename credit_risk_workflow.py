from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bert_standalone.bert_workflow import (
    DEFAULT_BCE_EMBED_PATH,
    evaluate_bert_embedding_local,
    prepare_bert_dataset,
    train_and_evaluate_bert_finetune,
)
from llm_standalone.llm_workflow import (
    DEFAULT_QWEN_06B_PATH,
    DEFAULT_QWEN_17B_PATH,
    DEFAULT_QWEN_LOCAL_PATH,
    evaluate_llm_local,
    evaluate_llm_remote,
    generate_teacher_reason_sft,
    prepare_llm_dataset,
    write_llamafactory_qlora_assets,
)
from ml_standalone.ml_workflow import prepare_ml_dataset, train_ml_models
from ml_standalone.shared_subset_workflow import build_shared_subset_from_raw
from workflow_common import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Credit-risk workflow with separated implementations: ML / BERT / LLM."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare_ml = sub.add_parser("prepare-ml", help="Prepare raw/processed datasets for traditional ML models.")
    p_prepare_ml.add_argument("--input", default="data/accepted_2007_to_2018Q4.csv")
    p_prepare_ml.add_argument("--raw-output", default="ml_standalone/data/raw/ml_raw.csv")
    p_prepare_ml.add_argument("--processed-output", default="ml_standalone/data/processed/ml_processed.csv")
    p_prepare_ml.add_argument("--chunksize", type=int, default=120000)
    p_prepare_ml.add_argument("--max-rows", type=int, default=180000)
    p_prepare_ml.add_argument("--max-chunks", type=int, default=None)

    p_build_shared = sub.add_parser(
        "build-shared-subset",
        help="Rebuild unified shared subset from raw accepted CSV (prepare-ml + deterministic subset rules).",
    )
    p_build_shared.add_argument("--input", default="data/accepted_2007_to_2018Q4.csv")
    p_build_shared.add_argument("--raw-output", default="ml_standalone/data/raw/ml_raw.csv")
    p_build_shared.add_argument("--processed-output", default="ml_standalone/data/processed/ml_processed.csv")
    p_build_shared.add_argument("--shared-output", default="data/shared/shared_subset.csv")
    p_build_shared.add_argument(
        "--ml-subset-output",
        default="ml_standalone/data/processed/ml_subset_precision_high.csv",
        help="Compatibility alias used by legacy scripts.",
    )
    p_build_shared.add_argument("--chunksize", type=int, default=120000)
    p_build_shared.add_argument("--max-rows", type=int, default=180000)
    p_build_shared.add_argument("--max-chunks", type=int, default=None)
    p_build_shared.add_argument(
        "--allowed-grades",
        nargs="+",
        default=["D", "E", "F", "G"],
        help="Risk-grade whitelist used for shared subset filtering.",
    )
    p_build_shared.add_argument("--min-int-rate", type=float, default=16.0)
    p_build_shared.add_argument("--max-annual-inc", type=float, default=50000.0)
    p_build_shared.add_argument(
        "--keep-ml-intermediate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep or clean ml_standalone/data/raw + ml_standalone/data/processed intermediate files (default keep).",
    )
    p_build_shared.add_argument(
        "--write-ml-subset-alias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to also write legacy alias file ml_standalone/data/processed/ml_subset_precision_high.csv.",
    )

    p_train_ml = sub.add_parser("train-ml", help="Train traditional ML baselines (default input: shared subset).")
    p_train_ml.add_argument("--input", default="data/shared/shared_subset.csv")
    p_train_ml.add_argument("--output-dir", default="output/models")
    p_train_ml.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional cap for positive-class train rows (2007-2013). Default uses all positives.",
    )
    p_train_ml.add_argument("--test-size", type=float, default=0.2)
    p_train_ml.add_argument("--precision-floor", type=float, default=0.3)
    p_train_ml.add_argument("--balance-test", action=argparse.BooleanOptionalAction, default=True)
    p_train_ml.add_argument("--test-pos-cap", type=int, default=None)
    p_train_ml.add_argument("--cv-folds", type=int, default=3)
    p_train_ml.add_argument("--random-state", type=int, default=42)
    p_train_ml.add_argument("--text-max-features", type=int, default=40000)
    p_train_ml.add_argument("--model", choices=("xgboost", "logistic"), default="xgboost")
    p_train_ml.add_argument("--n-jobs", type=int, default=-1)
    p_train_ml.add_argument("--train-year-start", type=int, default=2007)
    p_train_ml.add_argument("--train-year-end", type=int, default=2013)
    p_train_ml.add_argument("--test-year-start", type=int, default=2014)
    p_train_ml.add_argument("--test-year-end", type=int, default=2016)

    p_prepare_bert = sub.add_parser("prepare-bert", help="Prepare raw/processed datasets for BERT.")
    p_prepare_bert.add_argument("--input", default="data/accepted_2007_to_2018Q4.csv")
    p_prepare_bert.add_argument("--raw-output", default="bert_standalone/data/raw/bert_raw.csv")
    p_prepare_bert.add_argument("--processed-output", default="bert_standalone/data/processed/bert_processed.csv")
    p_prepare_bert.add_argument("--chunksize", type=int, default=120000)
    p_prepare_bert.add_argument("--max-rows", type=int, default=180000)
    p_prepare_bert.add_argument("--max-chunks", type=int, default=None)

    p_bert = sub.add_parser("bert-eval", help="Evaluate local BCE embedding model only (no rerank).")
    p_bert.add_argument("--input", default="bert_standalone/data/processed/bert_processed.csv")
    p_bert.add_argument("--output-dir", default="output/bert")
    p_bert.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional cap for positive-class train rows (2007-2013). Default uses all positives.",
    )
    p_bert.add_argument("--test-size", type=float, default=0.2)
    p_bert.add_argument("--precision-floor", type=float, default=0.3)
    p_bert.add_argument("--balance-test", action=argparse.BooleanOptionalAction, default=True)
    p_bert.add_argument("--test-pos-cap", type=int, default=None)
    p_bert.add_argument("--random-state", type=int, default=42)
    p_bert.add_argument("--device", default="cuda")
    p_bert.add_argument("--max-length", type=int, default=514)
    p_bert.add_argument("--embedding-model-path", default=DEFAULT_BCE_EMBED_PATH)
    p_bert.add_argument("--train-year-start", type=int, default=2007)
    p_bert.add_argument("--train-year-end", type=int, default=2013)
    p_bert.add_argument("--test-year-start", type=int, default=2014)
    p_bert.add_argument("--test-year-end", type=int, default=2016)

    p_bert_ft = sub.add_parser("bert-finetune", help="Train and evaluate fine-tuned BERT classifier.")
    p_bert_ft.add_argument("--input", default="bert_standalone/data/processed/bert_processed.csv")
    p_bert_ft.add_argument("--output-dir", default="output/bert_finetune")
    p_bert_ft.add_argument("--model-path", default=DEFAULT_BCE_EMBED_PATH)
    p_bert_ft.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional cap for positive-class train rows (2007-2013). Default uses all positives.",
    )
    p_bert_ft.add_argument("--test-size", type=float, default=0.2)
    p_bert_ft.add_argument("--precision-floor", type=float, default=0.3)
    p_bert_ft.add_argument("--balance-test", action=argparse.BooleanOptionalAction, default=True)
    p_bert_ft.add_argument("--test-pos-cap", type=int, default=None)
    p_bert_ft.add_argument("--random-state", type=int, default=42)
    p_bert_ft.add_argument("--max-length", type=int, default=514)
    p_bert_ft.add_argument("--learning-rate", type=float, default=2e-5)
    p_bert_ft.add_argument("--train-batch-size", type=int, default=8)
    p_bert_ft.add_argument("--eval-batch-size", type=int, default=16)
    p_bert_ft.add_argument("--num-train-epochs", type=float, default=1.0)
    p_bert_ft.add_argument("--weight-decay", type=float, default=0.01)
    p_bert_ft.add_argument("--logging-steps", type=int, default=20)
    p_bert_ft.add_argument("--save-total-limit", type=int, default=2)
    p_bert_ft.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    p_bert_ft.add_argument("--train-year-start", type=int, default=2007)
    p_bert_ft.add_argument("--train-year-end", type=int, default=2013)
    p_bert_ft.add_argument("--test-year-start", type=int, default=2014)
    p_bert_ft.add_argument("--test-year-end", type=int, default=2016)

    p_prepare_llm = sub.add_parser("prepare-llm", help="Prepare raw/processed datasets for LLM.")
    p_prepare_llm.add_argument("--input", default="data/accepted_2007_to_2018Q4.csv")
    p_prepare_llm.add_argument("--raw-output", default="llm_standalone/data/raw/llm_raw.csv")
    p_prepare_llm.add_argument("--processed-output", default="llm_standalone/data/processed/llm_processed.csv")
    p_prepare_llm.add_argument("--chunksize", type=int, default=120000)
    p_prepare_llm.add_argument("--max-rows", type=int, default=180000)
    p_prepare_llm.add_argument("--max-chunks", type=int, default=None)

    p_llm = sub.add_parser("llm-eval", help="Evaluate remote prompt LLM on LLM processed data.")
    p_llm.add_argument("--input", default="llm_standalone/data/processed/llm_processed.csv")
    p_llm.add_argument("--output", default="output/llm/llm_eval_predictions.csv")
    p_llm.add_argument("--sample-size", type=int, default=300)
    p_llm.add_argument("--timeout", type=int, default=60)
    p_llm.add_argument("--temperature", type=float, default=0.0)
    p_llm.add_argument("--sleep-ms", type=int, default=0)

    p_local_llm = sub.add_parser("local-llm-eval", help="Evaluate local Qwen model on LLM processed data.")
    p_local_llm.add_argument("--input", default="llm_standalone/data/processed/llm_processed.csv")
    p_local_llm.add_argument("--output", default="output/llm/local_qwen_eval_predictions.csv")
    p_local_llm.add_argument("--sample-size", type=int, default=0)
    p_local_llm.add_argument("--eval-pos-cap", type=int, default=1000)
    p_local_llm.add_argument("--eval-neg-cap", type=int, default=1000)
    p_local_llm.add_argument("--batch-size", type=int, default=4)
    p_local_llm.add_argument("--random-state", type=int, default=42)
    p_local_llm.add_argument("--model-path", default=DEFAULT_QWEN_LOCAL_PATH)
    p_local_llm.add_argument("--adapter-path", default=None)
    p_local_llm.add_argument("--max-new-tokens", type=int, default=256)
    p_local_llm.add_argument("--temperature", type=float, default=0.0)
    p_local_llm.add_argument("--device-map", default="auto")
    p_local_llm.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    p_local_llm.add_argument("--sleep-ms", type=int, default=0)
    p_local_llm.add_argument("--train-year-start", type=int, default=2007)
    p_local_llm.add_argument("--train-year-end", type=int, default=2013)
    p_local_llm.add_argument("--test-year-start", type=int, default=2014)
    p_local_llm.add_argument("--test-year-end", type=int, default=2016)

    p_teacher = sub.add_parser(
        "llm-generate-teacher-sft",
        help="Use local Qwen teacher (0.6B/1.7B) to generate <think> + JSON ShareGPT SFT data.",
    )
    p_teacher.add_argument("--input", default="llm_standalone/data/processed/llm_processed.csv")
    p_teacher.add_argument("--output-jsonl", default="llm_standalone/data/sft/credit_teacher_sft_en.jsonl")
    p_teacher.add_argument("--audit-csv", default="llm_standalone/data/sft/credit_teacher_sft_en_audit.csv")
    p_teacher.add_argument("--teacher-model-size", choices=("0.6b", "1.7b"), default="0.6b")
    p_teacher.add_argument(
        "--teacher-model-path",
        default=None,
        help="Optional explicit teacher model path. Overrides --teacher-model-size.",
    )
    p_teacher.add_argument("--sample-size", type=int, default=None)
    p_teacher.add_argument(
        "--train-pos-cap",
        type=int,
        default=2000,
        help="Positive-class cap for train split. Negatives are downsampled to match (default: 2000+2000).",
    )
    p_teacher.add_argument("--batch-size", type=int, default=4)
    p_teacher.add_argument("--random-state", type=int, default=42)
    p_teacher.add_argument("--max-new-tokens", type=int, default=768)
    p_teacher.add_argument("--temperature", type=float, default=0.2)
    p_teacher.add_argument("--device-map", default="auto")
    p_teacher.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    p_teacher.add_argument("--train-year-start", type=int, default=2007)
    p_teacher.add_argument("--train-year-end", type=int, default=2013)
    p_teacher.add_argument("--test-year-start", type=int, default=2014)
    p_teacher.add_argument("--test-year-end", type=int, default=2016)

    p_lf = sub.add_parser(
        "llm-write-lf-config",
        help="Write LlamaFactory dataset_info + QLoRA yaml/command (no training execution).",
    )
    p_lf.add_argument("--sft-jsonl", default="llm_standalone/data/sft/credit_teacher_sft_en.jsonl")
    p_lf.add_argument("--dataset-dir", default="llm_standalone/lf_data")
    p_lf.add_argument("--dataset-name", default="credit_teacher_sft_en")
    p_lf.add_argument("--student-model-path", default=DEFAULT_QWEN_06B_PATH)
    p_lf.add_argument("--output-yaml", default="llm_standalone/lf_data/qwen3_06b_credit_qlora_sft.yaml")
    p_lf.add_argument("--output-cmd", default="llm_standalone/lf_data/run_qwen3_06b_credit_qlora_train.ps1")
    p_lf.add_argument("--output-dir", default="output/llm/qwen3_06b_lora_sft")
    p_lf.add_argument("--num-train-epochs", type=float, default=2.0)
    p_lf.add_argument("--per-device-train-batch-size", type=int, default=1)
    p_lf.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p_lf.add_argument("--learning-rate", type=float, default=1e-4)
    p_lf.add_argument("--cutoff-len", type=int, default=1024)
    p_lf.add_argument("--max-samples", type=int, default=4000)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.command == "prepare-ml":
            summary = prepare_ml_dataset(
                input_csv=Path(args.input),
                raw_output_csv=Path(args.raw_output),
                processed_output_csv=Path(args.processed_output),
                chunksize=args.chunksize,
                max_rows=args.max_rows,
                max_chunks=args.max_chunks,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            save_json(Path(args.processed_output).with_suffix(".summary.json"), summary)
            return 0

        if args.command == "build-shared-subset":
            payload = build_shared_subset_from_raw(
                input_csv=Path(args.input),
                raw_output_csv=Path(args.raw_output),
                processed_output_csv=Path(args.processed_output),
                shared_output_csv=Path(args.shared_output),
                ml_subset_output_csv=Path(args.ml_subset_output),
                chunksize=args.chunksize,
                max_rows=args.max_rows,
                max_chunks=args.max_chunks,
                allowed_grades=args.allowed_grades,
                min_int_rate=args.min_int_rate,
                max_annual_inc=args.max_annual_inc,
                keep_ml_intermediate=bool(args.keep_ml_intermediate),
                write_ml_subset_alias=bool(args.write_ml_subset_alias),
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if args.command == "train-ml":
            report = train_ml_models(
                input_csv=Path(args.input),
                output_dir=Path(args.output_dir),
                sample_size=args.sample_size,
                test_size=args.test_size,
                precision_floor=args.precision_floor,
                balance_test=args.balance_test,
                test_pos_cap=args.test_pos_cap,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
                text_max_features=args.text_max_features,
                model_name=args.model,
                n_jobs=args.n_jobs,
                train_year_start=args.train_year_start,
                train_year_end=args.train_year_end,
                test_year_start=args.test_year_start,
                test_year_end=args.test_year_end,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        if args.command == "prepare-bert":
            summary = prepare_bert_dataset(
                input_csv=Path(args.input),
                raw_output_csv=Path(args.raw_output),
                processed_output_csv=Path(args.processed_output),
                chunksize=args.chunksize,
                max_rows=args.max_rows,
                max_chunks=args.max_chunks,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0

        if args.command == "bert-eval":
            payload = evaluate_bert_embedding_local(
                input_csv=Path(args.input),
                output_dir=Path(args.output_dir),
                embedding_model_path=Path(args.embedding_model_path),
                sample_size=args.sample_size,
                test_size=args.test_size,
                precision_floor=args.precision_floor,
                balance_test=args.balance_test,
                test_pos_cap=args.test_pos_cap,
                random_state=args.random_state,
                device=args.device,
                max_length=args.max_length,
                train_year_start=args.train_year_start,
                train_year_end=args.train_year_end,
                test_year_start=args.test_year_start,
                test_year_end=args.test_year_end,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if args.command == "bert-finetune":
            payload = train_and_evaluate_bert_finetune(
                input_csv=Path(args.input),
                output_dir=Path(args.output_dir),
                model_path=Path(args.model_path),
                sample_size=args.sample_size,
                test_size=args.test_size,
                precision_floor=args.precision_floor,
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
                train_year_start=args.train_year_start,
                train_year_end=args.train_year_end,
                test_year_start=args.test_year_start,
                test_year_end=args.test_year_end,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if args.command == "prepare-llm":
            summary = prepare_llm_dataset(
                input_csv=Path(args.input),
                raw_output_csv=Path(args.raw_output),
                processed_output_csv=Path(args.processed_output),
                chunksize=args.chunksize,
                max_rows=args.max_rows,
                max_chunks=args.max_chunks,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0

        if args.command == "llm-eval":
            payload = evaluate_llm_remote(
                input_csv=Path(args.input),
                output_csv=Path(args.output),
                sample_size=args.sample_size,
                timeout=args.timeout,
                temperature=args.temperature,
                sleep_ms=args.sleep_ms,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if args.command == "local-llm-eval":
            payload = evaluate_llm_local(
                input_csv=Path(args.input),
                output_csv=Path(args.output),
                model_path=Path(args.model_path),
                sample_size=args.sample_size,
                eval_pos_cap=args.eval_pos_cap,
                eval_neg_cap=args.eval_neg_cap,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device_map=args.device_map,
                enable_thinking=args.enable_thinking,
                sleep_ms=args.sleep_ms,
                random_state=args.random_state,
                adapter_path=Path(args.adapter_path) if args.adapter_path else None,
                train_year_start=args.train_year_start,
                train_year_end=args.train_year_end,
                test_year_start=args.test_year_start,
                test_year_end=args.test_year_end,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if args.command == "llm-generate-teacher-sft":
            teacher_model_path = (
                Path(args.teacher_model_path)
                if args.teacher_model_path
                else Path(DEFAULT_QWEN_06B_PATH if args.teacher_model_size == "0.6b" else DEFAULT_QWEN_17B_PATH)
            )
            payload = generate_teacher_reason_sft(
                input_csv=Path(args.input),
                output_jsonl=Path(args.output_jsonl),
                audit_csv=Path(args.audit_csv),
                teacher_model_path=teacher_model_path,
                sample_size=args.sample_size,
                train_pos_cap=args.train_pos_cap,
                batch_size=args.batch_size,
                random_state=args.random_state,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device_map=args.device_map,
                enable_thinking=args.enable_thinking,
                train_year_start=args.train_year_start,
                train_year_end=args.train_year_end,
                test_year_start=args.test_year_start,
                test_year_end=args.test_year_end,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        if args.command == "llm-write-lf-config":
            payload = write_llamafactory_qlora_assets(
                sft_jsonl=Path(args.sft_jsonl),
                dataset_dir=Path(args.dataset_dir),
                dataset_name=args.dataset_name,
                student_model_path=Path(args.student_model_path),
                output_yaml=Path(args.output_yaml),
                output_cmd=Path(args.output_cmd),
                output_dir=Path(args.output_dir),
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                cutoff_len=args.cutoff_len,
                max_samples=args.max_samples,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
