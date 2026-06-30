from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from eduapp_metrics.llm_judge import flatten_judge_result, judge_record_with_openai
from eduapp_metrics.metrics import compute_metrics_for_dataframe, row_to_record
from eduapp_metrics.schemas import MetricConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich EduApp post-editing CSV with SOTA metrics.")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output enriched CSV path")
    p.add_argument("--use_sacrebleu", action="store_true")
    p.add_argument("--use_bertscore", action="store_true")
    p.add_argument("--use_comet", action="store_true")
    p.add_argument("--use_qe", action="store_true")
    p.add_argument("--use_embeddings", action="store_true")
    p.add_argument("--use_llm_judge", action="store_true")
    p.add_argument("--llm_model", default="gpt-5.5")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    cfg = MetricConfig(
        use_sacrebleu=args.use_sacrebleu,
        use_bertscore=args.use_bertscore,
        use_comet=args.use_comet,
        use_qe=args.use_qe,
        use_embeddings=args.use_embeddings,
        use_llm_judge=False,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    enriched = compute_metrics_for_dataframe(df, cfg)

    if args.use_llm_judge:
        rows = []
        for _, row in tqdm(enriched.iterrows(), total=len(enriched), desc="LLM judging"):
            record = row_to_record(row)
            try:
                result = judge_record_with_openai(record, model=args.llm_model)
                rows.append(flatten_judge_result(result))
            except Exception as exc:
                rows.append({"llm_error": str(exc)})
        enriched = pd.concat([enriched.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)

    warnings = enriched.attrs.get("warnings", [])
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
