#!/usr/bin/env python3
"""
AudioMC rubric grading via LLM-as-a-Judge on a vLLM-hosted API.

Judge target format: model_name+ip[:port]
  e.g. google/gemma-4-26B-A4B-it+192.168.1.20
  e.g. openai/gpt-oss-20b+192.168.1.20:8000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm

from prompts import JUDGE_SYSTEM_TEMPLATE
from utils import (
    average_pass_rate,
    average_rubric_score,
    axis_breakdown,
    build_grading_conversation_history,
    extract_json_object,
    parse_judge_target,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["id"]: r for r in rows}


def grade_one(
    client: OpenAI,
    judge_model: str,
    history: str,
    rubric_item: str,
    temperature: float,
) -> Dict[str, Any]:
    prompt = JUDGE_SYSTEM_TEMPLATE.format(
        conversation_history=history,
        rubric_item=rubric_item,
    )
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024,
    )
    raw = resp.choices[0].message.content or ""
    try:
        parsed = extract_json_object(raw)
        criteria_met = bool(parsed.get("criteria_met"))
        explanation = str(parsed.get("explanation", ""))
    except Exception as e:
        criteria_met = False
        explanation = f"JUDGE_PARSE_ERROR: {e}; raw={raw[:500]}"
    return {
        "rubric_item": rubric_item,
        "criteria_met": criteria_met,
        "explanation": explanation,
        "raw": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "metadata.jsonl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL from infer_text.py",
    )
    parser.add_argument(
        "--judge",
        required=True,
        help="model_name+ip[:port] for the vLLM judge",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "judged.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "summary.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--judge-port-default", type=int, default=8000)
    args = parser.parse_args()

    judge_model, host, port = parse_judge_target(
        args.judge, default_port=args.judge_port_default
    )
    base_url = f"http://{host}:{port}/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key, timeout=args.timeout)

    meta = index_by_id(load_jsonl(args.metadata))
    preds = load_jsonl(args.predictions)
    if args.limit is not None:
        preds = preds[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if args.out.exists():
        with args.out.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    print(f"Judge: {judge_model} @ {base_url}")
    print(f"Predictions: {len(preds)} (resume skip={len(done_ids)})")

    results: List[Dict[str, Any]] = []
    # Keep previously judged for summary if resuming
    if args.out.exists():
        results.extend(load_jsonl(args.out))

    with args.out.open("a", encoding="utf-8") as fout:
        for pred in tqdm(preds):
            if pred["id"] in done_ids:
                continue
            row = meta.get(pred["id"])
            if row is None:
                print(f"WARN: missing metadata for {pred['id']}")
                continue
            response = pred.get("model_response") or ""
            history = build_grading_conversation_history(row, response)
            rubrics = row.get("rubrics") or []
            rubric_results = []
            for item in rubrics:
                try:
                    rubric_results.append(
                        grade_one(
                            client,
                            judge_model,
                            history,
                            item,
                            temperature=args.temperature,
                        )
                    )
                except Exception as e:
                    rubric_results.append(
                        {
                            "rubric_item": item,
                            "criteria_met": False,
                            "explanation": f"JUDGE_CALL_ERROR: {e}",
                            "raw": "",
                        }
                    )
                time.sleep(0.05)

            all_pass = bool(rubric_results) and all(
                r["criteria_met"] for r in rubric_results
            )
            out = {
                "id": pred["id"],
                "axis": row.get("axis") or pred.get("axis"),
                "model_response": response,
                "rubric_results": rubric_results,
                "pass": all_pass,
                "judge": args.judge,
                "pred_error": pred.get("error"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(out)

    summary = {
        "n": len(results),
        "APR": average_pass_rate(results),
        "ARS": average_rubric_score(results),
        "by_axis": axis_breakdown(results),
        "judge": args.judge,
        "predictions": str(args.predictions),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote judged: {args.out}")
    print(f"Wrote summary: {args.summary}")


if __name__ == "__main__":
    main()
