#!/usr/bin/env python3
"""Precompute LLM summaries for AudioMC inspection notebook."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from inspect_prompts import SUMMARIZE_SYSTEM
from utils import MAX_TURNS, extract_json_object, get_last_user_turn, parse_judge_target


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_turn_pairs(meta: Dict[str, Any], model_response: str) -> List[Dict[str, Any]]:
    last = get_last_user_turn(meta)
    pairs: List[Dict[str, Any]] = []
    for t in range(1, last + 1):
        user = str(meta.get(f"user_turn_{t}_transcript") or "").strip()
        if t < last:
            assistant = str(meta.get(f"assistant_turn_{t}_transcript") or "").strip()
            is_final = False
        else:
            assistant = str(model_response or "").strip()
            is_final = True
        pairs.append(
            {
                "turn": t,
                "is_final": is_final,
                "user_transcript": user,
                "assistant_text": assistant,
            }
        )
    return pairs


def format_conversation_for_prompt(
    meta: Dict[str, Any],
    model_response: str,
    rubrics: Optional[List[str]] = None,
) -> str:
    lines = [
        f"Example ID: {meta.get('id')}",
        f"Axis: {meta.get('axis')}",
        "",
        "Conversation:",
    ]
    for pair in build_turn_pairs(meta, model_response):
        t = pair["turn"]
        lines.append(f"--- Turn {t} (user) ---")
        lines.append(pair["user_transcript"] or "(empty)")
        if pair["is_final"]:
            lines.append(f"--- Turn {t} (assistant, MODEL GENERATED) ---")
            lines.append(pair["assistant_text"] or "(empty)")
        else:
            lines.append(f"--- Turn {t} (assistant, REFERENCE) ---")
            lines.append(pair["assistant_text"] or "(empty)")

    if rubrics:
        lines.extend(["", "Final-turn rubrics (use for gt on the last turn):"])
        for i, rubric in enumerate(rubrics, start=1):
            lines.append(f"{i}. {rubric}")
    return "\n".join(lines)


def summarize_example(
    client: OpenAI,
    model: str,
    meta: Dict[str, Any],
    model_response: str,
    rubrics: Optional[List[str]] = None,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    user_prompt = format_conversation_for_prompt(meta, model_response, rubrics=rubrics)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content or ""
    try:
        parsed = extract_json_object(raw)
        summary = {
            "context": str(parsed.get("context", "")).strip(),
            "question": str(parsed.get("question", "")).strip(),
            "answer": str(parsed.get("answer", "")).strip(),
            "gt": str(parsed.get("gt", "")).strip(),
        }
        return {"summary": summary, "raw": raw, "error": None}
    except Exception as e:
        return {"summary": None, "raw": raw, "error": f"PARSE_ERROR: {e}"}


def summaries_path_for_run(run_dir: Path) -> Path:
    return run_dir / "summaries.jsonl"


def load_summaries_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in load_jsonl(path)}


def extract_summary(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    summary = record.get("summary")
    if isinstance(summary, dict) and summary:
        return {
            "context": str(summary.get("context", "")).strip(),
            "question": str(summary.get("question", "")).strip(),
            "answer": str(summary.get("answer", "")).strip(),
            "gt": str(summary.get("gt", "")).strip(),
        }
    # Legacy: turn_summaries — use the last turn as overall summary.
    turns = record.get("turn_summaries") or []
    if turns:
        last = max(turns, key=lambda t: int(t.get("turn", 0)))
        return {
            "context": str(last.get("context", "")).strip(),
            "question": str(last.get("question", "")).strip(),
            "answer": str(last.get("answer", "")).strip(),
            "gt": str(last.get("gt", "")).strip(),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute inspection summaries with Gemma")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing judged.jsonl and predictions.jsonl",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "metadata.jsonl",
    )
    parser.add_argument(
        "--summarizer",
        default="google/gemma-4-26B-A4B-it+192.168.1.45:8000",
        help="Summarizer as model_name+ip[:port]",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--force", action="store_true", help="Re-summarize all examples")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    judged_path = run_dir / "judged.jsonl"
    preds_path = run_dir / "predictions.jsonl"
    out_path = summaries_path_for_run(run_dir)

    if not judged_path.exists():
        raise FileNotFoundError(f"Missing {judged_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Missing {args.metadata}")

    model, host, port = parse_judge_target(args.summarizer)
    base_url = f"http://{host}:{port}/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key, timeout=args.timeout)

    meta_by_id = {row["id"]: row for row in load_jsonl(args.metadata)}
    judged = load_jsonl(judged_path)
    preds_by_id = {row["id"]: row for row in load_jsonl(preds_path)}

    done_ids = set()
    if out_path.exists() and not args.force:
        done_ids = {row["id"] for row in load_jsonl(out_path)}

    rows = judged[: args.limit] if args.limit else judged
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.force else "a"
    if args.force and out_path.exists():
        out_path.unlink()

    # Resume: append only examples not already summarized.
    written = 0
    with out_path.open(mode, encoding="utf-8") as fout:
        for judged_row in tqdm(rows, desc="summarize"):
            ex_id = judged_row["id"]
            if ex_id in done_ids:
                continue
            meta = meta_by_id.get(ex_id)
            if meta is None:
                continue

            pred = preds_by_id.get(ex_id, judged_row)
            model_response = judged_row.get("model_response") or pred.get("model_response") or ""
            rubrics = meta.get("rubrics") or judged_row.get("rubrics") or []

            t0 = time.time()
            try:
                result = summarize_example(
                    client,
                    model,
                    meta,
                    model_response,
                    rubrics=rubrics,
                    temperature=args.temperature,
                )
                err = result.get("error")
            except Exception as e:
                result = {"summary": None, "raw": "", "error": str(e)}
                err = str(e)

            record = {
                "id": ex_id,
                "axis": meta.get("axis"),
                "last_user_turn": meta.get("last_user_turn"),
                "summarizer": args.summarizer,
                "summary": result.get("summary"),
                "error": err,
                "latency_s": time.time() - t0,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

    print(f"Wrote/updated {written} summaries -> {out_path}")


if __name__ == "__main__":
    main()
