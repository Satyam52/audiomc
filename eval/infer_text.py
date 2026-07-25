#!/usr/bin/env python3
"""
AudioMC text-output inference against a vLLM / vLLM-Omni OpenAI-compatible server.

Fixed-context protocol: seed prior turns (user audio or transcript + assistant text),
generate only the final assistant reply with modalities=["text"].

Target format: model_name+ip[:port]   e.g. Qwen/Qwen3-Omni-30B-A3B-Instruct+192.168.1.10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm

from utils import (
    build_fixed_context_messages,
    build_transcript_context_messages,
    get_last_user_turn,
    parse_model_target,
)


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "metadata.jsonl",
    )
    parser.add_argument(
        "--model-target",
        required=True,
        help="model_name+ip[:port] for the model under evaluation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "predictions.jsonl",
    )
    parser.add_argument(
        "--user-modality",
        choices=("audio", "text"),
        default="audio",
        help="User turn input: audio (default) or transcript-only text baseline",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="vLLM usually ignores API keys; default EMPTY",
    )
    args = parser.parse_args()

    model, host, port = parse_model_target(args.model_target, default_port=8091)
    base_url = f"http://{host}:{port}/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key, timeout=args.timeout)

    rows = load_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if args.out.exists():
        with args.out.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    print(f"Model target: {model} @ {base_url}")
    print(f"User modality: {args.user_modality}")
    print(f"Examples: {len(rows)} (resume skip={len(done_ids)})")

    with args.out.open("a", encoding="utf-8") as fout:
        for row in tqdm(rows):
            if row["id"] in done_ids:
                continue
            if args.user_modality == "text":
                messages = build_transcript_context_messages(row)
            else:
                audio_paths = {int(k): Path(v) for k, v in row["audio_paths"].items()}
                messages = build_fixed_context_messages(row, audio_paths)
            t0 = time.time()
            err = None
            response_text = ""
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    extra_body={"modalities": ["text"]},
                )
                # Omni servers may return multiple choices (text / audio);
                # take the first non-empty text content.
                for ch in resp.choices:
                    content = getattr(ch.message, "content", None)
                    if content:
                        response_text = content
                        break
                if not response_text and resp.choices:
                    response_text = resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
            elapsed = time.time() - t0
            out = {
                "id": row["id"],
                "axis": row.get("axis"),
                "last_user_turn": row.get("last_user_turn") or get_last_user_turn(row),
                "model_target": args.model_target,
                "user_modality": args.user_modality,
                "model_response": response_text,
                "latency_s": elapsed,
                "error": err,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote predictions: {args.out}")


if __name__ == "__main__":
    main()
