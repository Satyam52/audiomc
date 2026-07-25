#!/usr/bin/env python3
"""Download ScaleAI/audiomc and materialize per-turn wav files for offline eval."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

from utils import MAX_TURNS, get_last_user_turn, parse_rubrics, save_audio_array


def materialize_audio(audio_obj, out_path: Path) -> Path:
    """Write HF Audio (decoded dict or raw bytes/path) to wav without torchcodec."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_obj is None:
        raise ValueError("audio_obj is None")
    if isinstance(audio_obj, dict):
        if "array" in audio_obj:
            return save_audio_array(audio_obj, out_path)
        # decode=False: {"path": ..., "bytes": ...}
        raw = audio_obj.get("bytes")
        if raw is not None:
            arr, sr = sf.read(io.BytesIO(raw))
            sf.write(str(out_path), np.asarray(arr), int(sr))
            return out_path
        path = audio_obj.get("path")
        if path:
            arr, sr = sf.read(path)
            sf.write(str(out_path), np.asarray(arr), int(sr))
            return out_path
    raise TypeError(
        f"Unsupported audio type: {type(audio_obj)} "
        f"keys={getattr(audio_obj, 'keys', lambda: None)()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    audio_root = out_dir / "audio"
    meta_path = out_dir / "metadata.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading ScaleAI/audiomc ({args.split}) ...", flush=True)
    ds = load_dataset("ScaleAI/audiomc", split=args.split)
    # Prefer bytes path via soundfile (avoids torchcodec dependency).
    audio_cols = [f"user_turn_{t}_audio" for t in range(1, MAX_TURNS + 1)]
    for col in audio_cols:
        if col in ds.column_names:
            ds = ds.cast_column(col, Audio(sampling_rate=None, decode=False))

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"Writing {n} examples -> {out_dir}", flush=True)

    with meta_path.open("w", encoding="utf-8") as fout:
        for i in tqdm(range(n)):
            row = ds[i]
            ex_id = row["id"]
            last = get_last_user_turn(row)
            audio_paths = {}
            for t in range(1, last + 1):
                audio_obj = row.get(f"user_turn_{t}_audio")
                if audio_obj is None:
                    continue
                wav_path = audio_root / ex_id / f"user_turn_{t}.wav"
                materialize_audio(audio_obj, wav_path)
                audio_paths[str(t)] = str(wav_path.resolve())

            record = {
                "id": ex_id,
                "axis": row.get("axis"),
                "last_user_turn": last,
                "rubrics": parse_rubrics(row.get("rubric")),
                "audio_paths": audio_paths,
            }
            for t in range(1, MAX_TURNS + 1):
                ut = row.get(f"user_turn_{t}_transcript")
                at = row.get(f"assistant_turn_{t}_transcript")
                if ut is not None:
                    record[f"user_turn_{t}_transcript"] = ut
                if at is not None:
                    record[f"assistant_turn_{t}_transcript"] = at
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote metadata: {meta_path}", flush=True)
    print(f"Audio root: {audio_root}", flush=True)


if __name__ == "__main__":
    main()
