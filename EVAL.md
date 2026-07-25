# AudioMC evaluation guide

This repo runs end-to-end evaluation on [ScaleAI/audiomc](https://huggingface.co/datasets/ScaleAI/audiomc): a multi-turn spoken-dialogue benchmark with rubric-based grading. The pipeline downloads audio, runs a speech model against each example, grades responses with an LLM judge, and reports **APR** / **ARS** scores.

## Overview

```
ScaleAI/audiomc (test, 452 examples)
        │
        ▼
  prepare_data.py  ──►  data/metadata.jsonl + data/audio/<id>/*.wav
        │
        ▼
  infer_text.py    ──►  predictions.jsonl   (model under test)
        │
        ▼
  judge.py         ──►  judged.jsonl + all.json   (LLM-as-a-Judge)
        │
        ▼
  inspect_runs.ipynb / score.py   (browse or recompute metrics)
```

Each example is a multi-turn conversation. The model must produce the **final assistant reply** only; prior turns are seeded from the dataset. Grading checks whether that final reply satisfies every rubric item for the example.

### Challenge axes

| Axis | What it tests |
| --- | --- |
| `INFERENCE_MEMORY` | Reasoning over earlier turns (tone, intent, facts) |
| `INSTRUCTION_RETENTION` | Following standing instructions across the conversation |
| `SELF_COHERENCE` | Consistency with the model's own prior replies |
| `VOICE_EDITING` | Responding to spoken corrections or edits from the user |

## Prerequisites

### Hardware / servers

Evaluation assumes two vLLM OpenAI-compatible servers:

1. **Model server** — the speech model under test (e.g. Qwen3-Omni Instruct or Thinking).
2. **Judge server** — a text LLM for rubric grading (default: `google/gemma-4-26B-A4B-it`).

Deploy scripts live under `deploy/` and are meant to be submitted on a Slurm cluster:

```bash
sbatch deploy/qwen3_omni_serve.sh          # Instruct, port 8091, 2× GPU
sbatch deploy/qwen3_omni_thinking_serve.sh # Thinking variant, 4× GPU
sbatch deploy/judge_gemma4_serve.sh        # Judge, port 8000, 2× GPU
```

Each script prints the node IP and the target string to use in eval commands, e.g.:

```
MODEL_TARGET=Qwen/Qwen3-Omni-30B-A3B-Instruct+192.168.1.49
JUDGE=google/gemma-4-26B-A4B-it+192.168.1.50:8000
```

### Python environment

The orchestration script uses conda env `slm`. Install eval dependencies (at minimum: `datasets`, `soundfile`, `openai`, `tqdm`, `numpy`).

Set `PYTHONPATH` so imports resolve to the `eval/` package:

```bash
export PYTHONPATH="$(pwd)/eval${PYTHONPATH:+:$PYTHONPATH}"
```

## Quick start

Edit `MODEL_TARGET` and `JUDGE` in `run_eval.sh` to match your server IPs, then:

```bash
./run_eval.sh              # full 452-example run
./run_eval.sh --limit 5    # smoke test on 5 examples
```

This creates a timestamped output directory under `outputs/<YYYYMMDD_HHMMSS>/` and logs under `logs/eval/`.

## Step 1 — Prepare data

`eval/prepare_data.py` downloads the `test` split from Hugging Face and writes:

- `data/metadata.jsonl` — one JSON object per example (transcripts, rubrics, audio paths)
- `data/audio/<example_id>/user_turn_<n>.wav` — per-turn user audio

```bash
python eval/prepare_data.py --out-dir data
python eval/prepare_data.py --out-dir data --limit 10   # subset
```

`run_eval.sh` skips this step if `data/metadata.jsonl` already exists.

A pre-built `metadata.jsonl` is also checked into the repo root for inspection without re-downloading audio.

## Step 2 — Inference

`eval/infer_text.py` calls the model server with the **fixed-context protocol** (see below). It requests text output only (`modalities: ["text"]`).

```bash
python eval/infer_text.py \
  --metadata data/metadata.jsonl \
  --model-target "Qwen/Qwen3-Omni-30B-A3B-Instruct+192.168.1.49" \
  --out outputs/run/predictions.jsonl
```

### Target format

Both model and judge targets use `model_name+host[:port]`:

| Target | Default port |
| --- | --- |
| Model (`--model-target`) | 8091 |
| Judge (`--judge`) | 8000 |

### User input modality

| Flag | Behavior |
| --- | --- |
| `--user-modality audio` (default) | User turns sent as `audio_url` (base64 wav) |
| `--user-modality text` | User turns sent as transcripts only (text baseline) |

### Resume

If `--out` already exists, completed example IDs are skipped automatically. Safe to restart after failures.

### Output: `predictions.jsonl`

One line per example:

```json
{
  "id": "3tc3nw47nhr9deko",
  "axis": "INFERENCE_MEMORY",
  "last_user_turn": 4,
  "model_target": "Qwen/Qwen3-Omni-30B-A3B-Instruct+192.168.1.49",
  "user_modality": "audio",
  "model_response": "...",
  "latency_s": 4.14,
  "error": null
}
```

## Step 3 — Judge

`eval/judge.py` grades each prediction against every rubric item using the prompt in `eval/prompts.py` (from the official AudioMC dataset card). The judge sees the full conversation history as **text transcripts**, with the model's generated reply substituted for the final assistant turn.

```bash
python eval/judge.py \
  --metadata data/metadata.jsonl \
  --predictions outputs/run/predictions.jsonl \
  --judge "google/gemma-4-26B-A4B-it+192.168.1.50:8000" \
  --out outputs/run/judged.jsonl \
  --summary outputs/run/all.json
```

For each rubric item the judge returns `criteria_met` (bool) and `explanation` (string). An example **passes** only if **all** rubrics are met.

Judging also supports resume (skips IDs already in `--out`).

### Output: `judged.jsonl`

```json
{
  "id": "3tc3nw47nhr9deko",
  "axis": "INFERENCE_MEMORY",
  "model_response": "...",
  "rubric_results": [
    {
      "rubric_item": "Discerns that the sarcastic tone ...",
      "criteria_met": false,
      "explanation": "...",
      "raw": "..."
    }
  ],
  "pass": false,
  "judge": "google/gemma-4-26B-A4B-it+192.168.1.50:8000",
  "pred_error": null
}
```

### Output: `all.json` (summary)

```json
{
  "n": 452,
  "APR": 0.259,
  "ARS": 0.528,
  "by_axis": {
    "INFERENCE_MEMORY": { "n": 132, "APR": 0.242, "ARS": 0.344 },
    "INSTRUCTION_RETENTION": { "n": 120, "APR": 0.333, "ARS": 0.567 },
    "SELF_COHERENCE": { "n": 83, "APR": 0.277, "ARS": 0.536 },
    "VOICE_EDITING": { "n": 117, "APR": 0.188, "ARS": 0.692 }
  },
  "judge": "...",
  "predictions": "..."
}
```

## Metrics

| Metric | Name | Definition |
| --- | --- | --- |
| **APR** | Average Pass Rate | Fraction of examples where **every** rubric passes |
| **ARS** | Average Rubric Score | Mean, per example, of (rubrics passed ÷ total rubrics) |

Recompute from an existing `judged.jsonl` without re-running the judge:

```bash
python eval/score.py --judged outputs/run/judged.jsonl
python eval/score.py --judged outputs/run/judged.jsonl --out outputs/run/all.json
```

## Fixed-context protocol

This matches the AudioMC paper and Qwen3-Omni eval guidance:

1. **Prior turns** — user input is audio (or transcript in the text baseline); assistant side is the **reference transcript** from the dataset (not model-generated).
2. **Final turn** — user audio (or transcript) only; the model must generate the assistant reply.
3. **No system prompt.**

Message construction is in `eval/utils.py` (`build_fixed_context_messages`, `build_transcript_context_messages`).

For grading, `build_grading_conversation_history` assembles the same conversation as plain text, swapping the model's final reply in place of a reference answer.

## Browsing results

Pre-computed runs for Qwen3-Omni Instruct and Thinker live in:

```
Qwen3-Omni-Instruct/   # judged.jsonl, predictions.jsonl, all.json
Qwen3-Omni-Thinker/
```

Open `inspect_runs.ipynb` to filter by axis, pass/fail, or search text. The notebook auto-discovers run directories that contain `judged.jsonl` and `predictions.jsonl`.

Optional: `eval/summarize_inspection.py` precomputes short LLM summaries per example for faster notebook browsing (requires a running judge or other OpenAI-compatible server).

## Directory layout

```
.
├── run_eval.sh                 # end-to-end orchestration
├── EVAL.md                     # this file
├── metadata.jsonl              # bundled metadata (no audio required for inspection)
├── eval/
│   ├── prepare_data.py         # HF download + wav export
│   ├── infer_text.py           # model inference
│   ├── judge.py                # LLM-as-a-Judge grading
│   ├── score.py                # recompute APR/ARS
│   ├── prompts.py              # judge prompt template
│   ├── utils.py                # message building, metrics, parsing
│   ├── inspect_prompts.py      # prompts for inspection summaries
│   └── summarize_inspection.py # optional summary precompute
├── deploy/
│   ├── qwen3_omni_serve.sh
│   ├── qwen3_omni_thinking_serve.sh
│   └── judge_gemma4_serve.sh
├── data/                       # created by prepare_data.py
│   ├── metadata.jsonl
│   └── audio/<id>/user_turn_*.wav
├── outputs/<timestamp>/        # created by run_eval.sh
│   ├── predictions.jsonl
│   ├── judged.jsonl
│   └── all.json
└── logs/eval/                  # run logs
```

## Tips

- **Smoke test first** — `./run_eval.sh --limit 5` before a full 452-example run.
- **Update IPs** — after redeploying servers, edit `MODEL_TARGET` / `JUDGE` in `run_eval.sh` (or pass them directly to the Python scripts).
- **Compare modalities** — run inference twice with `--user-modality audio` vs `text` to measure how much performance comes from hearing audio vs reading transcripts.
- **Judge temperature** — `judge.py` defaults to `temperature=1.0` (matching common AudioMC setups). Lower it for more deterministic grading.
- **Parse errors** — if the judge returns malformed JSON, the rubric is scored as failed with `JUDGE_PARSE_ERROR` in the explanation.
