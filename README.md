# AudioMC — Qwen3-Omni results

Qwen3-Omni **Instruct** vs **Thinker** scores on [Audio MultiChallenge](https://huggingface.co/datasets/ScaleAI/audiomc), plus an interactive notebook to browse individual examples.

## Results

| **Metric** | Thinker | Instruct | **Δ** |
| --- | --- | --- | --- |
| **n** | 452 | 452 | — |
| **APR** | 36.5% | 25.9% | 10.6pp |
| **ARS** | 58.7% | 52.8% | 5.9pp |

| **Axis** | **n** | **Thinker (APR / ARS)** | **Instruct (APR / ARS)** | **Δ (APR / ARS)** |
| --- | --- | --- | --- | --- |
| Inference Memory | 132 | 37.9% / 47.2% | 24.2% / 34.4% | 13.7pp / 12.9pp |
| Instruction Retention | 120 | 42.5% / 60.5% | 33.3% / 56.7% | 9.2pp / 4.8pp |
| Self Coherence | 83 | 33.7% / 57.4% | 27.7% / 53.6% | 6.0pp / 3.8pp |
| Voice Editing | 117 | 30.8% / 70.8% | 18.8% / 69.2% | 12.0pp / 1.6pp |

## Repo layout

```
.
├── metadata.jsonl              # 452 benchmark examples (transcripts, rubrics, audio paths)
├── Qwen3-Omni-Instruct/        # Instruct run outputs
│   ├── judged.jsonl
│   ├── predictions.jsonl
│   ├── summaries.jsonl
│   └── all.json
├── Qwen3-Omni-Thinker/         # Thinker run outputs (same files)
├── inspect_runs.ipynb          # interactive browser
├── requirements.txt
└── audiomc/                    # optional — only needed for audio playback
    └── data/audio/<id>/user_turn_*.wav
```

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/Satyam52/audiomc.git
cd audiomc
```

### 2. Create a conda env and install deps

```bash
conda create -n audiomc-inspect python=3.11 -y
conda activate audiomc-inspect
pip install -r requirements.txt
```

No HuggingFace download is required — `metadata.jsonl` and both model runs are already bundled.

### 3. (Optional) Download audio from Hugging Face

Transcripts and judge results work without audio. To enable audio players in the notebook, export wav files from the official benchmark dataset:

**Dataset:** [ScaleAI/audiomc](https://huggingface.co/datasets/ScaleAI/audiomc) (`test` split, 452 examples, ~5 GB)

**Where to put files** (relative to repo root):

```
audiomc/data/audio/<example_id>/user_turn_1.wav
audiomc/data/audio/<example_id>/user_turn_2.wav
...
```

Example for id `3tc3nw47nhr9deko`:

```
audiomc/data/audio/3tc3nw47nhr9deko/user_turn_1.wav
audiomc/data/audio/3tc3nw47nhr9deko/user_turn_2.wav
```

**Export command** (run from repo root; first run downloads the dataset):

```bash
pip install datasets soundfile

python - <<'PY'
from pathlib import Path
import soundfile as sf
from datasets import load_dataset

out = Path("audiomc/data/audio")
ds = load_dataset("ScaleAI/audiomc", split="test")

for row in ds:
    ex_dir = out / row["id"]
    ex_dir.mkdir(parents=True, exist_ok=True)
    for turn in range(1, 9):
        audio = row.get(f"user_turn_{turn}_audio")
        if audio and audio.get("array") is not None:
            sf.write(ex_dir / f"user_turn_{turn}.wav", audio["array"], audio["sampling_rate"])

print(f"Wrote audio under {out.resolve()}")
PY
```

The notebook looks for audio only under `audiomc/data/audio/`. If a wav is missing, the player is skipped (transcripts still show).

### 4. Run the inspector

```bash
jupyter notebook inspect_runs.ipynb
```

Run all cells, then use the **Model** dropdown to switch between `Qwen3-Omni-Instruct` and `Qwen3-Omni-Thinker`. Filter by axis, pass/fail, or search text.
