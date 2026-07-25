"""Shared AudioMC helpers: dataset IO, message building, judge target parsing."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import soundfile as sf


MAX_TURNS = 8


def parse_judge_target(spec: str, default_port: int = 8000) -> Tuple[str, str, int]:
    """Parse `model_name+ip` or `model_name+ip:port` into (model, host, port)."""
    if "+" not in spec:
        raise ValueError(
            f"Expected judge as 'model_name+ip' (optional :port), got: {spec!r}"
        )
    model, hostport = spec.rsplit("+", 1)
    model = model.strip()
    hostport = hostport.strip()
    if not model or not hostport:
        raise ValueError(f"Invalid judge target: {spec!r}")
    if ":" in hostport:
        host, port_s = hostport.rsplit(":", 1)
        port = int(port_s)
    else:
        host, port = hostport, default_port
    return model, host, port


def parse_model_target(spec: str, default_port: int = 8091) -> Tuple[str, str, int]:
    """Same format as judge: `model_name+ip[:port]` for the model under eval."""
    return parse_judge_target(spec, default_port=default_port)


def audio_to_data_url(path: Path, mime: str = "audio/wav") -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def save_audio_array(audio_obj: Any, out_path: Path, target_sr: int = 16000) -> Path:
    """Save HF datasets Audio object (dict with array/sampling_rate) to wav."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_obj is None:
        raise ValueError("audio_obj is None")
    if isinstance(audio_obj, dict):
        arr = audio_obj["array"]
        sr = int(audio_obj["sampling_rate"])
    else:
        raise TypeError(f"Unsupported audio type: {type(audio_obj)}")
    # Downmix / write; soundfile handles float arrays in [-1, 1]
    sf.write(str(out_path), arr, sr)
    return out_path


def get_last_user_turn(row: Dict[str, Any]) -> int:
    last = 0
    for t in range(1, MAX_TURNS + 1):
        key = f"user_turn_{t}_transcript"
        val = row.get(key)
        if val is not None and str(val).strip():
            last = t
    return last


def parse_rubrics(rubric_field: Any) -> List[str]:
    """Rubric column may be a JSON list string or already a list."""
    if rubric_field is None:
        return []
    if isinstance(rubric_field, list):
        return [str(x).strip() for x in rubric_field if str(x).strip()]
    text = str(rubric_field).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: newline / semicolon separated
    parts = re.split(r"[\n;]+", text)
    return [p.strip() for p in parts if p.strip()]


def build_fixed_context_messages(
    row: Dict[str, Any],
    audio_paths: Dict[int, Path],
    *,
    use_data_urls: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fixed-context protocol (AudioMC paper):
    - Prior turns: user = audio, assistant = reference transcript text
    - Final turn: user = audio only (model must respond)
    No system prompt (matches Qwen eval guidance + AudioMC setup).
    """
    last = get_last_user_turn(row)
    messages: List[Dict[str, Any]] = []
    for t in range(1, last + 1):
        audio_path = audio_paths.get(t)
        if audio_path is None or not Path(audio_path).exists():
            raise FileNotFoundError(f"Missing audio for turn {t}: {audio_path}")
        if use_data_urls:
            url = audio_to_data_url(Path(audio_path))
        else:
            url = f"file://{Path(audio_path).resolve()}"
        user_content = [{"type": "audio_url", "audio_url": {"url": url}}]
        messages.append({"role": "user", "content": user_content})
        if t < last:
            asst = row.get(f"assistant_turn_{t}_transcript") or ""
            messages.append({"role": "assistant", "content": str(asst).strip()})
    return messages


def build_transcript_context_messages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Text baseline: same fixed-context protocol, but user turns are transcripts only
    (no audio). Assistant history remains reference transcript text.
    """
    last = get_last_user_turn(row)
    messages: List[Dict[str, Any]] = []
    for t in range(1, last + 1):
        user_text = str(row.get(f"user_turn_{t}_transcript") or "").strip()
        messages.append({"role": "user", "content": user_text})
        if t < last:
            asst = row.get(f"assistant_turn_{t}_transcript") or ""
            messages.append({"role": "assistant", "content": str(asst).strip()})
    return messages


def build_grading_conversation_history(
    row: Dict[str, Any], model_response: str
) -> str:
    """Dataset-card helper (lowercase column names)."""
    last_user_turn = get_last_user_turn(row)
    history_parts: List[str] = []
    for turn_num in range(1, last_user_turn + 1):
        user_text = str(row.get(f"user_turn_{turn_num}_transcript") or "").strip()
        if user_text:
            history_parts.append(f"User: {user_text}")
        if turn_num < last_user_turn:
            asst = str(row.get(f"assistant_turn_{turn_num}_transcript") or "").strip()
            if asst:
                history_parts.append(f"Assistant: {asst}")
        else:
            history_parts.append(f"Assistant: {model_response}")
    return "\n\n".join(history_parts)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse judge JSON, tolerating markdown fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def average_pass_rate(task_results: Iterable[Dict[str, Any]]) -> float:
    """APR: fraction of tasks where all rubrics pass."""
    tasks = list(task_results)
    if not tasks:
        return 0.0
    passes = 0
    for t in tasks:
        rubrics = t.get("rubric_results") or []
        if rubrics and all(bool(r.get("criteria_met")) for r in rubrics):
            passes += 1
    return passes / len(tasks)


def average_rubric_score(task_results: Iterable[Dict[str, Any]]) -> float:
    """ARS: mean per-task fraction of rubrics satisfied."""
    tasks = list(task_results)
    if not tasks:
        return 0.0
    scores = []
    for t in tasks:
        rubrics = t.get("rubric_results") or []
        if not rubrics:
            scores.append(0.0)
            continue
        scores.append(sum(1 for r in rubrics if r.get("criteria_met")) / len(rubrics))
    return sum(scores) / len(scores)


def axis_breakdown(task_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_axis: Dict[str, List[Dict[str, Any]]] = {}
    for t in task_results:
        by_axis.setdefault(t.get("axis", "UNKNOWN"), []).append(t)
    out = {}
    for axis, items in by_axis.items():
        out[axis] = {
            "n": len(items),
            "APR": average_pass_rate(items),
            "ARS": average_rubric_score(items),
        }
    return out
