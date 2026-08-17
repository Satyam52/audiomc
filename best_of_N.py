#!/usr/bin/env python3
"""Best-of-N oracle selection from multiple judged eval runs.

Selects the best candidate per example using example-level pass/fail (APR),
not per-rubric score. Compares against the best single-run baseline (highest
APR among the candidate runs) and records examples where best-of-N improves.

Default runs are the three Qwen3-Omni temp-0 replicates:
  145045, 150548, 152216 under runs/eval/allqwen/.

Usage (conda env from env.sh, default: slm):
  source ./env.sh && python scripts/best_of_N.py
  source ./env.sh && python scripts/best_of_N.py --runs runs/eval/allqwen/20260814_145045_* ...

Outputs under runs/eval/allqwen/best_of_ids/<id1>_<id2>_<id3>/ (e.g. 145045_150548_152216):
  improved.json          examples that beat the best single baseline
  predictions_by_run/    per-run predictions with pass_vector + rubric failure counts
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

from audiomc.jsonl import read_jsonl, write_jsonl  # noqa: E402
from audiomc.metrics import average_pass_rate, average_rubric_score, build_summary  # noqa: E402

DEFAULT_RUNS = [
    ROOT / "runs/eval/allqwen/20260814_145045_local_Qwen3-Omni-30B-A3B-Thinking",
    ROOT / "runs/eval/allqwen/20260814_150548_local_Qwen3-Omni-30B-A3B-Thinking",
    ROOT / "runs/eval/allqwen/20260814_152216_local_Qwen3-Omni-30B-A3B-Thinking",
]
DEFAULT_OUT_ROOT = ROOT / "runs/eval/allqwen/best_of_ids"
DEFAULT_METADATA = ROOT / "data/metadata.jsonl"


def _n_met(row: Dict[str, Any]) -> int:
    rubrics = row.get("rubric_results") or []
    return sum(1 for r in rubrics if r.get("criteria_met"))


def _run_label(run_dir: Path) -> str:
    name = run_dir.name
    for token in name.split("_"):
        if token.isdigit() and len(token) == 6:
            return token
    return name


def _pass_vector_entry(
    judged: Dict[str, Any],
    pv_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pass vector for one example: pass, score, n_rubrics, n_failed_checks."""
    if pv_row is not None:
        return {
            "pass": bool(pv_row.get("pass")),
            "score": pv_row.get("score"),
            "n_rubrics": int(pv_row.get("n_rubrics") or 0),
            "n_failed_checks": int(pv_row.get("n_failed_checks") or 0),
        }

    rubrics = judged.get("rubric_results") or []
    n_rubrics = len(rubrics)
    n_met = _n_met(judged)
    n_failed = n_rubrics - n_met
    return {
        "pass": bool(judged.get("pass")),
        "score": (n_met / n_rubrics) if n_rubrics else 0.0,
        "n_rubrics": n_rubrics,
        "n_failed_checks": n_failed,
    }


def _enrich_prediction(
    pred: Dict[str, Any],
    judged: Dict[str, Any],
    pass_vec: Dict[str, Any],
    run_label: str,
) -> Dict[str, Any]:
    """Merge prediction row with pass_vector and rubric failure summary."""
    out = dict(pred)
    out["run"] = run_label
    out["pass_vector"] = pass_vec
    out["pass"] = pass_vec["pass"]
    out["n_rubrics"] = pass_vec["n_rubrics"]
    out["n_failed_checks"] = pass_vec["n_failed_checks"]
    out["n_met"] = pass_vec["n_rubrics"] - pass_vec["n_failed_checks"]
    out["rubric_summary"] = (
        f"{pass_vec['n_failed_checks']}/{pass_vec['n_rubrics']} failed"
        if pass_vec["n_rubrics"]
        else "0/0 failed"
    )
    if judged.get("rubric_results"):
        out["rubric_results"] = judged["rubric_results"]
    return out


def load_run(
    run_dir: Path,
) -> Tuple[str, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], float]:
    judged_path = run_dir / "judged.jsonl"
    preds_path = run_dir / "predictions.jsonl"
    pv_path = run_dir / "pass_vector.jsonl"
    if not judged_path.exists():
        raise FileNotFoundError(f"missing judged.jsonl: {judged_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"missing predictions.jsonl: {preds_path}")

    judged = {r["id"]: r for r in read_jsonl(judged_path)}
    preds = {r["id"]: r for r in read_jsonl(preds_path)}
    pass_vectors: Dict[str, Dict[str, Any]] = {}
    if pv_path.exists():
        pass_vectors = {r["id"]: r for r in read_jsonl(pv_path)}

    apr = average_pass_rate(judged.values())
    return _run_label(run_dir), judged, preds, pass_vectors, apr


def pick_best_candidate(
    candidates: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]],
) -> Tuple[str, Dict[str, Any], Dict[str, Any], int]:
    """Pick best candidate by example pass, then n_met rubrics (APR not ARS)."""
    passing = [(label, judged, pred) for label, judged, pred in candidates if judged.get("pass")]
    if passing:
        best = max(passing, key=lambda x: (_n_met(x[1]), x[0]))
        return best[0], best[1], best[2], 0

    best = max(candidates, key=lambda x: (_n_met(x[1]), x[0]))
    return best[0], best[1], best[2], 1


def prepare_out_dir(out_dir: Path) -> None:
    """Reset output dir before writing fresh results."""
    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("best_of_N")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def plot_results(
    out_dir: Path,
    *,
    run_labels: List[str],
    per_run_apr: Dict[str, float],
    baseline_label: str,
    baseline_apr: float,
    best_of_n_apr: float,
    per_axis: Dict[str, Dict[str, float]],
    category_counts: Dict[str, int],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # APR comparison: each run + baseline + best-of-N
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = run_labels + ["baseline", "best_of_N"]
    values = [per_run_apr[l] for l in run_labels] + [baseline_apr, best_of_n_apr]
    colors = ["#4C72B0"] * len(run_labels) + ["#DD8452", "#55A868"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("APR")
    ax.set_title("APR: individual runs vs best baseline vs best-of-N")
    ax.set_ylim(0, min(1.0, max(values) * 1.25 + 0.05))
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "apr_comparison.png", dpi=150)
    plt.close(fig)

    # Improvement categories
    fig, ax = plt.subplots(figsize=(6, 4))
    cat_labels = ["baseline_only", "both_pass", "best_of_n_only", "both_fail"]
    cat_values = [category_counts.get(k, 0) for k in cat_labels]
    cat_colors = ["#DD8452", "#8172B3", "#55A868", "#C44E52"]
    ax.bar(cat_labels, cat_values, color=cat_colors)
    ax.set_ylabel("Examples")
    ax.set_title("Pass membership: baseline vs best-of-N")
    for i, val in enumerate(cat_values):
        ax.text(i, val + 1, str(val), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "improvement_categories.png", dpi=150)
    plt.close(fig)

    # Per-axis APR
    if per_axis:
        axes_sorted = sorted(per_axis.keys())
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = range(len(axes_sorted))
        width = 0.35
        base_vals = [per_axis[a]["baseline_apr"] for a in axes_sorted]
        bon_vals = [per_axis[a]["best_of_n_apr"] for a in axes_sorted]
        ax.bar([i - width / 2 for i in x], base_vals, width, label="baseline", color="#DD8452")
        ax.bar([i + width / 2 for i in x], bon_vals, width, label="best_of_N", color="#55A868")
        ax.set_xticks(list(x))
        ax.set_xticklabels([a.replace("_", "\n") for a in axes_sorted], fontsize=8)
        ax.set_ylabel("APR")
        ax.set_title("Per-axis APR")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "per_axis_apr.png", dpi=150)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        default=DEFAULT_RUNS,
        help="Eval run directories (each needs judged.jsonl + predictions.jsonl)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/eval/allqwen/best_of_ids/<id1>_<id2>_...)",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Metadata jsonl for example context in improved records",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dirs = [Path(r).resolve() for r in args.runs]
    n_runs = len(run_dirs)

    loaded = [load_run(d) for d in run_dirs]
    run_labels = [x[0] for x in loaded]
    judged_by_run = [x[1] for x in loaded]
    preds_by_run = [x[2] for x in loaded]
    pass_vector_by_run = [x[3] for x in loaded]
    apr_by_run = {label: apr for label, _, _, _, apr in loaded}

    common_ids = set(judged_by_run[0].keys())
    for judged, preds in zip(judged_by_run[1:], preds_by_run[1:]):
        common_ids &= set(judged.keys())
        common_ids &= set(preds.keys())
    common_ids = sorted(common_ids)

    # Best single baseline = highest-APR run among candidates
    baseline_idx = max(range(n_runs), key=lambda i: apr_by_run[run_labels[i]])
    baseline_label = run_labels[baseline_idx]
    baseline_judged = judged_by_run[baseline_idx]
    baseline_apr = apr_by_run[baseline_label]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    label_slug = "_".join(run_labels)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (DEFAULT_OUT_ROOT / label_slug).resolve()
    prepare_out_dir(out_dir)
    logs_dir = out_dir / "logs"
    logger = setup_logging(logs_dir / "best_of_N.log")

    metadata: Dict[str, Dict[str, Any]] = {}
    if args.metadata.exists():
        metadata = {r["id"]: r for r in read_jsonl(args.metadata)}

    logger.info("Best-of-%d from %d runs, %d common examples", n_runs, n_runs, len(common_ids))
    for label in run_labels:
        logger.info("  run %s APR=%.4f", label, apr_by_run[label])
    logger.info("Baseline (best single): %s APR=%.4f", baseline_label, baseline_apr)

    per_example: List[Dict[str, Any]] = []
    best_of_n_judged_rows: List[Dict[str, Any]] = []
    best_of_n_preds_rows: List[Dict[str, Any]] = []
    predictions_by_run: Dict[str, List[Dict[str, Any]]] = {label: [] for label in run_labels}

    gained = lost = both_pass = baseline_only = both_fail = 0
    pick_counts = Counter()
    per_axis_stats: Dict[str, Dict[str, List[bool]]] = defaultdict(
        lambda: {"baseline": [], "best_of_n": []}
    )

    for exid in common_ids:
        candidates = [
            (run_labels[i], judged_by_run[i][exid], preds_by_run[i][exid])
            for i in range(n_runs)
        ]
        passes = [bool(j.get("pass")) for _, j, _ in candidates]

        picked_label, picked_judged, picked_pred, _ = pick_best_candidate(candidates)
        pick_counts[picked_label] += 1

        baseline_pass = bool(baseline_judged[exid].get("pass"))
        best_of_n_pass = bool(picked_judged.get("pass"))
        oracle_pass = any(passes)

        if baseline_pass and best_of_n_pass:
            both_pass += 1
        elif baseline_pass and not best_of_n_pass:
            baseline_only += 1
            lost += 1
        elif not baseline_pass and best_of_n_pass:
            gained += 1
        else:
            both_fail += 1

        axis = picked_judged.get("axis") or metadata.get(exid, {}).get("axis") or "UNKNOWN"
        per_axis_stats[axis]["baseline"].append(baseline_pass)
        per_axis_stats[axis]["best_of_n"].append(best_of_n_pass)

        cand_records = []
        predictions_by_id: Dict[str, Dict[str, Any]] = {}
        for label, judged, pred in candidates:
            run_idx = run_labels.index(label)
            pv = _pass_vector_entry(judged, pass_vector_by_run[run_idx].get(exid))
            enriched = _enrich_prediction(pred, judged, pv, label)
            predictions_by_run[label].append(enriched)
            predictions_by_id[label] = enriched
            cand_records.append({
                "run": label,
                "pass": pv["pass"],
                "pass_vector": pv,
                "n_met": enriched["n_met"],
                "n_rubrics": pv["n_rubrics"],
                "n_failed_checks": pv["n_failed_checks"],
                "rubric_summary": enriched["rubric_summary"],
                "prediction": enriched,
                "model_response": enriched.get("model_response") or judged.get("model_response") or "",
                "pred_error": pred.get("error"),
                "rubric_results": judged.get("rubric_results") or [],
            })

        meta_row = metadata.get(exid, {})
        last_turn = meta_row.get("last_user_turn")
        final_query = ""
        if last_turn is not None:
            final_query = str(meta_row.get(f"user_turn_{last_turn}_transcript") or "").strip()

        rec = {
            "id": exid,
            "axis": axis,
            "baseline_run": baseline_label,
            "baseline_pass": baseline_pass,
            "best_of_n_pass": best_of_n_pass,
            "oracle_pass": oracle_pass,
            "improved": (not baseline_pass) and best_of_n_pass,
            "regressed": baseline_pass and (not best_of_n_pass),
            "picked_run": picked_label,
            "passes_by_run": dict(zip(run_labels, passes)),
            "candidates": cand_records,
            "predictions_by_run": predictions_by_id,
            "final_user_query": final_query,
            "picked_response": picked_judged.get("model_response") or "",
        }
        per_example.append(rec)

        out_judged = dict(picked_judged)
        out_judged["_best_of_n"] = {
            "picked_run": picked_label,
            "baseline_run": baseline_label,
            "baseline_pass": baseline_pass,
            "candidate_runs": run_labels,
        }
        best_of_n_judged_rows.append(out_judged)

        out_pred = dict(picked_pred)
        out_pred["_best_of_n_picked_run"] = picked_label
        best_of_n_preds_rows.append(out_pred)

    best_of_n_apr = average_pass_rate(best_of_n_judged_rows)
    best_of_n_ars = average_rubric_score(best_of_n_judged_rows)
    random_expected_apr = sum(sum(r["passes_by_run"].values()) / n_runs for r in per_example) / len(per_example)

    per_axis_summary = {
        axis: {
            "n": len(vals["baseline"]),
            "baseline_apr": sum(vals["baseline"]) / len(vals["baseline"]),
            "best_of_n_apr": sum(vals["best_of_n"]) / len(vals["best_of_n"]),
            "oracle_apr": sum(
                1 for ex in per_example if ex["axis"] == axis and ex["oracle_pass"]
            ) / len(vals["baseline"]),
        }
        for axis, vals in sorted(per_axis_stats.items())
    }

    category_counts = {
        "both_pass": both_pass,
        "baseline_only": baseline_only,
        "best_of_n_only": gained,
        "both_fail": both_fail,
    }

    summary = {
        "n": len(common_ids),
        "n_runs": n_runs,
        "run_labels": run_labels,
        "run_dirs": [str(d) for d in run_dirs],
        "apr_by_run": {k: round(v, 6) for k, v in apr_by_run.items()},
        "baseline_run": baseline_label,
        "baseline_apr": round(baseline_apr, 6),
        "baseline_ars": round(average_rubric_score(baseline_judged.values()), 6),
        "best_of_n_apr": round(best_of_n_apr, 6),
        "best_of_n_ars": round(best_of_n_ars, 6),
        "random_expected_apr": round(random_expected_apr, 6),
        "oracle_apr": round(sum(r["oracle_pass"] for r in per_example) / len(per_example), 6),
        "gain_vs_baseline": round(best_of_n_apr - baseline_apr, 6),
        "gained_pass": gained,
        "lost_pass": lost,
        "both_pass": both_pass,
        "baseline_only_pass": baseline_only,
        "both_fail": both_fail,
        "picked_run_counts": dict(pick_counts),
        "by_axis": {
            axis: {k: round(v, 6) if isinstance(v, float) else v for k, v in stats.items()}
            for axis, stats in per_axis_summary.items()
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_at": stamp,
    }

    improved = [r for r in per_example if r["improved"]]
    regressed = [r for r in per_example if r["regressed"]]

    write_jsonl(out_dir / "best_of_n_judged.jsonl", best_of_n_judged_rows)
    write_jsonl(out_dir / "best_of_n_predictions.jsonl", best_of_n_preds_rows)
    write_jsonl(out_dir / "per_example.jsonl", per_example)
    write_jsonl(out_dir / "improved.jsonl", improved)

    preds_dir = out_dir / "predictions_by_run"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for label in run_labels:
        rows = sorted(predictions_by_run[label], key=lambda r: r["id"])
        write_jsonl(preds_dir / f"predictions_{label}.jsonl", rows)
        with open(preds_dir / f"predictions_{label}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "improved.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_improved": len(improved),
            "baseline_run": baseline_label,
            "baseline_apr": baseline_apr,
            "best_of_n_apr": best_of_n_apr,
            "examples": improved,
        }, f, indent=2, ensure_ascii=False)

    with open(out_dir / "regressed.json", "w", encoding="utf-8") as f:
        json.dump({"n_regressed": len(regressed), "examples": regressed}, f, indent=2, ensure_ascii=False)

    all_summary = build_summary(best_of_n_judged_rows)
    all_summary.update({
        "selection": "best_of_n_oracle",
        "baseline_run": baseline_label,
        "candidate_runs": run_labels,
    })
    with open(out_dir / "all.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    plot_results(
        out_dir,
        run_labels=run_labels,
        per_run_apr=apr_by_run,
        baseline_label=baseline_label,
        baseline_apr=baseline_apr,
        best_of_n_apr=best_of_n_apr,
        per_axis=per_axis_summary,
        category_counts=category_counts,
    )

    logger.info("")
    logger.info("=" * 62)
    logger.info("  BEST-OF-%d RESULTS (selection by example pass / APR)", n_runs)
    logger.info("=" * 62)
    logger.info("  Examples: %d", len(common_ids))
    for label in run_labels:
        logger.info("  run %-8s APR=%.4f", label, apr_by_run[label])
    logger.info("  baseline (%s)     APR=%.4f", baseline_label, baseline_apr)
    logger.info("  random expected   APR=%.4f", random_expected_apr)
    logger.info("  best-of-%d         APR=%.4f  gain=%+.4f", n_runs, best_of_n_apr, best_of_n_apr - baseline_apr)
    logger.info("  oracle (any pass) APR=%.4f", summary["oracle_apr"])
    logger.info("")
    logger.info("  gained (baseline fail -> best-of-N pass): %d", gained)
    logger.info("  lost   (baseline pass -> best-of-N fail): %d", lost)
    logger.info("  both pass: %d | baseline only: %d | both fail: %d", both_pass, baseline_only, both_fail)
    logger.info("")
    logger.info("  Picked run counts: %s", dict(pick_counts))
    logger.info("  Output: %s", out_dir)
    logger.info("=" * 62)

    print(f"\nWrote best-of-{n_runs} results to {out_dir}")
    print(f"  summary.json, improved.json ({len(improved)} examples), plots/")
    print(f"  predictions_by_run/predictions_<run_id>.{{json,jsonl}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
