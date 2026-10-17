#!/usr/bin/env python3
"""Run repeatable metadata-guard benchmarks across multiple fixed random seeds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUN = ROOT / "benchmarks" / "benchmark_metadata_guard_classification.py"
DEFAULT_SEEDS = [2022, 2023, 2024, 2026]
METHODS = {
    "neuron_guard": "neuron_metadata_guard",
    "neuron_without_guard": "neuron_same_similarity_no_guard_ablation",
    "cnn_closed_set": "small_cnn_baseline_closed_set",
    "cnn_calibrated_rejection": "small_cnn_baseline_with_calibrated_rejection",
}
METRICS = [
    "known_end_to_end_accuracy", "known_coverage", "known_selective_accuracy",
    "unknown_rejection_rate", "unknown_false_accept_rate", "safe_overall_rate",
]


def summarize(reports: list[dict]) -> dict:
    summary = {}
    for alias, result_key in METHODS.items():
        summary[alias] = {}
        for metric in METRICS:
            values = [report["results"][result_key][metric] for report in reports]
            summary[alias][metric] = {
                "mean": round(statistics.mean(values), 4),
                "std_population": round(statistics.pstdev(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument(
        "--json-output",
        default="benchmarks/results/metadata-guard-multiseed-summary.json",
    )
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain one or more unique integer seeds")

    reports = []
    with tempfile.TemporaryDirectory(prefix="neuron-guard-benchmark-") as tmp:
        for seed in seeds:
            output_path = Path(tmp) / f"results-{seed}.json"
            command = [
                sys.executable, str(SINGLE_RUN),
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--json-output", str(output_path),
            ]
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Benchmark failed for seed {seed} (exit {completed.returncode}).\n"
                    f"STDOUT:\n{completed.stdout[-4000:]}\nSTDERR:\n{completed.stderr[-4000:]}"
                )
            report = json.loads(output_path.read_text(encoding="utf-8"))
            report["seed"] = seed
            reports.append(report)
            print(
                f"seed={seed} "
                f"guard_acc={report['results']['neuron_metadata_guard']['known_end_to_end_accuracy']:.4f} "
                f"cnn_acc={report['results']['small_cnn_baseline_closed_set']['known_end_to_end_accuracy']:.4f} "
                f"guard_unknown_reject={report['results']['neuron_metadata_guard']['unknown_rejection_rate']:.4f} "
                f"cnn_unknown_reject={report['results']['small_cnn_baseline_with_calibrated_rejection']['unknown_rejection_rate']:.4f}"
            )

    summary = summarize(reports)
    guard = summary["neuron_guard"]
    cnn_closed = summary["cnn_closed_set"]
    cnn_reject = summary["cnn_calibrated_rejection"]
    result = {
        "status": "EXECUTED",
        "benchmark": "Multi-seed Neuron metadata guard open-set image classification",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "dataset": reports[0]["dataset"],
        "protocol": reports[0]["protocol"],
        "guard_smoke_tests_passed_all_runs": all(
            report["guard_smoke_tests"].get("passed") == 4 for report in reports
        ),
        "summary_mean_population_std": summary,
        "derived_differences_percentage_points": {
            "neuron_guard_minus_cnn_closed_known_accuracy": round(
                100 * (
                    guard["known_end_to_end_accuracy"]["mean"]
                    - cnn_closed["known_end_to_end_accuracy"]["mean"]
                ), 2
            ),
            "neuron_guard_minus_calibrated_cnn_known_coverage": round(
                100 * (
                    guard["known_coverage"]["mean"]
                    - cnn_reject["known_coverage"]["mean"]
                ), 2
            ),
            "neuron_guard_minus_calibrated_cnn_selective_accuracy": round(
                100 * (
                    guard["known_selective_accuracy"]["mean"]
                    - cnn_reject["known_selective_accuracy"]["mean"]
                ), 2
            ),
            "neuron_guard_minus_calibrated_cnn_unknown_rejection": round(
                100 * (
                    guard["unknown_rejection_rate"]["mean"]
                    - cnn_reject["unknown_rejection_rate"]["mean"]
                ), 2
            ),
            "neuron_guard_minus_calibrated_cnn_safe_overall": round(
                100 * (
                    guard["safe_overall_rate"]["mean"]
                    - cnn_reject["safe_overall_rate"]["mean"]
                ), 2
            ),
        },
        "runs": [
            {
                "seed": report["seed"],
                "train_samples_known_fit": report["protocol"]["train_samples_known_fit"],
                "validation_samples": report["protocol"]["known_validation_samples_for_cnn_confidence_calibration"],
                "results": report["results"],
                "guard_smoke_tests": report["guard_smoke_tests"],
            }
            for report in reports
        ],
        "limitation": (
            "This benchmark tests the metadata guard/exemplar decision policy on normalized "
            "8x8 digit pixels. It is not the full Neuron face-recognition pipeline or a "
            "FaceNet embedding evaluation, and does not establish broad CNN superiority."
        ),
    }
    output = Path(args.json_output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "saved": str(output),
        "seeds": seeds,
        "guard_smoke_tests_passed_all_runs": result["guard_smoke_tests_passed_all_runs"],
        "summary_mean_population_std": summary,
        "derived_differences_percentage_points": result["derived_differences_percentage_points"],
    }, indent=2))


if __name__ == "__main__":
    main()
