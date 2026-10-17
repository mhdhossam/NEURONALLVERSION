#!/usr/bin/env python3
"""Test error-driven exemplar learning using Neuron's actual notebook metadata guard.

Protocol:
- FaceNet embeddings come from the same LFW preprocessing/checkpoint as the real-face benchmark.
- Each known identity is split into initial memory, a sequential feedback stream, and a frozen audit set.
- The frozen baseline predicts the entire stream without updates.
- The online system predicts one item at a time; only after a wrong/rejected prediction does it
  receive the correct label and append that sample as a new exemplar for the correct identity.
- It then replays the corrected item and evaluates an untouched audit set before/after learning.
- Held-out unknown identities are never used for feedback or added to memory.

This tests a narrow, explicit form of learning from mistakes, not general human-like learning.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
import warnings
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from sklearn.datasets import fetch_lfw_people

from benchmark_neuron_facenet_lfw import (
    guard_regression_checks,
    load_repository_guard,
    safe_normalize,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [2026, 2027, 2028]


def infer(guard: dict, brain: SimpleNamespace, candidates: set[str], embedding: np.ndarray):
    node_id, score = guard["find_best_in_candidates"](brain, embedding, candidates)
    label = brain.neurons[node_id].soma.metadata["label"] if node_id is not None else None
    return label, float(score), dict(guard["LAST_METADATA_DECISION"])


def evaluate_known(predictions: list[str | None], truth: list[str]) -> dict:
    n = len(truth)
    if not n:
        return {"count": 0, "accuracy": None, "coverage": None, "selective_accuracy": None}
    accepted = np.array([p is not None for p in predictions], dtype=bool)
    correct = np.array([p == y for p, y in zip(predictions, truth)], dtype=bool)
    correct_accepted = accepted & correct
    return {
        "count": n,
        "correct": int(correct.sum()),
        "accuracy": round(float(correct.mean()), 4),
        "coverage": round(float(accepted.mean()), 4),
        "selective_accuracy": round(float(correct_accepted.sum() / max(1, accepted.sum())), 4),
        "rejected": int((~accepted).sum()),
        "wrong_accepted": int((accepted & ~correct).sum()),
    }


def evaluate_unknown(predictions: list[str | None]) -> dict:
    n = len(predictions)
    rejected = sum(p is None for p in predictions)
    return {
        "count": n,
        "rejected": int(rejected),
        "rejection_rate": round(rejected / max(1, n), 4),
        "false_accepts": int(n - rejected),
        "false_accept_rate": round((n - rejected) / max(1, n), 4),
    }


def mean_std(values: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 4) if values else None,
        "std_population": round(statistics.pstdev(values), 4) if values else None,
        "min": round(min(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--known-classes", type=int, default=8)
    parser.add_argument("--unknown-classes", type=int, default=4)
    parser.add_argument("--max-known-per-person", type=int, default=70)
    parser.add_argument("--max-unknown-per-person", type=int, default=30)
    parser.add_argument("--json-output", default="benchmarks/results/neuron-error-learning-results.json")
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique integers")
    if args.known_classes < 2 or args.unknown_classes < 1:
        raise ValueError("Need at least two known classes and one unknown class")

    seed0 = seeds[0]
    random.seed(seed0)
    np.random.seed(seed0)
    torch.manual_seed(seed0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Python={platform.python_version()} torch={torch.__version__} device={device}", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lfw = fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0, funneled=True)
    images, labels = lfw.images, lfw.target.astype(int)
    counts = np.bincount(labels)
    ordered_classes = sorted(
        np.unique(labels).tolist(),
        key=lambda label: (-int(counts[label]), str(lfw.target_names[label]).casefold()),
    )
    required = args.known_classes + args.unknown_classes
    if len(ordered_classes) < required:
        raise RuntimeError(f"Need {required} identities, got {len(ordered_classes)}")
    known_ids = ordered_classes[:args.known_classes]
    unknown_ids = ordered_classes[args.known_classes:required]
    known_names = [str(lfw.target_names[i]) for i in known_ids]
    unknown_names = [str(lfw.target_names[i]) for i in unknown_ids]
    print("Known identities:", known_names, flush=True)
    print("Unknown identities (feedback prohibited):", unknown_names, flush=True)

    selected: list[tuple[int, int, str]] = []
    for class_id in known_ids + unknown_ids:
        name = str(lfw.target_names[class_id])
        rng = np.random.default_rng(seed0 + int(class_id) * 7919)
        indices = np.flatnonzero(labels == class_id).copy()
        rng.shuffle(indices)
        cap = args.max_known_per_person if class_id in known_ids else args.max_unknown_per_person
        selected.extend((int(i), int(class_id), name) for i in indices[:cap])

    # Keep the same pre-cropped LFW -> 160px -> [-1,1] preprocessing as the FaceNet benchmark.
    face_crops, person_names = [], []
    started = time.perf_counter()
    for j, (source_index, _class_id, name) in enumerate(selected, start=1):
        raw = np.asarray(images[source_index], dtype=np.float32)
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise RuntimeError(f"Expected RGB LFW crop, got {raw.shape}")
        if float(np.nanmax(raw)) <= 1.0:
            raw = raw * 255.0
        rgb = Image.fromarray(np.clip(raw, 0, 255).astype(np.uint8), mode="RGB")
        rgb = rgb.resize((160, 160), resample=Image.Resampling.BILINEAR)
        crop = torch.from_numpy(np.asarray(rgb, dtype=np.float32)).permute(2, 0, 1).contiguous()
        crop = crop / 127.5 - 1.0
        face_crops.append(crop)
        person_names.append(name)
        if j % 100 == 0 or j == len(selected):
            print(f"Prepared crops {j}/{len(selected)}", flush=True)
    crops = torch.stack(face_crops)
    people = np.asarray(person_names, dtype=object)
    preprocessing_seconds = time.perf_counter() - started

    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    started = time.perf_counter()
    embedding_batches = []
    with torch.inference_mode():
        for begin in range(0, len(crops), 32):
            embedding_batches.append(model(crops[begin:begin + 32].to(device)).cpu().numpy())
            print(f"FaceNet embeddings {min(begin + 32, len(crops))}/{len(crops)}", flush=True)
    embeddings = safe_normalize(np.concatenate(embedding_batches, axis=0).astype(np.float32))
    embedding_seconds = time.perf_counter() - started

    guard = load_repository_guard()
    smoke = guard_regression_checks(guard)
    known_set = set(known_names)
    label_to_node = {name: f"identity_{i}" for i, name in enumerate(known_names)}
    brain = SimpleNamespace(neurons={
        label_to_node[name]: SimpleNamespace(soma=SimpleNamespace(metadata={"label": name}))
        for name in known_names
    })
    candidates = set(brain.neurons)
    bank = guard["FEATURES_BY_NEURON"]
    per_seed = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        base_idx, feedback_idx, audit_idx = [], [], []
        for name in known_names:
            person_idx = np.flatnonzero(people == name).copy()
            rng.shuffle(person_idx)
            n = len(person_idx)
            # Allocate about 55% to initial memory, 20% to feedback, and the rest to frozen audit.
            n_base = max(6, int(n * 0.55))
            n_feedback = max(3, int(n * 0.20))
            if n_base + n_feedback > n - 2:
                n_base = max(4, n - max(4, int(n * 0.30)))
                n_feedback = max(2, n - n_base - 2)
            base_idx.extend(person_idx[:n_base].tolist())
            feedback_idx.extend(person_idx[n_base:n_base + n_feedback].tolist())
            audit_idx.extend(person_idx[n_base + n_feedback:].tolist())

        base_idx = np.asarray(base_idx, dtype=int)
        feedback_idx = np.asarray(feedback_idx, dtype=int)
        audit_known_idx = np.asarray(audit_idx, dtype=int)
        rng.shuffle(feedback_idx)
        unknown_idx = np.flatnonzero(np.isin(people, unknown_names))

        # Initialise memory from base partition only.
        bank.clear()
        for name in known_names:
            rows = embeddings[[i for i in base_idx if people[i] == name]]
            bank[label_to_node[name]] = [row.copy() for row in rows]

        base_counts = {name: len(bank[label_to_node[name]]) for name in known_names}
        truth_feedback = people[feedback_idx].tolist()
        truth_audit = people[audit_known_idx].tolist()

        # Frozen/no-learning control: entire stream scored against an unchanged copy of base memory.
        base_snapshot = {nid: [row.copy() for row in rows] for nid, rows in bank.items()}
        baseline_feedback_predictions = [
            infer(guard, brain, candidates, embeddings[i])[0] for i in feedback_idx
        ]
        baseline_audit_predictions = [
            infer(guard, brain, candidates, embeddings[i])[0] for i in audit_known_idx
        ]
        baseline_unknown_predictions = [
            infer(guard, brain, candidates, embeddings[i])[0] for i in unknown_idx
        ]
        baseline_feedback = evaluate_known(baseline_feedback_predictions, truth_feedback)
        baseline_audit = evaluate_known(baseline_audit_predictions, truth_audit)
        baseline_unknown = evaluate_unknown(baseline_unknown_predictions)

        # Reset to identical memory; prequential online learning sees labels only after each prediction.
        bank.clear()
        bank.update({nid: [row.copy() for row in rows] for nid, rows in base_snapshot.items()})
        online_feedback_predictions: list[str | None] = []
        corrections = []
        replay_corrected = 0
        for idx, true_label in zip(feedback_idx, truth_feedback):
            predicted, _score, diagnostic = infer(guard, brain, candidates, embeddings[idx])
            online_feedback_predictions.append(predicted)
            if predicted != true_label:
                # Explicit supervised feedback: update the correct identity's external exemplar bank,
                # preserving the notebook design that embeddings do not enter Soma.data/metadata.
                target_node = label_to_node[true_label]
                normalised = guard["_normalise_embedding"](embeddings[idx])
                if normalised is None:
                    raise RuntimeError("A valid FaceNet embedding failed repository normalisation")
                bank[target_node].append(normalised.copy())
                replay_predicted, replay_score, replay_diag = infer(
                    guard, brain, candidates, embeddings[idx]
                )
                fixed = replay_predicted == true_label
                replay_corrected += int(fixed)
                corrections.append({
                    "true_label": true_label,
                    "prediction_before_feedback": predicted,
                    "reason_before_feedback": diagnostic.get("reason"),
                    "prediction_after_feedback_replay": replay_predicted,
                    "reason_after_feedback_replay": replay_diag.get("reason"),
                    "replay_fixed": bool(fixed),
                    "score_after_feedback_replay": round(float(replay_score), 6),
                })

        online_feedback = evaluate_known(online_feedback_predictions, truth_feedback)

        # Audit items are never added to memory. Compare the original and adapted systems on them.
        post_audit_predictions = [
            infer(guard, brain, candidates, embeddings[i])[0] for i in audit_known_idx
        ]
        post_unknown_predictions = [
            infer(guard, brain, candidates, embeddings[i])[0] for i in unknown_idx
        ]
        post_audit = evaluate_known(post_audit_predictions, truth_audit)
        post_unknown = evaluate_unknown(post_unknown_predictions)

        # Did feedback corrupt anything that the baseline answered correctly on the frozen audit?
        audit_regressions = [
            {"true_label": truth, "before": before, "after": after}
            for truth, before, after in zip(truth_audit, baseline_audit_predictions, post_audit_predictions)
            if before == truth and after != truth
        ]
        audit_repairs = [
            {"true_label": truth, "before": before, "after": after}
            for truth, before, after in zip(truth_audit, baseline_audit_predictions, post_audit_predictions)
            if before != truth and after == truth
        ]
        unknown_changed_accepts = int(sum(
            before is None and after is not None
            for before, after in zip(baseline_unknown_predictions, post_unknown_predictions)
        ))

        row = {
            "seed": seed,
            "counts": {
                "known_base_memory": int(len(base_idx)),
                "known_feedback_stream": int(len(feedback_idx)),
                "known_frozen_audit": int(len(audit_known_idx)),
                "unknown_frozen_probe": int(len(unknown_idx)),
                "initial_exemplars_by_identity": base_counts,
                "correction_events": len(corrections),
            },
            "frozen_baseline": {
                "feedback_stream": baseline_feedback,
                "known_audit": baseline_audit,
                "unknown_probe": baseline_unknown,
            },
            "online_after_feedback": {
                "prequential_feedback_stream": online_feedback,
                "known_audit": post_audit,
                "unknown_probe": post_unknown,
                "correction_replay_fixed": replay_corrected,
                "correction_replay_total": len(corrections),
                "correction_replay_fix_rate": round(replay_corrected / max(1, len(corrections)), 4),
                "frozen_audit_regressions": len(audit_regressions),
                "frozen_audit_repairs": len(audit_repairs),
                "unknowns_changed_from_rejected_to_accepted": unknown_changed_accepts,
            },
            "correction_log": corrections,
            "audit_regression_examples": audit_regressions[:10],
            "audit_repair_examples": audit_repairs[:10],
        }
        per_seed.append(row)
        print(json.dumps({
            "seed": seed,
            "baseline_feedback_accuracy": baseline_feedback["accuracy"],
            "online_feedback_accuracy": online_feedback["accuracy"],
            "correction_events": len(corrections),
            "replay_fixed": replay_corrected,
            "audit_accuracy_before": baseline_audit["accuracy"],
            "audit_accuracy_after": post_audit["accuracy"],
            "audit_regressions": len(audit_regressions),
            "unknown_rejection_before": baseline_unknown["rejection_rate"],
            "unknown_rejection_after": post_unknown["rejection_rate"],
        }, indent=2), flush=True)

    metrics = {
        "baseline_feedback_accuracy": [r["frozen_baseline"]["feedback_stream"]["accuracy"] for r in per_seed],
        "online_feedback_accuracy": [r["online_after_feedback"]["prequential_feedback_stream"]["accuracy"] for r in per_seed],
        "baseline_audit_accuracy": [r["frozen_baseline"]["known_audit"]["accuracy"] for r in per_seed],
        "post_learning_audit_accuracy": [r["online_after_feedback"]["known_audit"]["accuracy"] for r in per_seed],
        "baseline_unknown_rejection": [r["frozen_baseline"]["unknown_probe"]["rejection_rate"] for r in per_seed],
        "post_learning_unknown_rejection": [r["online_after_feedback"]["unknown_probe"]["rejection_rate"] for r in per_seed],
        "correction_events": [float(r["counts"]["correction_events"]) for r in per_seed],
        "replay_fix_rate": [r["online_after_feedback"]["correction_replay_fix_rate"] for r in per_seed if r["counts"]["correction_events"]],
        "audit_regressions": [float(r["online_after_feedback"]["frozen_audit_regressions"]) for r in per_seed],
        "audit_repairs": [float(r["online_after_feedback"]["frozen_audit_repairs"]) for r in per_seed],
    }
    report = {
        "status": "EXECUTED",
        "benchmark": "Neuron metadata-guard error-driven exemplar learning on LFW",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "face_extractor": "facenet-pytorch InceptionResnetV1(pretrained='vggface2')",
            "dataset": "LFW via sklearn.fetch_lfw_people(color=True, resize=1.0, min_faces_per_person=20, funneled=True)",
            "preprocessing": "same as real-face benchmark: pre-cropped LFW RGB crops resized to 160x160, pixels normalized to [-1,1]",
            "preprocessing_seconds": round(preprocessing_seconds, 3),
            "embedding_seconds": round(embedding_seconds, 3),
        },
        "protocol": {
            "known_identities": known_names,
            "held_out_unknown_identities": unknown_names,
            "split": "per known identity: about 55% initial memory, 20% sequential supervised feedback stream, remainder frozen audit; unknown identities are frozen probes only",
            "online_update_rule": "predict first; only on a wrong label or rejection, append that sample's normalized FaceNet embedding to the correct known identity's FEATURES_BY_NEURON list; keep notebook similarity/margin thresholds unchanged",
            "no_test_label_leakage": "feedback labels are revealed only after the online prediction; the frozen audit set and unknown identities never update the memory",
            "guard_source": "functions and thresholds AST-extracted from Untitled36.ipynb via benchmark_neuron_facenet_lfw.py; all guard smoke checks passed",
            "pretraining_overlap_caveat": "VGGFace2 weights may have identity overlap with LFW; this evaluates feedback mechanics on this checkpoint, not decontaminated facial recognition or brain equivalence",
            "limitation": "Correcting an error by storing its labeled example tests supervised exemplar memory. It does not yet test autonomous causal diagnosis of why an error occurred.",
        },
        "guard_regression_checks": smoke,
        "aggregate": {key: mean_std(value) for key, value in metrics.items()},
        "per_seed": per_seed,
        "interpretation": (
            "Compare frozen baseline with prequential online feedback, then inspect the untouched known audit "
            "and held-out unknown probe. A fix is useful only if feedback improves the stream without unacceptable "
            "audit regressions or loss of unknown rejection. Replaying the exact corrected item measures retention "
            "of feedback but is not evidence by itself of generalization."
        ),
    }
    output = Path(args.json_output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n=== ERROR-LEARNING SUMMARY ===", flush=True)
    print(json.dumps({"saved": str(output), "status": report["status"], "aggregate": report["aggregate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
