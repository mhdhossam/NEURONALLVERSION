#!/usr/bin/env python3
"""Open-set image benchmark for Neuron's repository metadata-selection guard.

The guard functions/constants are AST-extracted directly from Untitled36.ipynb,
so the primary run exercises the repository implementation rather than a copy.
Digits 0-7 are known classes; 8-9 are kept out of training and evaluated as OOD.
This offline guard-policy diagnostic is not a FaceNet/full face-recognition benchmark.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Untitled36.ipynb"
SEED = 2026
KNOWN_LABELS = set(range(8))
UNKNOWN_LABELS = {8, 9}


def load_repository_guard(notebook_path: Path = NOTEBOOK) -> dict:
    """Load the exact metadata-guard functions and constants from the notebook."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    guard_cell = next(
        ("".join(cell.get("source", [])) for cell in notebook.get("cells", [])
         if "# Metadata-selection guard" in "".join(cell.get("source", []))
         and "def find_best_in_candidates" in "".join(cell.get("source", []))),
        None,
    )
    if guard_cell is None:
        raise RuntimeError("Metadata guard not found in Untitled36.ipynb")
    tree = ast.parse(guard_cell)
    assignments = {
        "FEATURES_BY_NEURON", "NEURON_METADATA_MIN_SIMILARITY",
        "NEURON_METADATA_MIN_MARGIN", "NEURON_METADATA_TOP_EXEMPLARS",
        "LAST_METADATA_DECISION",
    }
    functions = {"_normalise_embedding", "find_best_in_candidates"}
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & assignments:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(notebook_path), "exec"), namespace)
    missing = (assignments | functions) - set(namespace)
    if missing:
        raise RuntimeError(f"Extracted guard is incomplete: {sorted(missing)}")
    return namespace


def run_guard_smoke_tests(namespace: dict) -> dict:
    """Check strong match acceptance and the guard's three rejection cases."""
    feature_bank = namespace["FEATURES_BY_NEURON"]
    old_features = dict(feature_bank)
    old_decision = dict(namespace["LAST_METADATA_DECISION"])
    checks = {}
    try:
        feature_bank.clear()
        feature_bank.update({
            "good": [np.array([1., 0., 0.], np.float32), np.array([.99, .01, 0.], np.float32)],
            "other": [np.array([0., 1., 0.], np.float32), np.array([0., .99, .01], np.float32)],
        })
        brain = SimpleNamespace(neurons={
            "good": SimpleNamespace(soma=SimpleNamespace(metadata={"label": "correct"})),
            "other": SimpleNamespace(soma=SimpleNamespace(metadata={"label": "other"})),
            "missing": SimpleNamespace(soma=SimpleNamespace(metadata={})),
        })
        selected, score = namespace["find_best_in_candidates"](
            brain, np.array([1., 0., 0.]), {"good", "other"})
        checks["strong_match_accepted"] = (
            selected == "good" and score >= namespace["NEURON_METADATA_MIN_SIMILARITY"])

        feature_bank["other"] = list(feature_bank["good"])
        selected, _ = namespace["find_best_in_candidates"](
            brain, np.array([1., 0., 0.]), {"good", "other"})
        checks["ambiguous_match_rejected"] = (
            selected is None and namespace["LAST_METADATA_DECISION"].get("reason") == "ambiguous_top_candidates")

        feature_bank.clear()
        feature_bank["other"] = [np.array([0., 1., 0.], np.float32)]
        selected, _ = namespace["find_best_in_candidates"](
            brain, np.array([1., 0., 0.]), {"other"})
        checks["weak_visual_match_rejected"] = (
            selected is None and namespace["LAST_METADATA_DECISION"].get("reason") == "similarity_below_threshold")

        feature_bank["good"] = [np.array([1., 0., 0.], np.float32)]
        selected, _ = namespace["find_best_in_candidates"](
            brain, np.array([1., 0., 0.]), {"missing"})
        checks["missing_label_metadata_rejected"] = (
            selected is None and namespace["LAST_METADATA_DECISION"].get("reason") == "no_valid_metadata_candidates")
    finally:
        feature_bank.clear()
        feature_bank.update(old_features)
        namespace["LAST_METADATA_DECISION"].clear()
        namespace["LAST_METADATA_DECISION"].update(old_decision)
    if not all(checks.values()):
        raise AssertionError(f"Metadata guard smoke tests failed: {checks}")
    return {"passed": len(checks), "checks": checks}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def evaluate(predictions, truth, known_mask, latencies) -> dict:
    pred, truth = np.asarray(predictions, dtype=object), np.asarray(truth)
    known = np.asarray(known_mask, dtype=bool)
    unknown = ~known
    accepted = np.array([item is not None for item in pred])
    correct_known = np.array([
        item is not None and int(item) == int(label) for item, label in zip(pred, truth)
    ]) & known
    accepted_known = accepted & known
    rejected_unknown = (~accepted) & unknown
    return {
        "known_test_count": int(known.sum()),
        "unknown_test_count": int(unknown.sum()),
        "known_end_to_end_accuracy": round(float(correct_known.sum() / max(1, known.sum())), 4),
        "known_coverage": round(float(accepted_known.sum() / max(1, known.sum())), 4),
        "known_selective_accuracy": round(float(correct_known.sum() / max(1, accepted_known.sum())), 4),
        "unknown_rejection_rate": round(float(rejected_unknown.sum() / max(1, unknown.sum())), 4),
        "unknown_false_accept_rate": round(float((accepted & unknown).sum() / max(1, unknown.sum())), 4),
        "safe_overall_rate": round(float((correct_known.sum() + rejected_unknown.sum()) / max(1, len(truth))), 4),
        "latency_ms_mean": round(float(statistics.mean(latencies)), 4),
        "latency_ms_p50": round(float(percentile(latencies, .50)), 4),
        "latency_ms_p95": round(float(percentile(latencies, .95)), 4),
    }


def class_score(query, exemplars, top_k=3):
    values = [float(np.dot(query, item)) for item in exemplars
              if np.asarray(item).shape == query.shape and np.isfinite(item).all()]
    return float(np.mean(sorted(values, reverse=True)[:top_k])) if values else None


def build_cnn(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 2 * 2, 64), nn.ReLU(), nn.Linear(64, 8),
    )


def main() -> None:
    global SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", default="benchmarks/results/metadata-guard-classification-results.json")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    SEED = args.seed

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    dataset = load_digits()
    images = dataset.images.astype(np.float32) / 16.0
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        images, dataset.target.astype(int), test_size=.25, stratify=dataset.target,
        random_state=SEED,
    )
    keep_known = np.isin(y_train_all, list(KNOWN_LABELS))
    X_train_pool, y_train_pool = X_train_all[keep_known], y_train_all[keep_known]
    # Reserve known-only calibration data so the CNN can use a fair confidence-based
    # rejection policy without ever tuning on the withheld unknown test classes.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool, y_train_pool, test_size=0.15, stratify=y_train_pool,
        random_state=SEED,
    )
    known_mask = np.isin(y_test, list(KNOWN_LABELS))

    # Both graph-based runs get identical normalized image features and exemplars.
    train_features = X_train.reshape(len(X_train), -1)
    test_features = X_test.reshape(len(X_test), -1)
    train_features = train_features / np.maximum(np.linalg.norm(train_features, axis=1, keepdims=True), 1e-12)
    test_features = test_features / np.maximum(np.linalg.norm(test_features, axis=1, keepdims=True), 1e-12)

    guard = load_repository_guard()
    smoke = run_guard_smoke_tests(guard)
    brain = SimpleNamespace(neurons={
        str(label): SimpleNamespace(soma=SimpleNamespace(metadata={"label": str(label)}))
        for label in sorted(KNOWN_LABELS)
    })
    bank = guard["FEATURES_BY_NEURON"]
    bank.clear()
    for label in sorted(KNOWN_LABELS):
        bank[str(label)] = [row.astype(np.float32) for row in train_features[y_train == label]]
    candidates = set(brain.neurons)

    guarded, guarded_ms, plain, plain_ms = [], [], [], []
    for feature in test_features:
        start = time.perf_counter_ns()
        neuron_id, _ = guard["find_best_in_candidates"](brain, feature, candidates)
        guarded_ms.append((time.perf_counter_ns() - start) / 1e6)
        guarded.append(int(neuron_id) if neuron_id is not None else None)

        start = time.perf_counter_ns()
        query = guard["_normalise_embedding"](feature)
        scores = [(class_score(query, bank[str(label)], guard["NEURON_METADATA_TOP_EXEMPLARS"]), label)
                  for label in sorted(KNOWN_LABELS)]
        scores = [(score, label) for score, label in scores if score is not None]
        plain.append(max(scores)[1] if scores else None)
        plain_ms.append((time.perf_counter_ns() - start) / 1e6)

    # A small end-to-end CNN baseline trained only on known classes.
    torch.manual_seed(SEED)
    model = build_cnn(SEED).cpu()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train[:, None], dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    optimizer, loss_fn = torch.optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss()
    train_start = time.perf_counter()
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    cnn_train_seconds = time.perf_counter() - train_start

    model.eval()
    cnn, cnn_rejected, cnn_ms = [], [], []
    with torch.inference_mode():
        val_tensor = torch.tensor(X_val[:, None], dtype=torch.float32)
        val_confidence = model(val_tensor).softmax(dim=1).max(dim=1).values.cpu().numpy()
        # Calibrate the CNN rejection threshold using only known-class validation images.
        # The threshold targets ~95% validation coverage and never uses unknown labels.
        cnn_rejection_threshold = float(np.quantile(val_confidence, 0.05))
        for image in X_test:
            tensor = torch.tensor(image[None, None], dtype=torch.float32)
            start = time.perf_counter_ns()
            probabilities = model(tensor).softmax(dim=1)[0]
            confidence, prediction = probabilities.max(dim=0)
            predicted_label = int(prediction.item())
            accepted = float(confidence.item()) >= cnn_rejection_threshold
            cnn.append(predicted_label)
            cnn_rejected.append(predicted_label if accepted else None)
            cnn_ms.append((time.perf_counter_ns() - start) / 1e6)
    cnn_calibration_coverage = float(np.mean(val_confidence >= cnn_rejection_threshold))

    result = {
        "benchmark": "Neuron metadata guard open-set image classification",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "dataset": "sklearn.datasets.load_digits (8x8 grayscale; offline)",
        "protocol": {
            "split": "stratified 75/25 train/test; known training pool further split 85/15 fit/calibration; random_state=2026",
            "known_classes": sorted(KNOWN_LABELS),
            "unknown_classes_held_out_from_training": sorted(UNKNOWN_LABELS),
            "train_samples_known_fit": int(len(y_train)),
            "known_validation_samples_for_cnn_confidence_calibration": int(len(y_val)),
            "test_samples_total": int(len(y_test)),
            "test_samples_known": int(known_mask.sum()),
            "test_samples_unknown": int((~known_mask).sum()),
            "neuron_features": "L2-normalized flattened pixels; class score is mean top-3 cosine similarity",
            "guard_source": "AST-extracted _normalise_embedding/find_best_in_candidates and constants from Untitled36.ipynb",
            "guard_thresholds": {
                "minimum_similarity": float(guard["NEURON_METADATA_MIN_SIMILARITY"]),
                "minimum_margin": float(guard["NEURON_METADATA_MIN_MARGIN"]),
                "top_exemplars": int(guard["NEURON_METADATA_TOP_EXEMPLARS"]),
            },
            "cnn": f"two-convolution CPU CNN trained for {args.epochs} epochs on fit split; unknown classes are excluded from training",
            "cnn_rejection_calibration": "confidence threshold is the 5th percentile of max-softmax confidence on a separate known-only calibration split; unknown test labels are not used",
            "cnn_rejection_threshold": cnn_rejection_threshold,
            "cnn_calibration_coverage": round(cnn_calibration_coverage, 4),
            "limitation": "Controlled small-image guard-policy comparison, not a FaceNet benchmark or a run on the unavailable local face dataset.",
        },
        "guard_smoke_tests": smoke,
        "cnn_training_seconds": round(cnn_train_seconds, 4),
        "results": {
            "neuron_metadata_guard": evaluate(guarded, y_test, known_mask, guarded_ms),
            "neuron_same_similarity_no_guard_ablation": evaluate(plain, y_test, known_mask, plain_ms),
            "small_cnn_baseline_closed_set": evaluate(cnn, y_test, known_mask, cnn_ms),
            "small_cnn_baseline_with_calibrated_rejection": evaluate(cnn_rejected, y_test, known_mask, cnn_ms),
        },
    }
    output = Path(args.json_output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved results: {output}")


if __name__ == "__main__":
    main()
