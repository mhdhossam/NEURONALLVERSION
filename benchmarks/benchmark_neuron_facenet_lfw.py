#!/usr/bin/env python3
"""Evaluate Neuron's notebook metadata guard on real LFW faces with FaceNet.

The guard constants and functions are AST-extracted from Untitled36.ipynb.
A fixed set of identities is divided into known training identities and entirely
held-out unknown identities. The same aligned face crops/splits feed all methods:
(1) Neuron's FaceNet exemplar bank + metadata guard, (2) the same exemplar bank
without rejection, (3) an RBF SVM trained on the same FaceNet embeddings, and
(4) a small CNN trained on the same aligned face crops. SVM/CNN rejection cutoffs
are calibrated from known-only validation images; unknown test images never tune
models or thresholds.

This is an open-set identification benchmark on LFW, not a clinical/security claim.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time
from types import SimpleNamespace
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "Untitled36.ipynb"
DEFAULT_SEEDS = [2026, 2027, 2028]


def load_repository_guard(notebook_path: Path = NOTEBOOK_PATH) -> dict:
    """Load the exact guard functions and configured thresholds from the notebook."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell_source = next(
        ("".join(cell.get("source", [])) for cell in notebook.get("cells", [])
         if "# Metadata-selection guard" in "".join(cell.get("source", []))
         and "def find_best_in_candidates" in "".join(cell.get("source", []))),
        None,
    )
    if cell_source is None:
        raise RuntimeError(f"Metadata-selection guard not found in {notebook_path}")
    tree = ast.parse(cell_source)
    assignments = {
        "FEATURES_BY_NEURON", "NEURON_METADATA_MIN_SIMILARITY",
        "NEURON_METADATA_MIN_MARGIN", "NEURON_METADATA_TOP_EXEMPLARS",
        "LAST_METADATA_DECISION",
    }
    function_names = {"_normalise_embedding", "find_best_in_candidates"}
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & assignments:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected.append(node)
    namespace = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(notebook_path), "exec"), namespace)
    missing = (assignments | function_names) - set(namespace)
    if missing:
        raise RuntimeError(f"Incomplete extracted metadata guard: {sorted(missing)}")
    return namespace


def guard_regression_checks(guard: dict) -> dict:
    """Exercise the actual repository guard: good, ambiguous, weak, missing metadata."""
    features = guard["FEATURES_BY_NEURON"]
    old_features, old_decision = dict(features), dict(guard["LAST_METADATA_DECISION"])
    checks: dict[str, bool] = {}
    try:
        features.clear()
        features.update({
            "known_good": [np.array([1., 0., 0.], np.float32),
                           np.array([.99, .01, 0.], np.float32)],
            "other": [np.array([0., 1., 0.], np.float32),
                      np.array([0., .99, .01], np.float32)],
        })
        brain = SimpleNamespace(neurons={
            "known_good": SimpleNamespace(soma=SimpleNamespace(metadata={"label": "known_good"})),
            "other": SimpleNamespace(soma=SimpleNamespace(metadata={"label": "other"})),
            "missing_label": SimpleNamespace(soma=SimpleNamespace(metadata={})),
        })
        selected, score = guard["find_best_in_candidates"](
            brain, np.array([1., 0., 0.]), {"known_good", "other"})
        checks["strong_unique_match_accepted"] = (
            selected == "known_good" and score >= guard["NEURON_METADATA_MIN_SIMILARITY"])
        features["other"] = list(features["known_good"])
        selected, _ = guard["find_best_in_candidates"](
            brain, np.array([1., 0., 0.]), {"known_good", "other"})
        checks["ambiguous_match_rejected"] = (
            selected is None and guard["LAST_METADATA_DECISION"].get("reason") == "ambiguous_top_candidates")
        features.clear()
        features["other"] = [np.array([0., 1., 0.], np.float32)]
        selected, _ = guard["find_best_in_candidates"](
            brain, np.array([1., 0., 0.]), {"other"})
        checks["weak_visual_match_rejected"] = (
            selected is None and guard["LAST_METADATA_DECISION"].get("reason") == "similarity_below_threshold")
        selected, _ = guard["find_best_in_candidates"](
            brain, np.array([1., 0., 0.]), {"missing_label"})
        checks["missing_metadata_rejected"] = (
            selected is None and guard["LAST_METADATA_DECISION"].get("reason") == "no_valid_metadata_candidates")
    finally:
        features.clear()
        features.update(old_features)
        guard["LAST_METADATA_DECISION"].clear()
        guard["LAST_METADATA_DECISION"].update(old_decision)
    if not all(checks.values()):
        raise AssertionError(f"Repository guard regression checks failed: {checks}")
    return {"passed": len(checks), "checks": checks}


def safe_normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def class_similarity_score(query: np.ndarray, exemplars: list[np.ndarray], top_k: int) -> float | None:
    values = [float(np.dot(query, exemplar)) for exemplar in exemplars
              if np.asarray(exemplar).shape == query.shape and np.isfinite(exemplar).all()]
    return float(np.mean(sorted(values, reverse=True)[:top_k])) if values else None


def evaluate(predictions: list[str | None], truth: list[str], known_people: set[str],
             decision_latencies_ms: list[float]) -> dict:
    accepted = np.array([prediction is not None for prediction in predictions], dtype=bool)
    known = np.array([label in known_people for label in truth], dtype=bool)
    correct_known = np.array([
        pred is not None and pred == label for pred, label in zip(predictions, truth)
    ], dtype=bool) & known
    accepted_known = accepted & known
    accepted_unknown = accepted & ~known
    rejected_unknown = (~accepted) & ~known
    return {
        "known_test_images": int(known.sum()),
        "unknown_test_images": int((~known).sum()),
        "known_end_to_end_accuracy": round(float(correct_known.sum() / max(1, known.sum())), 4),
        "known_coverage": round(float(accepted_known.sum() / max(1, known.sum())), 4),
        "known_selective_accuracy": round(float(correct_known.sum() / max(1, accepted_known.sum())), 4),
        "unknown_rejection_rate": round(float(rejected_unknown.sum() / max(1, (~known).sum())), 4),
        "unknown_false_accept_rate": round(float(accepted_unknown.sum() / max(1, (~known).sum())), 4),
        "safe_overall_rate": round(float((correct_known.sum() + rejected_unknown.sum()) / max(1, len(truth))), 4),
        "decision_latency_ms_mean": round(float(statistics.mean(decision_latencies_ms)), 4),
        "decision_latency_ms_p50": round(float(percentile(decision_latencies_ms, .50)), 4),
        "decision_latency_ms_p95": round(float(percentile(decision_latencies_ms, .95)), 4),
    }


class SmallFaceCNN(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Dropout(p=0.20), nn.Linear(64, class_count),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def train_cnn(images64: torch.Tensor, train_idx: np.ndarray, train_y: np.ndarray,
              validation_idx: np.ndarray, class_count: int, seed: int, epochs: int,
              device: torch.device) -> tuple[SmallFaceCNN, float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = SmallFaceCNN(class_count).to(device)
    dataset = TensorDataset(
        images64[train_idx],
        torch.tensor(train_y, dtype=torch.long),
    )
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    training_seconds = time.perf_counter() - start
    model.eval()
    with torch.inference_mode():
        vals = []
        for begin in range(0, len(validation_idx), 32):
            idx = validation_idx[begin:begin + 32]
            probs = model(images64[idx].to(device)).softmax(dim=1)
            vals.extend(probs.max(dim=1).values.cpu().tolist())
    threshold = float(np.quantile(np.asarray(vals, dtype=np.float32), 0.05))
    return model, threshold, training_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--known-classes", type=int, default=8)
    parser.add_argument("--unknown-classes", type=int, default=4)
    parser.add_argument("--max-known-per-person", type=int, default=90)
    parser.add_argument("--max-unknown-per-person", type=int, default=35)
    parser.add_argument("--cnn-epochs", type=int, default=8)
    parser.add_argument("--json-output", default="benchmarks/results/neuron-facenet-lfw-results.json")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds needs one or more unique integer seeds")
    if args.known_classes < 2 or args.unknown_classes < 1:
        raise ValueError("Need at least two known classes and one unknown class")

    seed0 = seeds[0]
    np.random.seed(seed0)
    random.seed(seed0)
    torch.manual_seed(seed0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Environment: Python {platform.python_version()}, torch {torch.__version__}, device={device}", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lfw = fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0, funneled=True)
    images = lfw.images
    labels = lfw.target.astype(int)
    counts = np.bincount(labels)
    ordered_classes = sorted(
        np.unique(labels).tolist(),
        key=lambda label: (-int(counts[label]), str(lfw.target_names[label]).casefold()),
    )
    required_classes = args.known_classes + args.unknown_classes
    if len(ordered_classes) < required_classes:
        raise RuntimeError(f"LFW has only {len(ordered_classes)} identities; need {required_classes}")
    known_ids = ordered_classes[:args.known_classes]
    unknown_ids = ordered_classes[args.known_classes:required_classes]
    known_names = [str(lfw.target_names[label]) for label in known_ids]
    unknown_names = [str(lfw.target_names[label]) for label in unknown_ids]
    print("Known identities:", known_names, flush=True)
    print("Held-out unknown identities:", unknown_names, flush=True)

    # Deterministically cap images per identity to bound CI time. Unknown identities
    # are never used to fit FaceNet exemplars, SVM, CNN, or calibration thresholds.
    selected: list[tuple[int, int, str]] = []
    class_caps: dict[str, int] = {}
    for class_id in known_ids + unknown_ids:
        name = str(lfw.target_names[class_id])
        rng = np.random.default_rng(seed0 + int(class_id) * 7919)
        indices = np.flatnonzero(labels == class_id).copy()
        rng.shuffle(indices)
        cap = args.max_known_per_person if class_id in known_ids else args.max_unknown_per_person
        selected_indices = indices[:cap]
        class_caps[name] = len(selected_indices)
        selected.extend((int(index), int(class_id), name) for index in selected_indices)
    print(f"Selected {len(selected)} LFW images; LFW supplies pre-cropped/funneled faces", flush=True)

    # fetch_lfw_people returns face crops already aligned/cropped to the subject.
    # Running MTCNN again on these tight crops yields zero detections, so preserve
    # the crop geometry and resize directly to FaceNet's 160x160 input. Standardize
    # pixel values to [-1, 1], matching facenet-pytorch's MTCNN output convention.
    from facenet_pytorch import InceptionResnetV1
    face_crops: list[torch.Tensor] = []
    person_names: list[str] = []
    source_indices: list[int] = []
    preprocessing_seconds_start = time.perf_counter()
    for i, (source_index, class_id, name) in enumerate(selected, start=1):
        raw = np.asarray(images[source_index], dtype=np.float32)
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise RuntimeError(f"Expected RGB LFW crop, got shape {raw.shape}")
        if float(np.nanmax(raw)) <= 1.0:
            raw = raw * 255.0
        rgb = Image.fromarray(np.clip(raw, 0, 255).astype(np.uint8), mode="RGB")
        rgb = rgb.resize((160, 160), resample=Image.Resampling.BILINEAR)
        pixels = np.asarray(rgb, dtype=np.float32)
        crop = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        crop = crop / 127.5 - 1.0
        if crop.shape == (3, 160, 160) and bool(torch.isfinite(crop).all()):
            face_crops.append(crop)
            person_names.append(name)
            source_indices.append(source_index)
        if i % 50 == 0 or i == len(selected):
            print(f"Prepared LFW crops {i}/{len(selected)}", flush=True)
    preprocessing_seconds = time.perf_counter() - preprocessing_seconds_start
    if len(face_crops) < 100:
        raise RuntimeError(f"Too few usable LFW crops after resize/normalization: {len(face_crops)}")
    crops = torch.stack(face_crops)
    people = np.asarray(person_names, dtype=object)
    known_set = set(known_names)
    for name in known_names:
        count = int(np.sum(people == name))
        if count < 12:
            raise RuntimeError(f"Only {count} detected images for known identity {name}; need >=12")
    for name in unknown_names:
        count = int(np.sum(people == name))
        if count < 8:
            raise RuntimeError(f"Only {count} detected images for held-out identity {name}; need >=8")

    # Precompute the exact FaceNet embeddings used by the notebook pipeline.
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    embedding_start = time.perf_counter()
    chunks = []
    with torch.inference_mode():
        for begin in range(0, len(crops), 32):
            chunks.append(resnet(crops[begin:begin + 32].to(device)).detach().cpu().numpy())
            done = min(begin + 32, len(crops))
            print(f"FaceNet embeddings {done}/{len(crops)}", flush=True)
    raw_embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    embeddings = safe_normalize(raw_embeddings)
    embedding_seconds = time.perf_counter() - embedding_start
    crops64 = F.interpolate(crops, size=(64, 64), mode="bilinear", align_corners=False).contiguous()

    guard = load_repository_guard()
    smoke = guard_regression_checks(guard)
    known_to_index = {name: i for i, name in enumerate(known_names)}
    all_seed_results = []
    cnn_training_seconds_by_seed = {}
    svc_training_seconds_by_seed = {}
    embedding_diagnostics = {
        "images_selected_from_lfw_pre_cropped_faces": int(len(selected)),
        "images_preprocessed": int(len(crops)),
        "crop_preprocessing": "resize pre-cropped LFW RGB faces to 160x160; normalize raw pixel values to [-1,1] (the FaceNet MTCNN-output convention)",
        "crop_preprocessing_seconds": round(preprocessing_seconds, 3),
        "facenet_embedding_seconds": round(embedding_seconds, 3),
        "facenet_embedding_throughput_ms_per_image": round(1000 * embedding_seconds / max(1, len(crops)), 4),
        "preprocessed_faces_by_identity": {name: int(np.sum(people == name)) for name in known_names + unknown_names},
        "source_images_per_identity": class_caps,
    }

    for seed in seeds:
        print(f"\n=== Split/training seed {seed} ===", flush=True)
        rng = np.random.default_rng(seed)
        train_idx_list, val_idx_list, test_known_idx_list = [], [], []
        for name in known_names:
            person_idx = np.flatnonzero(people == name).copy()
            rng.shuffle(person_idx)
            count = len(person_idx)
            train_n = int(count * 0.65)
            val_n = int(count * 0.15)
            # Guarantee non-empty fit/validation/test partitions.
            train_n = max(6, min(train_n, count - 5))
            val_n = max(2, min(val_n, count - train_n - 2))
            test_n = count - train_n - val_n
            if test_n < 2:
                raise RuntimeError(f"Invalid split size for {name}: {count}")
            train_idx_list.extend(person_idx[:train_n].tolist())
            val_idx_list.extend(person_idx[train_n:train_n + val_n].tolist())
            test_known_idx_list.extend(person_idx[train_n + val_n:].tolist())
        train_idx = np.asarray(train_idx_list, dtype=int)
        val_idx = np.asarray(val_idx_list, dtype=int)
        test_known_idx = np.asarray(test_known_idx_list, dtype=int)
        unknown_idx = np.flatnonzero(np.isin(people, unknown_names))
        test_idx = np.concatenate([test_known_idx, unknown_idx])
        y_train_names = people[train_idx].tolist()
        y_val_names = people[val_idx].tolist()
        y_test_names = people[test_idx].tolist()

        neuron_ids = {name: f"identity_{i}" for i, name in enumerate(known_names)}
        fake_brain = SimpleNamespace(neurons={
            neuron_ids[name]: SimpleNamespace(soma=SimpleNamespace(metadata={"label": name}))
            for name in known_names
        })
        bank = guard["FEATURES_BY_NEURON"]
        bank.clear()
        for name in known_names:
            rows = embeddings[[j for j in train_idx if people[j] == name]]
            bank[neuron_ids[name]] = [row.astype(np.float32) for row in rows]
        candidates = set(fake_brain.neurons)

        guarded_predictions, plain_predictions = [], []
        guarded_latency, plain_latency = [], []
        for idx in test_idx:
            query = embeddings[idx]
            start = time.perf_counter_ns()
            nid, _ = guard["find_best_in_candidates"](fake_brain, query, candidates)
            guarded_latency.append((time.perf_counter_ns() - start) / 1e6)
            guarded_predictions.append(
                fake_brain.neurons[nid].soma.metadata["label"] if nid is not None else None
            )

            start = time.perf_counter_ns()
            scores = []
            for name in known_names:
                score = class_similarity_score(
                    guard["_normalise_embedding"](query), bank[neuron_ids[name]],
                    int(guard["NEURON_METADATA_TOP_EXEMPLARS"]),
                )
                if score is not None:
                    scores.append((score, name))
            plain_predictions.append(max(scores, key=lambda item: (item[0], item[1]))[1] if scores else None)
            plain_latency.append((time.perf_counter_ns() - start) / 1e6)

        # Baseline A: RBF SVM trained on the same FaceNet embeddings and fit split.
        train_y_idx = np.asarray([known_to_index[name] for name in y_train_names], dtype=int)
        val_y_idx = np.asarray([known_to_index[name] for name in y_val_names], dtype=int)
        test_is_known = np.asarray([name in known_set for name in y_test_names], dtype=bool)
        svc_start = time.perf_counter()
        svc = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                  class_weight="balanced", random_state=seed)
        svc.fit(embeddings[train_idx], train_y_idx)
        svc_training_seconds_by_seed[str(seed)] = round(time.perf_counter() - svc_start, 3)
        svc_val_probs = svc.predict_proba(embeddings[val_idx])
        svc_threshold = float(np.quantile(svc_val_probs.max(axis=1), 0.05))
        svc_preds, svc_latency = [], []
        for idx in test_idx:
            start = time.perf_counter_ns()
            probs = svc.predict_proba(embeddings[idx:idx + 1])[0]
            top = int(np.argmax(probs))
            svc_preds.append(known_names[int(svc.classes_[top])] if float(probs[top]) >= svc_threshold else None)
            svc_latency.append((time.perf_counter_ns() - start) / 1e6)

        # Baseline B: train a CNN from scratch on the exact same aligned face crops.
        cnn_train_y = train_y_idx
        cnn_start = time.perf_counter()
        cnn, cnn_threshold, train_seconds = train_cnn(
            crops64, train_idx, cnn_train_y, val_idx,
            class_count=len(known_names), seed=seed, epochs=args.cnn_epochs, device=device,
        )
        cnn_training_seconds_by_seed[str(seed)] = round(train_seconds, 3)
        cnn_preds, cnn_closed_preds, cnn_latency = [], [], []
        with torch.inference_mode():
            for idx in test_idx:
                sample = crops64[idx:idx + 1].to(device)
                start = time.perf_counter_ns()
                probs = cnn(sample).softmax(dim=1)[0]
                cnn_latency.append((time.perf_counter_ns() - start) / 1e6)
                top_idx = int(torch.argmax(probs).item())
                pred_name = known_names[top_idx]
                cnn_closed_preds.append(pred_name)
                cnn_preds.append(pred_name if float(probs[top_idx].item()) >= cnn_threshold else None)

        # Evaluation uses the same test image indices and the same ground-truth names.
        truth = y_test_names
        all_seed_results.append({
            "seed": seed,
            "counts": {
                "known_train_images": int(len(train_idx)),
                "known_validation_images": int(len(val_idx)),
                "known_test_images": int(len(test_known_idx)),
                "unknown_test_images": int(len(unknown_idx)),
                "test_images_total": int(len(test_idx)),
                "train_images_by_identity": {name: int(np.sum(people[train_idx] == name)) for name in known_names},
                "known_test_images_by_identity": {name: int(np.sum(people[test_known_idx] == name)) for name in known_names},
            },
            "thresholds": {
                "neuron_min_similarity": float(guard["NEURON_METADATA_MIN_SIMILARITY"]),
                "neuron_min_margin": float(guard["NEURON_METADATA_MIN_MARGIN"]),
                "neuron_top_exemplars": int(guard["NEURON_METADATA_TOP_EXEMPLARS"]),
                "svm_confidence_threshold_known_validation_5th_percentile": round(svc_threshold, 6),
                "cnn_confidence_threshold_known_validation_5th_percentile": round(cnn_threshold, 6),
            },
            "training_seconds": {
                "svm": svc_training_seconds_by_seed[str(seed)],
                "cnn": cnn_training_seconds_by_seed[str(seed)],
            },
            "results": {
                "neuron_facenet_metadata_guard": evaluate(guarded_predictions, truth, known_set, guarded_latency),
                "neuron_facenet_same_exemplars_without_guard": evaluate(plain_predictions, truth, known_set, plain_latency),
                "svm_on_same_facenet_embeddings_calibrated_rejection": evaluate(svc_preds, truth, known_set, svc_latency),
                "cnn_on_same_aligned_face_crops_closed_set": evaluate(cnn_closed_preds, truth, known_set, cnn_latency),
                "cnn_on_same_aligned_face_crops_calibrated_rejection": evaluate(cnn_preds, truth, known_set, cnn_latency),
            },
        })
        print(json.dumps({
            "seed": seed,
            **{name: {
                "known_accuracy": result["known_end_to_end_accuracy"],
                "known_coverage": result["known_coverage"],
                "selective_accuracy": result["known_selective_accuracy"],
                "unknown_rejection": result["unknown_rejection_rate"],
            } for name, result in all_seed_results[-1]["results"].items()}
        }, indent=2), flush=True)

    methods = list(all_seed_results[0]["results"])
    metric_names = [
        "known_end_to_end_accuracy", "known_coverage", "known_selective_accuracy",
        "unknown_rejection_rate", "unknown_false_accept_rate", "safe_overall_rate",
    ]
    aggregate = {}
    for method in methods:
        aggregate[method] = {}
        for metric in metric_names:
            values = [row["results"][method][metric] for row in all_seed_results]
            aggregate[method][metric] = {
                "mean": round(statistics.mean(values), 4),
                "std_population": round(statistics.pstdev(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

    report = {
        "status": "EXECUTED",
        "benchmark": "Neuron FaceNet + metadata guard vs CNN and FaceNet-feature baseline",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "face_extractor": "facenet-pytorch InceptionResnetV1(pretrained='vggface2')",
            "face_preprocessing": "LFW funneled subject crops resized to 160x160 and normalized to [-1,1]; MTCNN intentionally skipped because this dataset is already cropped",
            "dataset": "Labeled Faces in the Wild (LFW), scikit-learn fetch_lfw_people, color=True, resize=1.0, min_faces_per_person=20",
        },
        "protocol": {
            "known_identity_count": len(known_names),
            "known_identities": known_names,
            "unknown_identity_count": len(unknown_names),
            "held_out_unknown_identities": unknown_names,
            "identity_selection": "Most frequent LFW identities after min_faces_per_person=20; fixed once before seed-specific splits",
            "max_images_per_known_identity": args.max_known_per_person,
            "max_images_per_unknown_identity": args.max_unknown_per_person,
            "known_split": "Per identity 65% fit / 15% known-only validation calibration / remaining 20% test, split separately per seed",
            "unknown_split": "Selected images of unknown identities are test-only; never used for training, exemplars, or threshold calibration",
            "shared_test_crops": "All compared methods use the same pre-cropped LFW images resized to FaceNet/CNN input sizes; all train/validation/test image indices are shared per seed",
            "neuron_guard_source": "Exact guard functions/constants AST-extracted from Untitled36.ipynb; all identity nodes are in one global candidate region",
            "cnn": f"Small CNN trained from scratch on the aligned training face crops for {args.cnn_epochs} epochs per seed",
            "calibration": "SVM and CNN rejection thresholds are the 5th percentile of maximum class confidence on known-only validation images; no unknown test labels are used for calibration",
            "pretraining_overlap_caveat": "InceptionResnetV1 VGGFace2 weights may have identity overlap with LFW; conclusions apply to this benchmark and checkpoint, not decontaminated face recognition generally.",
        },
        "preprocessing_diagnostics": embedding_diagnostics,
        "guard_regression_checks": smoke,
        "training_seconds_by_seed": {
            "svm": svc_training_seconds_by_seed,
            "cnn": cnn_training_seconds_by_seed,
        },
        "aggregate_mean_std": aggregate,
        "runs": all_seed_results,
        "interpretation_note": (
            "This is an actual FaceNet-based test of the repository's metadata guard decision logic on LFW. "
            "It compares a scratch CNN and an SVM on the same pretrained FaceNet embeddings and matched test crops. "
            "It is not evidence of universal superiority; note the selected identities, pretraining-overlap caveat, "
            "detection coverage, per-seed variation, and rejection/coverage tradeoffs."
        ),
    }
    output = Path(args.json_output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n=== FACE BENCHMARK SUMMARY ===", flush=True)
    print(json.dumps({
        "saved": str(output),
        "status": report["status"],
        "known_identities": known_names,
        "unknown_identities": unknown_names,
        "preprocessing": embedding_diagnostics,
        "aggregate_mean_std": aggregate,
        "guard_regression_checks": smoke,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
