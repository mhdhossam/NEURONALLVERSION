"""Compare Neuron v3 using raw pixel tokens vs CNN feature-embedding tokens.

This is a no-RL/no-genetic hybrid experiment. The CNN is trained on the training
split, and its 64-D penultimate-layer activations are quantized into per-feature
tokens. Both Neuron variants use the real NeuronGraph retrieval and activation
methods. The embedding result is hybrid, not vision learned by Neuron alone.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core import NeuronGraph  # noqa: E402


def encode_pixels(image: np.ndarray, levels: int = 9) -> str:
    flat = np.asarray(image, dtype=np.float32).reshape(-1)
    q = np.rint(np.clip(flat, 0, 16) * (levels - 1) / 16).astype(int)
    return " ".join(f"px{i:02d}_v{v:02d}" for i, v in enumerate(q))


def make_embedding_thresholds(train_embeddings: np.ndarray, bins: int = 8) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    # One quantile boundary array per feature; thresholds are fitted on train only.
    return np.quantile(train_embeddings, quantiles, axis=0).T


def encode_embedding(vector: np.ndarray, thresholds: np.ndarray) -> str:
    bucket = np.array([
        np.searchsorted(thresholds[i], float(value), side="right")
        for i, value in enumerate(vector)
    ], dtype=int)
    return " ".join(f"f{i:02d}_q{int(q):02d}" for i, q in enumerate(bucket))


def build_graph(x_train, y_train, encoder, encoder_args=None):
    encoder_args = encoder_args or {}
    graph = NeuronGraph()
    class_nodes = {
        c: graph.add_concept(f"digit_class_{c}", kind="image_class")
        for c in range(10)
    }
    ids = []
    started = time.perf_counter()
    for i, (x, y) in enumerate(zip(x_train, y_train)):
        content = encoder(x, **encoder_args)
        nid = graph.add_concept(
            f"training_image_{i:05d}", kind="image_example", content=content,
            metadata={"split": "train"},
        )
        graph.connect(nid, class_nodes[int(y)], "has_digit_class", weight=1.0)
        ids.append(nid)
    return graph, class_nodes, ids, (time.perf_counter() - started) * 1000.0


def neuron_predict(graph, class_nodes, image, train_n, encoder, encoder_args=None):
    query = encoder(image, **(encoder_args or {}))
    start = time.perf_counter()
    candidates = graph.find_candidates(query, top_k=train_n)
    seed_scores = {nid: score for nid, score in candidates}
    activated = graph.spread_activation(
        seed_scores, max_hops=1, damping=0.75, min_activation=0.0,
        max_nodes=train_n + 10, workers=1, commit_snapshot=False,
    )
    scores = {item.neuron_id: item.score for item in activated}
    class_scores = {c: scores.get(nid, 0.0) for c, nid in class_nodes.items()}
    pred = max(class_scores, key=lambda c: (class_scores[c], -c))
    return int(pred), (time.perf_counter() - start) * 1000.0


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 4 * 4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.layers(x)

    def embed(self, x):
        return self.layers[:-1](x)


def accuracy(y, p):
    return round(float(np.mean(np.asarray(y) == np.asarray(p))), 4)


def run_neuron(graph, classes, images, encoder, encoder_args=None):
    predictions, latencies = [], []
    for image in images:
        pred, ms = neuron_predict(
            graph, classes, image, len(graph.neurons) - len(classes),
            encoder, encoder_args,
        )
        predictions.append(pred)
        latencies.append(ms)
    return np.asarray(predictions), latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", default="artifacts/neuron-v3-image-embedding-results.json")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(1)
    data = load_digits()
    x = data.images.astype(np.float32)
    y = data.target.astype(np.int64)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    xtr, ytr = x[train_idx], y[train_idx]
    xte, yte = x[test_idx], y[test_idx]

    # Train a small CNN only on the fixed training split.
    model = SmallCNN()
    xtr_t = torch.from_numpy((xtr / 16.0)[:, None, :, :])
    ytr_t = torch.from_numpy(ytr)
    generator = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xtr_t, ytr_t),
        batch_size=64, shuffle=True, generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    start = time.perf_counter()
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    cnn_train_ms = (time.perf_counter() - start) * 1000.0
    model.eval()
    with torch.no_grad():
        emb_train = model.embed(xtr_t).numpy().astype(np.float32)
        emb_test = model.embed(torch.from_numpy((xte / 16.0)[:, None, :, :])).numpy().astype(np.float32)
        cnn_logits = model(torch.from_numpy((xte / 16.0)[:, None, :, :]))
        cnn_pred = cnn_logits.argmax(dim=1).numpy().astype(np.int64)

    # V3 baseline 1: fixed raw pixel-position tokens.
    graph_pixels, classes_pixels, ids_pixels, pixels_build_ms = build_graph(
        xtr, ytr, encode_pixels
    )
    pred_pixels, pixel_lat = run_neuron(
        graph_pixels, classes_pixels, xte, encode_pixels
    )

    # V3 baseline 2: same graph engine but tokenized CNN feature embeddings.
    thresholds = make_embedding_thresholds(emb_train, bins=8)
    graph_emb, classes_emb, ids_emb, emb_build_ms = build_graph(
        emb_train, ytr, encode_embedding, {"thresholds": thresholds}
    )
    pred_emb, emb_lat = run_neuron(
        graph_emb, classes_emb, emb_test, encode_embedding, {"thresholds": thresholds}
    )

    # Baselines to distinguish representation benefit from graph benefit.
    raw_svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma="scale"))
    start = time.perf_counter()
    raw_svm.fit((xtr / 16.0).reshape(len(xtr), -1), ytr)
    raw_svm_train_ms = (time.perf_counter() - start) * 1000.0
    raw_svm_pred = raw_svm.predict((xte / 16.0).reshape(len(xte), -1))

    emb_svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma="scale"))
    start = time.perf_counter()
    emb_svm.fit(emb_train, ytr)
    emb_svm_train_ms = (time.perf_counter() - start) * 1000.0
    emb_svm_pred = emb_svm.predict(emb_test)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    start = time.perf_counter()
    knn.fit(emb_train, ytr)
    knn_train_ms = (time.perf_counter() - start) * 1000.0
    knn_pred = knn.predict(emb_test)

    results = {
        "neuron_v3_raw_pixel_tokens": {
            "accuracy": accuracy(yte, pred_pixels),
            "errors": int(np.sum(pred_pixels != yte)),
            "mean_inference_ms": round(float(np.mean(pixel_lat)), 4),
            "p95_inference_ms": round(float(np.percentile(pixel_lat, 95)), 4),
            "graph_nodes": len(graph_pixels.neurons),
            "graph_edges": len(graph_pixels.synapses),
            "graph_build_ms": round(pixels_build_ms, 3),
        },
        "neuron_v3_cnn_embedding_tokens": {
            "accuracy": accuracy(yte, pred_emb),
            "errors": int(np.sum(pred_emb != yte)),
            "mean_inference_ms": round(float(np.mean(emb_lat)), 4),
            "p95_inference_ms": round(float(np.percentile(emb_lat, 95)), 4),
            "graph_nodes": len(graph_emb.neurons),
            "graph_edges": len(graph_emb.synapses),
            "graph_build_ms": round(emb_build_ms, 3),
        },
        "cnn_classifier": {
            "accuracy": accuracy(yte, cnn_pred),
            "errors": int(np.sum(cnn_pred != yte)),
            "train_ms": round(cnn_train_ms, 3),
        },
        "rbf_svm_raw_pixels": {
            "accuracy": accuracy(yte, raw_svm_pred),
            "errors": int(np.sum(raw_svm_pred != yte)),
            "train_ms": round(raw_svm_train_ms, 3),
        },
        "rbf_svm_cnn_embeddings": {
            "accuracy": accuracy(yte, emb_svm_pred),
            "errors": int(np.sum(emb_svm_pred != yte)),
            "train_ms": round(emb_svm_train_ms, 3),
        },
        "knn_cnn_embeddings_k5": {
            "accuracy": accuracy(yte, knn_pred),
            "errors": int(np.sum(knn_pred != yte)),
            "train_ms": round(knn_train_ms, 3),
        },
    }
    result = {
        "experiment": "neuron_v3_raw_pixels_vs_cnn_embeddings_no_rl_no_genetic_algorithm",
        "dataset": "sklearn.datasets.load_digits",
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "shape": [8, 8],
        "split": {"random_state": 42, "stratified": True, "test_size": 0.2},
        "cnn_encoder": "64-D penultimate-layer supervised feature activations",
        "embedding_quantization": "per-feature 8-quantile bins fitted on train embeddings only",
        "neuron_engine": "repository neuron_core.NeuronGraph.find_candidates + spread_activation",
        "rl_used": False,
        "genetic_algorithm_used": False,
        "embedding_caveat": "The embedding variants are a hybrid: the CNN learns visual features; Neuron v3 performs indexed token retrieval and graph activation over those features.",
        "results": results,
    }
    out = Path(args.json_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
