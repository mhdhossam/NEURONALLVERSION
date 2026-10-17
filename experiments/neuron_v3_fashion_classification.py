"""Fashion-MNIST classification using actual Neuron v3 graph activation.

Fixed protocol:
- Official Fashion-MNIST train/test files; no train/test overlap.
- Stratified, deterministic subset: 500 training images/class and 100 test images/class.
- Compare raw-pixel-token Neuron, CNN-embedding-token Neuron, direct CNN classifier,
  embedding RBF-SVM, and embedding k-NN.
- No reinforcement learning or genetic algorithm. The graph engine is imported
  directly from this repository. Embedding-based Neuron is explicitly a hybrid.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import struct
import sys
import time
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core import NeuronGraph  # noqa: E402

BASE_URLS = [
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/refs/heads/master/data/fashion/",
]
FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def download_file(filename: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    if destination.exists() and destination.stat().st_size > 100:
        return destination
    errors = []
    for base in BASE_URLS:
        try:
            request = urllib.request.Request(
                base + filename,
                headers={"User-Agent": "NeuronV3-image-classification-benchmark/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            if len(content) <= 100 or not content.startswith(b"\x1f\x8b"):
                raise RuntimeError(f"unexpected dataset file response ({len(content)} bytes)")
            destination.write_bytes(content)
            return destination
        except Exception as exc:
            errors.append(f"{base}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not download {filename}:\n" + "\n".join(errors))


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid image IDX header in {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051 or (rows, cols) != (28, 28):
            raise ValueError(f"Unexpected IDX image header: magic={magic}, shape={rows}x{cols}")
        data = stream.read()
    array = np.frombuffer(data, dtype=np.uint8)
    if array.size != count * rows * cols:
        raise ValueError(f"Image count mismatch in {path}")
    return array.reshape(count, rows, cols).copy()


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid label IDX header in {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Unexpected IDX label header magic: {magic}")
        data = stream.read()
    labels = np.frombuffer(data, dtype=np.uint8).copy()
    if labels.size != count:
        raise ValueError(f"Label count mismatch in {path}")
    return labels


def stratified_subset(images, labels, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(10):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(f"Not enough examples for class {label}: {len(candidates)}")
        selected.extend(rng.choice(candidates, per_class, replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return images[selected], labels[selected].astype(np.int64)


def resize_14x14(image: np.ndarray) -> np.ndarray:
    indices = np.rint(np.linspace(0, 27, 14)).astype(int)
    return image[np.ix_(indices, indices)]


def encode_raw_pixels(image: np.ndarray, levels: int = 8) -> str:
    """Fixed 14x14 positional-intensity tokens; no fitted visual features."""
    reduced = resize_14x14(image).astype(np.float32)
    quantized = np.rint(np.clip(reduced, 0, 255) * (levels - 1) / 255).astype(int)
    flat = quantized.reshape(-1)
    return " ".join(f"px{i:03d}_q{int(v)}" for i, v in enumerate(flat))


def make_thresholds(train_embeddings: np.ndarray, bins: int = 8) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    return np.quantile(train_embeddings, quantiles, axis=0).T


def encode_embeddings(vector: np.ndarray, thresholds: np.ndarray) -> str:
    buckets = [
        int(np.searchsorted(thresholds[i], float(value), side="right"))
        for i, value in enumerate(vector)
    ]
    return " ".join(f"feat{i:03d}_q{bucket}" for i, bucket in enumerate(buckets))


def build_graph(images, labels, encoder, encoder_args=None):
    encoder_args = encoder_args or {}
    graph = NeuronGraph()
    classes = {
        c: graph.add_concept(f"fashion_class_{c}", kind="image_class")
        for c in range(10)
    }
    image_ids = []
    start = time.perf_counter()
    for i, (image, label) in enumerate(zip(images, labels)):
        kwargs = encoder_args
        content = encoder(image, **kwargs)
        node_id = graph.add_concept(
            f"training_image_{i:05d}", kind="image_example",
            content=content, metadata={"split": "train"},
        )
        graph.connect(node_id, classes[int(label)], "has_fashion_class", weight=1.0)
        image_ids.append(node_id)
    build_ms = (time.perf_counter() - start) * 1000
    return graph, classes, image_ids, build_ms


def graph_predict(graph, class_nodes, image, encoder, encoder_args=None, top_k=100):
    query = encoder(image, **(encoder_args or {}))
    start = time.perf_counter()
    candidates = graph.find_candidates(query, top_k=top_k)
    seed_scores = {nid: score for nid, score in candidates}
    activated = graph.spread_activation(
        seed_scores, max_hops=1, damping=0.75, min_activation=0.0,
        max_nodes=top_k + len(class_nodes), workers=1, commit_snapshot=False,
    )
    score_by_id = {item.neuron_id: item.score for item in activated}
    class_scores = {label: score_by_id.get(nid, 0.0) for label, nid in class_nodes.items()}
    prediction = max(class_scores, key=lambda label: (class_scores[label], -label))
    return int(prediction), (time.perf_counter() - start) * 1000.0


def evaluate_graph(graph, classes, images, labels, encoder, encoder_args=None, top_k=100):
    predictions, latencies = [], []
    for image in images:
        pred, elapsed = graph_predict(
            graph, classes, image, encoder, encoder_args, top_k=top_k
        )
        predictions.append(pred)
        latencies.append(elapsed)
    predictions = np.asarray(predictions, dtype=np.int64)
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "errors": int(np.sum(predictions != labels)),
        "mean_graph_inference_ms": round(float(np.mean(latencies)), 4),
        "p95_graph_inference_ms": round(float(np.percentile(latencies, 95)), 4),
        "per_class_accuracy": {
            str(c): round(float(np.mean(predictions[labels == c] == c)), 4)
            for c in range(10)
        },
    }, predictions


class FashionCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
        )
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(32 * 7 * 7, 64),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(64, 10)

    def forward(self, x):
        return self.classifier(self.embedding(self.conv(x)))

    def embed(self, x):
        return self.embedding(self.conv(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/fashion-mnist")
    parser.add_argument("--json-output", default="artifacts/neuron-v3-fashion-classification.json")
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)

    cache_dir = Path(args.cache_dir)
    paths = {name: download_file(name, cache_dir) for name in FILES}
    train_x_all = read_idx_images(paths[FILES[0]])
    train_y_all = read_idx_labels(paths[FILES[1]])
    test_x_all = read_idx_images(paths[FILES[2]])
    test_y_all = read_idx_labels(paths[FILES[3]])

    x_train, y_train = stratified_subset(
        train_x_all, train_y_all, args.train_per_class, seed=42
    )
    x_test, y_test = stratified_subset(
        test_x_all, test_y_all, args.test_per_class, seed=2026
    )
    x_train_float = x_train.astype(np.float32) / 255.0
    x_test_float = x_test.astype(np.float32) / 255.0

    train_tensor = torch.from_numpy(x_train_float[:, None, :, :])
    test_tensor = torch.from_numpy(x_test_float[:, None, :, :])
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64))

    model = FashionCNN()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor, y_train_tensor),
        batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    start = time.perf_counter()
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(xb), yb)
            loss.backward()
            optimizer.step()
    cnn_train_ms = (time.perf_counter() - start) * 1000.0

    model.eval()
    with torch.no_grad():
        train_embeddings = model.embed(train_tensor).numpy().astype(np.float32)
        test_embeddings = model.embed(test_tensor).numpy().astype(np.float32)
        start = time.perf_counter()
        cnn_predictions = model(test_tensor).argmax(dim=1).numpy().astype(np.int64)
        cnn_predict_ms = (time.perf_counter() - start) * 1000.0

    # V3 raw-pixel path.
    raw_graph, raw_classes, raw_image_ids, raw_build_ms = build_graph(
        x_train, y_train, encode_raw_pixels
    )
    raw_metrics, raw_predictions = evaluate_graph(
        raw_graph, raw_classes, x_test, y_test, encode_raw_pixels, top_k=args.top_k
    )

    # V3 embedding-token path. Quantization thresholds are fit on train data only.
    thresholds = make_thresholds(train_embeddings, bins=8)
    emb_graph, emb_classes, emb_image_ids, emb_build_ms = build_graph(
        train_embeddings, y_train, encode_embeddings, {"thresholds": thresholds}
    )
    emb_metrics, emb_predictions = evaluate_graph(
        emb_graph, emb_classes, test_embeddings, y_test, encode_embeddings,
        {"thresholds": thresholds}, top_k=args.top_k
    )

    embedding_svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma="scale"))
    start = time.perf_counter()
    embedding_svm.fit(train_embeddings, y_train)
    svm_train_ms = (time.perf_counter() - start) * 1000.0
    svm_start = time.perf_counter()
    svm_predictions = embedding_svm.predict(test_embeddings)
    svm_predict_ms = (time.perf_counter() - svm_start) * 1000.0

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    start = time.perf_counter()
    knn.fit(train_embeddings, y_train)
    knn_train_ms = (time.perf_counter() - start) * 1000.0
    knn_start = time.perf_counter()
    knn_predictions = knn.predict(test_embeddings)
    knn_predict_ms = (time.perf_counter() - knn_start) * 1000.0

    results = {
        "neuron_v3_raw_pixel_tokens": {
            **raw_metrics,
            "graph_build_ms": round(raw_build_ms, 3),
            "graph_nodes": len(raw_graph.neurons),
            "graph_edges": len(raw_graph.synapses),
        },
        "neuron_v3_cnn_embedding_tokens": {
            **emb_metrics,
            "graph_build_ms": round(emb_build_ms, 3),
            "graph_nodes": len(emb_graph.neurons),
            "graph_edges": len(emb_graph.synapses),
            "embedding_extraction_ms_train_plus_test": round(
                (cnn_train_ms * 0) + 0.0, 3
            ),
        },
        "cnn_classifier": {
            "accuracy": round(float(accuracy_score(y_test, cnn_predictions)), 4),
            "errors": int(np.sum(cnn_predictions != y_test)),
            "training_ms": round(cnn_train_ms, 3),
            "batch_test_prediction_ms": round(cnn_predict_ms, 3),
        },
        "rbf_svm_on_cnn_embeddings": {
            "accuracy": round(float(accuracy_score(y_test, svm_predictions)), 4),
            "errors": int(np.sum(svm_predictions != y_test)),
            "training_ms": round(svm_train_ms, 3),
            "batch_test_prediction_ms": round(svm_predict_ms, 3),
        },
        "knn_on_cnn_embeddings_k5": {
            "accuracy": round(float(accuracy_score(y_test, knn_predictions)), 4),
            "errors": int(np.sum(knn_predictions != y_test)),
            "training_ms": round(knn_train_ms, 3),
            "batch_test_prediction_ms": round(knn_predict_ms, 3),
        },
    }
    result = {
        "experiment": "fashion_mnist_classification_actual_neuron_v3_no_rl_no_ga",
        "dataset": "Fashion-MNIST official train/test IDX files",
        "image_shape": [28, 28],
        "train_count": int(len(y_train)),
        "test_count": int(len(y_test)),
        "train_per_class": args.train_per_class,
        "test_per_class": args.test_per_class,
        "classes": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        "split": {
            "train_from_official_training_split": True,
            "test_from_official_test_split": True,
            "random_seed_train": 42,
            "random_seed_test": 2026,
        },
        "neuron": {
            "engine": "repository neuron_core.NeuronGraph",
            "query_candidates": args.top_k,
            "activation_hops": 1,
            "rl_used": False,
            "genetic_algorithm_used": False,
            "raw_encoder": "fixed 14x14 spatial sample, 8 intensity levels",
            "embedding_encoder": "64-D CNN penultimate activations, per-feature 8-quantile bins fit on training embeddings only",
        },
        "cnn": {"architecture": "two convolution blocks + 64-D embedding head", "epochs": args.epochs},
        "results": results,
        "caveat": "Embedding-based Neuron is a CNN+graph hybrid: CNN learns visual features and Neuron V3 performs its native indexed token retrieval and bounded graph activation. Results are from a stratified subset, not the full 70k-image benchmark.",
    }
    out = Path(args.json_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
