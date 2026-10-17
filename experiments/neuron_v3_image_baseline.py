"""No-RL/no-GA image experiment through the real Neuron v3 core.

A deterministic 8x8 grayscale image encoder creates position/intensity tokens.
NeuronGraph's own inverted lexical index retrieves training-image neurons;
the existing graph activation engine propagates candidate scores to digit-class
neurons. There are no learned Neuron weights, reward updates, or genetic search.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core import NeuronGraph  # noqa: E402


def encode_image(image: np.ndarray, levels: int = 9) -> str:
    """Encode each image pixel as a location-sensitive, quantized token."""
    flat = np.asarray(image, dtype=np.float32).reshape(-1)
    # sklearn's 8x8 digits are grayscale values in [0, 16].
    quantized = np.rint(np.clip(flat, 0, 16) * (levels - 1) / 16).astype(int)
    return " ".join(
        f"px{index:02d}_v{value:02d}" for index, value in enumerate(quantized)
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        str(label): round(float(np.mean(y_pred[y_true == label] == label)), 4)
        for label in sorted(np.unique(y_true))
    }


def build_neuron_graph(x_train: np.ndarray, y_train: np.ndarray):
    graph = NeuronGraph()
    label_ids = {
        label: graph.add_concept(f"digit_class_{label}", kind="image_class")
        for label in range(10)
    }
    image_ids: list[str] = []
    image_labels: dict[str, int] = {}
    started = time.perf_counter()
    for i, (image, label) in enumerate(zip(x_train, y_train)):
        node_id = graph.add_concept(
            f"training_image_{i:05d}",
            kind="image_example",
            content=encode_image(image),
            metadata={"split": "train"},
        )
        # Use the actual V3 typed graph relation; no evidence/feedback is added.
        graph.connect(node_id, label_ids[int(label)], "has_digit_class", weight=1.0)
        image_ids.append(node_id)
        image_labels[node_id] = int(label)
    build_ms = (time.perf_counter() - started) * 1000.0
    return graph, label_ids, image_ids, image_labels, build_ms


def predict_with_neuron(
    graph: NeuronGraph, label_ids: dict[int, str], image: np.ndarray,
    train_image_count: int
) -> tuple[int, float]:
    query = encode_image(image)
    started = time.perf_counter()
    # Get all indexed candidate images so class propagation considers the
    # complete train split, rather than cherry-picking a hand-tuned K.
    candidates = graph.find_candidates(query, top_k=train_image_count)
    seed_scores = {node_id: score for node_id, score in candidates}
    activated = graph.spread_activation(
        seed_scores,
        max_hops=1,
        damping=0.75,
        min_activation=0.0,
        max_nodes=train_image_count + 10,
        workers=1,
        commit_snapshot=False,
    )
    activated_by_id = {item.neuron_id: item.score for item in activated}
    # V3 merges incoming activation by maximum, so each class is scored by its
    # strongest matching training example. This is the engine's native rule,
    # not a learned or externally weighted class vote.
    label_scores = {
        label: activated_by_id.get(node_id, 0.0)
        for label, node_id in label_ids.items()
    }
    pred = max(label_scores, key=lambda label: (label_scores[label], -label))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return int(pred), elapsed_ms


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 4 * 4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.features(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", default="neuron-v3-image-baseline.json")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    digits = load_digits()
    x_all = digits.images.astype(np.float32)
    y_all = digits.target.astype(np.int64)
    train_idx, test_idx = train_test_split(
        np.arange(len(y_all)),
        test_size=0.2,
        random_state=42,
        stratify=y_all,
    )
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_test, y_test = x_all[test_idx], y_all[test_idx]

    # Actual V3 engine: fixed pixel-token encoder -> inverted-index candidate
    # retrieval -> one-hop bounded graph activation to class nodes.
    graph, label_ids, image_ids, _, graph_build_ms = build_neuron_graph(x_train, y_train)
    neuron_preds: list[int] = []
    neuron_latencies_ms: list[float] = []
    for image in x_test:
        pred, elapsed = predict_with_neuron(graph, label_ids, image, len(image_ids))
        neuron_preds.append(pred)
        neuron_latencies_ms.append(elapsed)
    neuron_preds_arr = np.asarray(neuron_preds, dtype=np.int64)

    # Same train/test indices for SVM.
    x_train_flat = (x_train / 16.0).reshape(len(x_train), -1)
    x_test_flat = (x_test / 16.0).reshape(len(x_test), -1)
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma="scale"))
    started = time.perf_counter()
    svm.fit(x_train_flat, y_train)
    svm_train_ms = (time.perf_counter() - started) * 1000.0
    svm_preds = []
    svm_latencies_ms = []
    for row in x_test_flat:
        started = time.perf_counter()
        svm_preds.append(int(svm.predict(row.reshape(1, -1))[0]))
        svm_latencies_ms.append((time.perf_counter() - started) * 1000.0)
    svm_preds_arr = np.asarray(svm_preds, dtype=np.int64)

    # Small CNN trained from scratch, deterministic CPU path, no pretrained weights.
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = SmallCNN()
    xtr = torch.from_numpy((x_train / 16.0)[:, None, :, :])
    ytr = torch.from_numpy(y_train)
    generator = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xtr, ytr),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    started = time.perf_counter()
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    cnn_train_ms = (time.perf_counter() - started) * 1000.0

    cnn_preds = []
    cnn_latencies_ms = []
    model.eval()
    with torch.no_grad():
        for image in x_test:
            inp = torch.from_numpy((image / 16.0).astype(np.float32)[None, None, :, :])
            started = time.perf_counter()
            pred = int(model(inp).argmax(dim=1).item())
            cnn_latencies_ms.append((time.perf_counter() - started) * 1000.0)
            cnn_preds.append(pred)
    cnn_preds_arr = np.asarray(cnn_preds, dtype=np.int64)

    n_test = int(len(y_test))
    models = {
        "neuron_v3_graph_baseline": {
            "accuracy": round(accuracy(y_test, neuron_preds_arr), 4),
            "errors": int(np.sum(y_test != neuron_preds_arr)),
            "mean_single_image_inference_ms": round(float(np.mean(neuron_latencies_ms)), 4),
            "p95_single_image_inference_ms": round(float(np.percentile(neuron_latencies_ms, 95)), 4),
            "per_class_accuracy": per_class_accuracy(y_test, neuron_preds_arr),
        },
        "cnn": {
            "accuracy": round(accuracy(y_test, cnn_preds_arr), 4),
            "errors": int(np.sum(y_test != cnn_preds_arr)),
            "train_ms": round(cnn_train_ms, 3),
            "mean_single_image_inference_ms": round(float(np.mean(cnn_latencies_ms)), 4),
            "p95_single_image_inference_ms": round(float(np.percentile(cnn_latencies_ms, 95)), 4),
            "per_class_accuracy": per_class_accuracy(y_test, cnn_preds_arr),
        },
        "rbf_svm": {
            "accuracy": round(accuracy(y_test, svm_preds_arr), 4),
            "errors": int(np.sum(y_test != svm_preds_arr)),
            "train_ms": round(svm_train_ms, 3),
            "mean_single_image_inference_ms": round(float(np.mean(svm_latencies_ms)), 4),
            "p95_single_image_inference_ms": round(float(np.percentile(svm_latencies_ms, 95)), 4),
            "per_class_accuracy": per_class_accuracy(y_test, svm_preds_arr),
        },
    }
    result = {
        "experiment": "neuron_v3_image_classification_no_rl_no_genetic_algorithm",
        "dataset": "sklearn.datasets.load_digits",
        "image_shape": [8, 8],
        "intensity_range": [0, 16],
        "train_count": int(len(train_idx)),
        "test_count": n_test,
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "neuron_v3": {
            "engine": "repository neuron_core.NeuronGraph",
            "encoder": "fixed pixel-position/intensity tokens, 9 quantization levels",
            "decision": "max candidate similarity propagated one hop to class neurons",
            "rl_used": False,
            "genetic_algorithm_used": False,
            "training_ms": round(graph_build_ms, 3),
            "neurons": len(graph.neurons),
            "synapses": len(graph.synapses),
        },
        "cnn": {"architecture": "small 8x8 CNN trained from scratch", "epochs": args.epochs, "rl_used": False},
        "svm": {"kernel": "RBF", "C": 5, "gamma": "scale"},
        "results": models,
        "caveat": (
            "This tests V3's actual lexical candidate index and bounded graph activation, "
            "but image perception is supplied by a fixed hand-designed encoder. It is not "
            "evidence that the general-purpose V3 core independently learns visual features."
        ),
    }
    out = Path(args.json_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
