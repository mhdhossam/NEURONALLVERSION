"""Experimental sparse graph-native decoder Transformer operations.

All learned linear parameters are stored as directed weighted edges. Runtime
forward passes iterate those edges; no dense weight matrices or neural framework
are used. This is a research prototype, not a pretrained Qwen replacement.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class WeightedEdge:
    source: int
    target: int
    weight: float


class GraphLinear:
    """Sparse bipartite computation graph implementing y = W x + b."""

    def __init__(self, in_features: int, out_features: int,
                 edges: Iterable[WeightedEdge], bias: Sequence[float] | None = None):
        if in_features < 1 or out_features < 1:
            raise ValueError("feature dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.edges = tuple(edges)
        self.bias = tuple(bias or (0.0,) * out_features)
        if len(self.bias) != out_features:
            raise ValueError("bias dimension mismatch")
        for edge in self.edges:
            if not (0 <= edge.source < in_features and 0 <= edge.target < out_features):
                raise ValueError("edge endpoint outside layer dimensions")

    @classmethod
    def random_sparse(cls, in_features: int, out_features: int, fan_in: int,
                      seed: int = 0, use_bias: bool = True) -> "GraphLinear":
        """Create a reproducible sparse layer with fan_in edges per output."""
        if fan_in < 1:
            raise ValueError("fan_in must be positive")
        rng = random.Random(seed)
        count = min(fan_in, in_features)
        scale = 1.0 / math.sqrt(count)
        edges = []
        for target in range(out_features):
            sources = rng.sample(range(in_features), count)
            edges.extend(WeightedEdge(source, target, rng.uniform(-scale, scale))
                         for source in sources)
        bias = [rng.uniform(-scale, scale) for _ in range(out_features)] if use_bias else None
        return cls(in_features, out_features, edges, bias)

    def forward(self, x: Sequence[float]) -> list[float]:
        if len(x) != self.in_features:
            raise ValueError("input feature dimension mismatch")
        out = list(self.bias)
        for edge in self.edges:
            out[edge.target] += x[edge.source] * edge.weight
        return out

    @property
    def parameter_count(self) -> int:
        return len(self.edges) + len(self.bias)


def rms_norm(x: Sequence[float], eps: float = 1e-6) -> list[float]:
    if not x:
        return []
    scale = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * scale for v in x]


def apply_rope(x: Sequence[float], position: int, theta: float = 10000.0) -> list[float]:
    """Apply rotary position encoding to adjacent feature pairs."""
    out = list(x)
    for i in range(0, len(out) - 1, 2):
        frequency = theta ** (-i / max(1, len(out)))
        angle = position * frequency
        c, s = math.cos(angle), math.sin(angle)
        a, b = out[i], out[i + 1]
        out[i], out[i + 1] = a * c - b * s, a * s + b * c
    return out


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


class GraphCausalAttention:
    """Dynamic weighted token graph; edges define which past tokens may attend."""

    def __init__(self, head_dim: int, window: int | None = None):
        self.head_dim = head_dim
        self.window = window

    def forward(self, queries: Sequence[Sequence[float]],
                keys: Sequence[Sequence[float]],
                values: Sequence[Sequence[float]]) -> tuple[list[list[float]], int]:
        if not (len(queries) == len(keys) == len(values)):
            raise ValueError("Q/K/V token counts must match")
        outputs = []
        edge_count = 0
        scale = 1.0 / math.sqrt(self.head_dim)
        for i, query in enumerate(queries):
            start = 0 if self.window is None else max(0, i - self.window + 1)
            token_ids = list(range(start, i + 1))
            scores = [
                sum(query[d] * keys[j][d] for d in range(self.head_dim)) * scale
                for j in token_ids
            ]
            weights = _softmax(scores)
            out = [0.0] * self.head_dim
            for weight, j in zip(weights, token_ids):
                for d in range(self.head_dim):
                    out[d] += weight * values[j][d]
            outputs.append(out)
            edge_count += len(token_ids)
        return outputs, edge_count


class GraphTransformerBlock:
    """Single-head pre-norm decoder block built entirely from sparse graph layers."""

    def __init__(self, hidden_size: int, intermediate_size: int, *,
                 projection_fan_in: int, attention_window: int | None = None,
                 seed: int = 0):
        h, m = hidden_size, intermediate_size
        self.q = GraphLinear.random_sparse(h, h, projection_fan_in, seed + 1)
        self.k = GraphLinear.random_sparse(h, h, projection_fan_in, seed + 2)
        self.v = GraphLinear.random_sparse(h, h, projection_fan_in, seed + 3)
        self.o = GraphLinear.random_sparse(h, h, projection_fan_in, seed + 4)
        self.gate = GraphLinear.random_sparse(h, m, projection_fan_in, seed + 5)
        self.up = GraphLinear.random_sparse(h, m, projection_fan_in, seed + 6)
        self.down = GraphLinear.random_sparse(m, h, min(projection_fan_in, m), seed + 7)
        self.attention = GraphCausalAttention(h, attention_window)
        self.hidden_size = h
        self.intermediate_size = m

    def forward(self, tokens: Sequence[Sequence[float]]) -> dict:
        if any(len(token) != self.hidden_size for token in tokens):
            raise ValueError("token hidden size mismatch")
        normed = [rms_norm(token) for token in tokens]
        q = [apply_rope(self.q.forward(x), i) for i, x in enumerate(normed)]
        k = [apply_rope(self.k.forward(x), i) for i, x in enumerate(normed)]
        v = [self.v.forward(x) for x in normed]
        attended, attention_edges = self.attention.forward(q, k, v)
        projected = [self.o.forward(row) for row in attended]
        residual = [
            [tokens[i][d] + projected[i][d] for d in range(self.hidden_size)]
            for i in range(len(tokens))
        ]
        result = []
        for token in residual:
            x = rms_norm(token)
            gate = self.gate.forward(x)
            up = self.up.forward(x)
            # SwiGLU: silu(gate) * up
            hidden = [(g / (1.0 + math.exp(-max(-60.0, min(60.0, g))))) * u
                      for g, u in zip(gate, up)]
            down = self.down.forward(hidden)
            result.append([token[d] + down[d] for d in range(self.hidden_size)])
        return {
            "tokens": result,
            "attention_edges": attention_edges,
            "parameter_edges": sum(len(layer.edges) for layer in
                                   (self.q, self.k, self.v, self.o,
                                    self.gate, self.up, self.down)),
            "parameter_count": sum(layer.parameter_count for layer in
                                   (self.q, self.k, self.v, self.o,
                                    self.gate, self.up, self.down)),
        }
