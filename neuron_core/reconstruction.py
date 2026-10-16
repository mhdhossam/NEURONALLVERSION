"""Feedback-driven reconstruction of Neuron's activation and path exploration.

Unlike the candidate-only decision ranker, this policy changes seed activation and
which graph edges are explored first. Feedback updates the policy used by later
queries. It is intentionally small, inspectable, and dependency-free.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping

def _tokens(text: str) -> set[str]:
    return {t.casefold() for t in re.findall(r"[^\\W_]+", text, flags=re.UNICODE)}

def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))

def _cosmetic_overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left))

class ReconstructedNeuron:
    """A policy-controlled Neuron execution loop with online reward feedback.

    Call search(), inspect its trace, then observe() only when an answer has been
    verified. The learned policy is kept separate from NeuronGraph's stored facts.
    """
    VERSION = 1

    def __init__(self, graph: Any, *, learning_rate: float = 0.08,
                 max_weight: float = 3.0):
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive")
        self.graph = graph
        self.learning_rate = float(learning_rate)
        self.max_weight = float(max_weight)
        # Activation policy: lexical match, confidence, prior feedback.
        self.activation_weights = [0.65, 0.20, 0.15]
        # Path policy: edge reliability, query/relation fit, target fit, shortness, evidence.
        self.path_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        self.feedback_count = 0
        self._last_traces: dict[str, dict[str, Any]] = {}

    def _activation_features(self, query: str, nid: str, lexical: float) -> list[float]:
        neuron = self.graph.neurons[nid]
        confidence = float(neuron.confidence)
        prior = _clip((neuron.support - neuron.contradiction) /
                      max(1.0, neuron.support + neuron.contradiction))
        return [_clip(lexical), _clip(confidence), _clip((prior + 1.0) / 2.0)]

    def _edge_features(self, query_tokens: set[str], edge: Any,
                       target_id: str, hop: int) -> list[float]:
        target = self.graph.neurons[target_id]
        rel_tokens = _tokens(edge.relation.replace("_", " "))
        target_tokens = _tokens(" ".join([
            target.name, target.content, str(target.metadata),
            str(self.graph.knowledge[target.knowledge_id].attributes),
        ]))
        relation_fit = _cosmetic_overlap(query_tokens, rel_tokens)
        target_fit = _cosmetic_overlap(query_tokens, target_tokens)
        evidence = (edge.support - edge.contradiction) / max(
            1.0, edge.support + edge.contradiction)
        return [
            _clip(edge.effective_strength),
            _clip(relation_fit),
            _clip(target_fit),
            1.0 / max(1, hop),
            _clip((evidence + 1.0) / 2.0),
        ]

    @staticmethod
    def _dot(weights: list[float], features: list[float]) -> float:
        return sum(w * x for w, x in zip(weights, features))

    def search(self, query: str, *, top_k: int = 5, max_hops: int = 2,
               max_nodes: int = 200, region_id: str | None = None,
               commit_snapshot: bool = False) -> list[dict[str, Any]]:
        if top_k < 1 or max_hops < 0 or max_nodes < 1:
            raise ValueError("top_k/max_nodes must be positive and max_hops non-negative")
        query_tokens = _tokens(query)
        seeds = self.graph.find_candidates(
            query, top_k=max(10, top_k * 4), region_id=region_id)
        if not seeds:
            self._last_traces = {}
            return []

        scores: dict[str, float] = {}
        paths: dict[str, list[str]] = {}
        path_edges: dict[str, list[str]] = {}
        activation_features: dict[str, list[float]] = {}
        edge_feature_traces: dict[str, list[list[float]]] = {}
        frontier: list[tuple[str, float]] = []
        for nid, lexical in seeds:
            features = self._activation_features(query, nid, lexical)
            activation_features[nid] = features
            # The policy directly controls initial activation, not just final order.
            score = _clip(self._dot(self.activation_weights, features))
            if score <= 0:
                continue
            scores[nid] = score
            paths[nid] = [nid]
            path_edges[nid] = []
            edge_feature_traces[nid] = []
            frontier.append((nid, score))

        for hop in range(1, max_hops + 1):
            next_frontier: dict[str, tuple[float, list[str], list[str], list[list[float]]]] = {}
            # Policy chooses edge expansion priority; deterministic tie-breaking keeps runs reproducible.
            actions = []
            for source_id, source_score in frontier:
                for sid in self.graph._outgoing.get(source_id, []):
                    edge = self.graph.synapses[sid]
                    target_id = edge.target_id
                    if target_id in paths[source_id]:
                        continue
                    features = self._edge_features(query_tokens, edge, target_id, hop)
                    policy_value = _clip(self._dot(self.path_weights, features))
                    propagated = source_score * policy_value
                    actions.append((propagated, source_id, sid, target_id, features))
            actions.sort(key=lambda row: (-row[0], row[2]))
            for propagated, source_id, sid, target_id, features in actions:
                if propagated <= 0:
                    continue
                if target_id not in scores and target_id not in next_frontier and len(scores) + len(next_frontier) >= max_nodes:
                    continue
                candidate_score = propagated
                if candidate_score <= scores.get(target_id, 0.0):
                    continue
                edge = self.graph.synapses[sid]
                source_path = paths[source_id]
                source_edge_path = path_edges[source_id]
                source_features = edge_feature_traces[source_id]
                next_frontier[target_id] = (
                    candidate_score, source_path + [target_id],
                    source_edge_path + [sid], source_features + [features])
            if not next_frontier:
                break
            frontier = []
            for nid, (score, path, edges, features) in next_frontier.items():
                if score > scores.get(nid, 0.0):
                    scores[nid] = score
                    paths[nid] = path
                    path_edges[nid] = edges
                    edge_feature_traces[nid] = features
                    frontier.append((nid, score))
            if len(scores) >= max_nodes:
                break

        rows = []
        self._last_traces = {}
        for nid, score in scores.items():
            neuron = self.graph.neurons[nid]
            knowledge = self.graph.knowledge[neuron.knowledge_id]
            row = {
                "neuron_id": nid, "knowledge_id": knowledge.id, "name": knowledge.name,
                "kind": knowledge.kind, "score": round(score, 8),
                "activation": round(score, 8), "lexical_score": next(
                    (lex for seed_id, lex in seeds if seed_id == nid), 0.0),
                "path": [self.graph.neurons[path_id].name for path_id in paths[nid]],
                "hops": len(path_edges[nid]), "attributes": dict(knowledge.attributes),
                "reconstruction_trace": {
                    "seed_features": activation_features.get(paths[nid][0], []),
                    "edge_ids": list(path_edges[nid]),
                    "edge_features": edge_feature_traces[nid],
                },
            }
            rows.append(row)
            self._last_traces[nid] = {
                "query": query,
                "seed_id": paths[nid][0],
                "seed_features": activation_features.get(paths[nid][0], []),
                "edge_features": edge_feature_traces[nid],
                "path": list(paths[nid]),
            }

        rows.sort(key=lambda row: (-row["score"], row["hops"], row["name"].casefold(), row["neuron_id"]))
        if commit_snapshot:
            # Use the graph's own bounded activation method to preserve snapshot semantics.
            # The reconstructed policy's scores remain query-local by default.
            for nid, score in scores.items():
                self.graph.neurons[nid].activation = _clip(score)
                self.graph.neurons[nid].last_active_step = len(path_edges[nid])
        return rows[:top_k]

    def observe(self, query: str, answer_id: str, reward: float) -> None:
        """Update the activation and traversal policy from verified answer feedback."""
        reward = float(reward)
        if not math.isfinite(reward) or not -1.0 <= reward <= 1.0:
            raise ValueError("reward must be finite and between -1 and 1")
        trace = self._last_traces.get(answer_id)
        if trace is None or trace["query"] != query:
            raise ValueError("run search(query) first and provide one of its result IDs")
        lr = self.learning_rate * reward
        seed_features = trace["seed_features"]
        for i, value in enumerate(seed_features):
            self.activation_weights[i] = _clip(
                self.activation_weights[i] + lr * value, -self.max_weight, self.max_weight)
        # Reward is credited to the actual explored path, not every retrieved candidate.
        for features in trace["edge_features"]:
            for i, value in enumerate(features):
                self.path_weights[i] = _clip(
                    self.path_weights[i] + lr * value, -self.max_weight, self.max_weight)
        self.feedback_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "learning_rate": self.learning_rate,
            "max_weight": self.max_weight,
            "activation_weights": list(self.activation_weights),
            "path_weights": list(self.path_weights),
            "feedback_count": self.feedback_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != self.VERSION:
            raise ValueError("incompatible reconstruction policy state")
        activation = [float(v) for v in state["activation_weights"]]
        path = [float(v) for v in state["path_weights"]]
        if len(activation) != 3 or len(path) != 5:
            raise ValueError("invalid reconstruction policy dimensions")
        if not all(math.isfinite(v) for v in activation + path):
            raise ValueError("policy weights must be finite")
        self.learning_rate = float(state["learning_rate"])
        self.max_weight = float(state["max_weight"])
        self.activation_weights = activation
        self.path_weights = path
        self.feedback_count = int(state["feedback_count"])
