"""Observability helpers for Neuron's non-decisional processing state.

This module exposes the intermediate state produced by lexical seeding,
relation-phrase matching, and graph activation. It deliberately does not rank
or select a final answer; applications can inspect the trace and apply their
own interpretation or decision policy.
"""
from __future__ import annotations

from typing import Any

from .engine import NeuronGraph, _tokens


def trace_processing(
    graph: NeuronGraph,
    query: str,
    *,
    seed_limit: int = 10,
    max_hops: int = 3,
    damping: float = 0.75,
    max_nodes: int = 500,
    workers: int = 1,
) -> dict[str, Any]:
    """Return observable query-processing state without selecting an answer.

    The trace contains lexical seeds, activated concepts grouped by hop count,
    relationship-phrase interpretations and evidence-bearing edges. It is a
    deterministic inspection surface over the current engine, not a biological
    brain simulator and not a decision-maker.
    """
    if seed_limit < 1:
        raise ValueError("seed_limit must be >= 1")
    if max_nodes < 1:
        raise ValueError("max_nodes must be >= 1")

    query_tokens = _tokens(query)
    seeds = graph.find_candidates(query, top_k=seed_limit)
    seed_ids = {neuron_id for neuron_id, _ in seeds}
    activation = []
    if seeds:
        activation = graph.spread_activation(
            {neuron_id: max(0.25, score) for neuron_id, score in seeds},
            max_hops=max_hops,
            damping=damping,
            max_nodes=max_nodes,
            workers=workers,
        )

    # Preserve the set of phrase interpretations instead of collapsing them
    # into a single relation or answer.
    relation_candidates: set[str] = set()
    for token in query_tokens:
        relation_candidates.update(graph._relation_token_index.get(token, ()))

    relation_hypotheses: list[dict[str, Any]] = []
    for relation in sorted(relation_candidates):
        phrases = graph._relation_phrases.get(relation, {})
        matches = [
            (phrase_tokens, alias)
            for phrase_tokens, alias in phrases.items()
            if phrase_tokens.issubset(query_tokens)
        ]
        if not matches:
            continue
        matches.sort(key=lambda pair: (-len(pair[0]), pair[1].casefold()))
        edge_options = []
        for source_id in sorted(seed_ids):
            for synapse_id in graph._outgoing_by_relation.get(source_id, {}).get(relation, ()):
                edge = graph.synapses[synapse_id]
                source = graph.neurons[edge.source_id]
                target = graph.neurons[edge.target_id]
                edge_options.append({
                    "synapse_id": edge.id,
                    "source_id": source.id,
                    "source": graph.knowledge[source.knowledge_id].name,
                    "target_id": target.id,
                    "target": graph.knowledge[target.knowledge_id].name,
                    "weight": edge.weight,
                    "support": edge.support,
                    "contradiction": edge.contradiction,
                    "confidence": edge.confidence,
                    "effective_strength": edge.effective_strength,
                })
        relation_hypotheses.append({
            "relation": relation,
            "matched_aliases": [alias for _, alias in matches],
            "seed_edges": edge_options,
        })

    active_by_hop: dict[str, list[dict[str, Any]]] = {}
    for item in sorted(activation, key=lambda value: (value.hops, value.neuron_id)):
        neuron = graph.neurons[item.neuron_id]
        knowledge = graph.knowledge[neuron.knowledge_id]
        path_names = [
            graph.knowledge[graph.neurons[nid].knowledge_id].name
            for nid in item.path
        ]
        path_edges = []
        for source_id, target_id in zip(item.path, item.path[1:]):
            source_edges = []
            for synapse_id in graph._outgoing.get(source_id, ()):
                edge = graph.synapses[synapse_id]
                if edge.target_id != target_id:
                    continue
                source_edges.append({
                    "synapse_id": edge.id,
                    "relation": edge.relation,
                    "weight": edge.weight,
                    "support": edge.support,
                    "contradiction": edge.contradiction,
                    "confidence": edge.confidence,
                    "effective_strength": edge.effective_strength,
                })
            path_edges.append({
                "from_id": source_id,
                "to_id": target_id,
                "edge_options": source_edges,
            })
        active_by_hop.setdefault(str(item.hops), []).append({
            "neuron_id": neuron.id,
            "knowledge_id": knowledge.id,
            "name": knowledge.name,
            "kind": knowledge.kind,
            "activation": round(item.score, 8),
            "path": path_names,
            "path_edges": path_edges,
        })

    return {
        "query": query,
        "query_tokens": sorted(query_tokens),
        "lexical_seeds": [
            {
                "neuron_id": neuron_id,
                "name": graph.knowledge[graph.neurons[neuron_id].knowledge_id].name,
                "lexical_score": round(score, 8),
            }
            for neuron_id, score in seeds
        ],
        "relation_hypotheses": relation_hypotheses,
        "activation_by_hop": active_by_hop,
        "active_concept_count": len(activation),
        "note": (
            "Intermediate processing trace only. It does not identify a correct answer "
            "or choose among interpretations."
        ),
    }
