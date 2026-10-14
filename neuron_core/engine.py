"""A dependency-free, graph-first knowledge and activation engine for Neuron v3."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Optional
from uuid import uuid4

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_number(value: float, name: str, low: float, high: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return number


def _tokens(text: str) -> set[str]:
    # Unicode-aware tokenization; Arabic and other scripts are retained.
    return {token for token in re.findall(r"\w+", text.casefold()) if len(token) > 1}


@dataclass
class Evidence:
    id: str = field(default_factory=lambda: str(uuid4()))
    positive: bool = True
    source: str = "unspecified"
    strength: float = 1.0
    note: str = ""
    created_at: str = field(default_factory=_now)


@dataclass
class KnowledgeObject:
    """Canonical identity/concept record, independent of any representation."""
    id: str
    name: str
    kind: str = "concept"
    attributes: dict[str, Any] = field(default_factory=dict)
    relationships: dict[str, list[str]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def relate(self, relation: str, target_id: str) -> None:
        targets = self.relationships.setdefault(relation, [])
        if target_id not in targets:
            targets.append(target_id)
            self.updated_at = _now()

    def observe(self, event: str, **details: Any) -> None:
        self.history.append({"event": event, "timestamp": _now(), **details})
        self.updated_at = _now()


@dataclass
class Neuron:
    """Addressable activation unit pointing to canonical knowledge."""
    id: str
    knowledge_id: str
    name: str
    kind: str = "concept"
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    support: float = 0.0
    contradiction: float = 0.0
    activation: float = 0.0  # latest retrieval cycle only
    last_active_step: int = 0

    @property
    def confidence(self) -> float:
        # Neutral Beta(1, 1) prior; activation itself does not update confidence.
        return (self.support + 1.0) / (self.support + self.contradiction + 2.0)


@dataclass
class Synapse:
    id: str
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    support: float = 0.0
    contradiction: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return (self.support + 1.0) / (self.support + self.contradiction + 2.0)

    @property
    def effective_strength(self) -> float:
        return self.weight * self.confidence


@dataclass
class Region:
    id: str
    name: str
    domain: str = "general"
    neuron_ids: list[str] = field(default_factory=list)


@dataclass
class ActivationResult:
    neuron_id: str
    score: float
    path: list[str]
    hops: int


class NeuronGraph:
    """Graph-first memory, retrieval, bounded activation and evidence updates."""

    def __init__(self) -> None:
        self.neurons: dict[str, Neuron] = {}
        self.knowledge: dict[str, KnowledgeObject] = {}
        self.synapses: dict[str, Synapse] = {}
        self.regions: dict[str, Region] = {}
        self._outgoing: dict[str, list[str]] = {}

    def add_concept(
        self, name: str, *, kind: str = "concept", content: str = "",
        attributes: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        neuron_id: Optional[str] = None, knowledge_id: Optional[str] = None,
    ) -> str:
        """Create a neuron and canonical knowledge object with consistent IDs.

        Pass knowledge_id to add another representation of an existing object.
        """
        if not name or not name.strip():
            raise ValueError("name must not be empty")
        nid = neuron_id or (
            knowledge_id if knowledge_id and knowledge_id not in self.knowledge else str(uuid4())
        )
        kid = knowledge_id or nid
        if nid in self.neurons:
            raise ValueError(f"neuron ID already exists: {nid}")
        if kid not in self.knowledge:
            self.knowledge[kid] = KnowledgeObject(
                id=kid, name=name.strip(), kind=kind, attributes=dict(attributes or {})
            )
        elif knowledge_id is None:
            raise ValueError(f"knowledge ID already exists: {kid}")
        self.neurons[nid] = Neuron(
            id=nid, knowledge_id=kid, name=name.strip(), kind=kind,
            content=content, metadata=dict(metadata or {}),
        )
        self._outgoing[nid] = []
        return nid

    def add_region(self, name: str, *, domain: str = "general",
                   region_id: Optional[str] = None) -> str:
        if not name or not name.strip():
            raise ValueError("region name must not be empty")
        rid = region_id or str(uuid4())
        if rid in self.regions:
            raise ValueError(f"region ID already exists: {rid}")
        self.regions[rid] = Region(rid, name.strip(), domain)
        return rid

    def assign_to_region(self, neuron_id: str, region_id: str) -> None:
        self._require_neuron(neuron_id)
        region = self.regions.get(region_id)
        if region is None:
            raise KeyError(f"unknown region: {region_id}")
        if neuron_id not in region.neuron_ids:
            region.neuron_ids.append(neuron_id)

    def connect(self, source_id: str, target_id: str, relation: str, *,
                weight: float = 1.0, synapse_id: Optional[str] = None) -> str:
        self._require_neuron(source_id)
        self._require_neuron(target_id)
        if not relation or not relation.strip():
            raise ValueError("relation must not be empty")
        w = _bounded_number(weight, "weight", 0.0, 1.0)
        sid = synapse_id or str(uuid4())
        if sid in self.synapses:
            raise ValueError(f"synapse ID already exists: {sid}")
        edge = Synapse(sid, source_id, target_id, relation.strip(), weight=w)
        self.synapses[sid] = edge
        self._outgoing[source_id].append(sid)
        source = self.neurons[source_id]
        target = self.neurons[target_id]
        source_knowledge = self.knowledge[source.knowledge_id]
        source_knowledge.relate(relation.strip(), target.knowledge_id)
        source_knowledge.observe(
            "relationship_added", relation=relation.strip(), target_id=target.knowledge_id
        )
        return sid

    def add_edge_evidence(self, synapse_id: str, *, positive: bool,
                          source: str = "unspecified", strength: float = 1.0,
                          note: str = "") -> float:
        edge = self.synapses.get(synapse_id)
        if edge is None:
            raise KeyError(f"unknown synapse: {synapse_id}")
        amount = _bounded_number(strength, "strength", 0.0, 1.0)
        if amount == 0:
            raise ValueError("strength must be greater than zero")
        edge.evidence.append(Evidence(
            positive=bool(positive), source=source, strength=amount, note=note
        ))
        if positive:
            edge.support += amount
        else:
            edge.contradiction += amount
        source_neuron = self.neurons[edge.source_id]
        self.knowledge[source_neuron.knowledge_id].observe(
            "relationship_evidence", synapse_id=synapse_id, positive=bool(positive),
            source=source, strength=amount,
        )
        return edge.confidence

    def record_feedback(self, neuron_id: str, *, positive: bool,
                        source: str = "user_feedback", strength: float = 1.0,
                        note: str = "") -> float:
        neuron = self._require_neuron(neuron_id)
        amount = _bounded_number(strength, "strength", 0.0, 1.0)
        if amount == 0:
            raise ValueError("strength must be greater than zero")
        if positive:
            neuron.support += amount
        else:
            neuron.contradiction += amount
        self.knowledge[neuron.knowledge_id].observe(
            "feedback", positive=bool(positive), source=source, strength=amount,
            note=note, neuron_id=neuron_id,
        )
        return neuron.confidence

    def spread_activation(self, seeds: Mapping[str, float] | Iterable[str], *,
                          max_hops: int = 3, damping: float = 0.75,
                          min_activation: float = 0.01,
                          max_nodes: int = 500) -> list[ActivationResult]:
        if max_hops < 0:
            raise ValueError("max_hops must be >= 0")
        if max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        damp = _bounded_number(damping, "damping", 0.0, 1.0)
        floor = _bounded_number(min_activation, "min_activation", 0.0, 1.0)
        raw_seeds = seeds.items() if isinstance(seeds, Mapping) else ((nid, 1.0) for nid in seeds)
        seed_scores: dict[str, float] = {}
        for nid, score in raw_seeds:
            self._require_neuron(nid)
            value = _bounded_number(score, "seed score", 0.0, 1.0)
            if value > 0:
                seed_scores[nid] = max(seed_scores.get(nid, 0.0), value)
        if not seed_scores:
            return []

        scores = dict(seed_scores)
        paths = {nid: [nid] for nid in seed_scores}
        hops_used = {nid: 0 for nid in seed_scores}
        frontier = dict(seed_scores)
        for hop in range(1, max_hops + 1):
            next_frontier: dict[str, float] = {}
            next_paths: dict[str, list[str]] = {}
            for source_id in sorted(frontier):
                for sid in self._outgoing.get(source_id, []):
                    edge = self.synapses[sid]
                    target_id = edge.target_id
                    signal = frontier[source_id] * damp * edge.effective_strength
                    if signal < floor or target_id in paths[source_id]:
                        continue
                    if signal > next_frontier.get(target_id, 0.0):
                        next_frontier[target_id] = signal
                        next_paths[target_id] = paths[source_id] + [target_id]
            if not next_frontier:
                break
            for nid in sorted(next_frontier):
                if nid not in scores and len(scores) >= max_nodes:
                    continue
                if next_frontier[nid] > scores.get(nid, 0.0):
                    scores[nid] = next_frontier[nid]
                    paths[nid] = next_paths[nid]
                    hops_used[nid] = hop
            frontier = {
                nid: score for nid, score in next_frontier.items()
                if nid in scores and hops_used.get(nid) == hop
            }
            if not frontier or len(scores) >= max_nodes:
                break

        for neuron in self.neurons.values():
            neuron.activation = 0.0
        for nid, score in scores.items():
            self.neurons[nid].activation = min(1.0, score)
            self.neurons[nid].last_active_step = hops_used[nid]
        results = [
            ActivationResult(nid, score, paths[nid], hops_used[nid])
            for nid, score in scores.items()
        ]
        return sorted(results, key=lambda item: (-item.score, item.hops, item.neuron_id))

    def find_path(self, source_id: str, target_id: str, *,
                  relation: Optional[str] = None, max_depth: int = 5
                  ) -> Optional[list[dict[str, str]]]:
        """Return shortest directed path, or None if no matching path exists."""
        self._require_neuron(source_id)
        self._require_neuron(target_id)
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if source_id == target_id:
            return []
        queue = deque([(source_id, [])])
        best_depth = {source_id: 0}
        while queue:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for sid in self._outgoing.get(current, []):
                edge = self.synapses[sid]
                if relation is not None and edge.relation != relation:
                    continue
                depth = len(path) + 1
                if edge.target_id in best_depth and best_depth[edge.target_id] <= depth:
                    continue
                step = path + [{
                    "synapse_id": sid, "source_id": edge.source_id,
                    "target_id": edge.target_id, "relation": edge.relation,
                }]
                if edge.target_id == target_id:
                    return step
                best_depth[edge.target_id] = len(step)
                queue.append((edge.target_id, step))
        return None

    def find_candidates(self, query: str, *, top_k: int = 5,
                        region_id: Optional[str] = None) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_terms = _tokens(query)
        if not query_terms:
            return []
        allowed = None
        if region_id is not None:
            region = self.regions.get(region_id)
            if region is None:
                raise KeyError(f"unknown region: {region_id}")
            allowed = set(region.neuron_ids)
        scored = []
        for nid, neuron in self.neurons.items():
            if allowed is not None and nid not in allowed:
                continue
            ko = self.knowledge[neuron.knowledge_id]
            attrs = " ".join(f"{k} {v}" for k, v in ko.attributes.items())
            text = " ".join([
                neuron.name, neuron.kind, neuron.content, str(neuron.metadata), attrs, ko.name
            ])
            terms = _tokens(text)
            overlap = len(query_terms & terms) / len(query_terms) if terms else 0.0
            if overlap > 0:
                scored.append((nid, overlap))
        scored.sort(key=lambda p: (-p[1], self.neurons[p[0]].name.casefold(), p[0]))
        return scored[:top_k]

    def search(self, query: str, *, top_k: int = 5, max_hops: int = 2,
               damping: float = 0.75, region_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Lexically seed candidates, then return related context through the graph.

        Candidate matching is deliberately a simple baseline, not semantic
        language understanding. An embedding encoder can be added at this boundary.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        seeds = self.find_candidates(query, top_k=max(top_k, min(10, top_k * 2)),
                                     region_id=region_id)
        if not seeds:
            return []
        activation = self.spread_activation(
            {nid: max(0.25, score) for nid, score in seeds},
            max_hops=max_hops, damping=damping, max_nodes=max(100, top_k * 10),
        )
        lexical = dict(seeds)
        results = []
        for item in activation:
            neuron = self.neurons[item.neuron_id]
            ko = self.knowledge[neuron.knowledge_id]
            score = 0.7 * item.score + 0.3 * lexical.get(neuron.id, 0.0)
            results.append({
                "neuron_id": neuron.id, "knowledge_id": ko.id, "name": ko.name,
                "kind": ko.kind, "score": round(min(1.0, score), 6),
                "lexical_score": round(lexical.get(neuron.id, 0.0), 6),
                "activation": round(item.score, 6),
                "path": [self.neurons[nid].name for nid in item.path],
                "hops": item.hops, "attributes": dict(ko.attributes),
            })
        results.sort(key=lambda row: (
            -row["score"], row["hops"], row["name"].casefold(), row["neuron_id"]
        ))
        for row in results[:top_k]:
            self.knowledge[row["knowledge_id"]].observe(
                "retrieved", query=query, score=row["score"], path=row["path"]
            )
        return results[:top_k]

    def get_relationships(self, neuron_id: str) -> list[dict[str, Any]]:
        self._require_neuron(neuron_id)
        rows = []
        for sid in self._outgoing.get(neuron_id, []):
            edge = self.synapses[sid]
            target = self.neurons[edge.target_id]
            rows.append({
                "synapse_id": edge.id, "relation": edge.relation,
                "target_id": target.id,
                "target_name": self.knowledge[target.knowledge_id].name,
                "weight": edge.weight, "confidence": edge.confidence,
                "evidence_count": len(edge.evidence),
            })
        return sorted(rows, key=lambda row: (row["relation"], row["target_name"].casefold()))

    def _require_neuron(self, neuron_id: str) -> Neuron:
        neuron = self.neurons.get(neuron_id)
        if neuron is None:
            raise KeyError(f"unknown neuron: {neuron_id}")
        return neuron

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "neurons": [asdict(item) for item in self.neurons.values()],
            "knowledge": [asdict(item) for item in self.knowledge.values()],
            "synapses": [asdict(item) for item in self.synapses.values()],
            "regions": [asdict(item) for item in self.regions.values()],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False
            ) as handle:
                handle.write(payload)
                handle.flush()
                temp_path = Path(handle.name)
            temp_path.replace(destination)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "NeuronGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {payload.get('schema_version')!r}")
        graph = cls()
        for raw in payload.get("knowledge", []):
            item = dict(raw)
            graph.knowledge[item["id"]] = KnowledgeObject(**item)
        for raw in payload.get("neurons", []):
            item = dict(raw)
            graph.neurons[item["id"]] = Neuron(**item)
            graph._outgoing[item["id"]] = []
        for raw in payload.get("synapses", []):
            item = dict(raw)
            item["evidence"] = [Evidence(**ev) for ev in item.get("evidence", [])]
            edge = Synapse(**item)
            if edge.source_id not in graph.neurons or edge.target_id not in graph.neurons:
                raise ValueError(f"synapse {edge.id} references a missing neuron")
            graph.synapses[edge.id] = edge
            graph._outgoing[edge.source_id].append(edge.id)
        for raw in payload.get("regions", []):
            item = dict(raw)
            graph.regions[item["id"]] = Region(**item)
        for neuron in graph.neurons.values():
            if neuron.knowledge_id not in graph.knowledge:
                raise ValueError(f"neuron {neuron.id} references missing knowledge {neuron.knowledge_id}")
        for region in graph.regions.values():
            if any(nid not in graph.neurons for nid in region.neuron_ids):
                raise ValueError(f"region {region.id} references a missing neuron")
        return graph
