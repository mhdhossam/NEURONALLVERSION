"""A dependency-free, graph-first knowledge and activation engine for Neuron v3."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tempfile
from threading import Lock
from typing import Any, Iterable, Mapping, Optional
from uuid import uuid4

SCHEMA_VERSION = 1

# Small, explicit vocabulary used to map common query wording to typed graph edges.
# These phrases are pre-indexed; they are not inferred by a neural model at query time.
DEFAULT_RELATION_ALIASES: dict[str, tuple[str, ...]] = {
    "profession": ("occupation", "job", "career", "work", "kind of work"),
    "specialty": ("expertise", "subject", "area of expertise", "skilled in"),
    "research_field": ("research area", "area", "study", "studies", "field of research"),
    "home_city": ("home town", "hometown", "live", "lives", "reside", "residence"),
    "primary_tool": ("tool", "main tool", "mainly use", "usually use"),
    "affiliation": ("organization", "connected with", "associated with", "member of"),
    "discipline": ("subject", "academic discipline", "field of study"),
    "invention": ("create", "created", "make", "made", "build", "built"),
}


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
    activation: float = 0.0  # most recent committed activation snapshot
    last_active_step: int = 0
    search_text: str = ""  # prepared once; refresh after changing source knowledge
    search_tokens: list[str] = field(default_factory=list)

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
        self._outgoing_by_relation: dict[str, dict[str, list[str]]] = {}
        self._token_index: dict[str, set[str]] = {}
        self._relation_token_index: dict[str, set[str]] = {}
        self._relation_phrases: dict[str, dict[frozenset[str], str]] = {}
        self._custom_relation_aliases: dict[str, set[str]] = {}
        self._active_neuron_ids: set[str] = set()
        self._activation_lock = Lock()

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
        self._outgoing_by_relation[nid] = {}
        self._index_neuron(nid)
        return nid

    def _index_neuron(self, neuron_id: str) -> None:
        """Preprocess one neuron's searchable metadata and refresh its postings."""
        neuron = self._require_neuron(neuron_id)
        for token in neuron.search_tokens:
            posting = self._token_index.get(token)
            if posting is not None:
                posting.discard(neuron_id)
                if not posting:
                    del self._token_index[token]

        knowledge = self.knowledge[neuron.knowledge_id]
        attrs = " ".join(f"{key} {value}" for key, value in knowledge.attributes.items())
        prepared_text = " ".join([
            neuron.name, neuron.kind, neuron.content, str(neuron.metadata),
            attrs, knowledge.name,
        ])
        prepared_tokens = _tokens(prepared_text)
        neuron.search_text = prepared_text
        neuron.search_tokens = sorted(prepared_tokens)
        for token in prepared_tokens:
            self._token_index.setdefault(token, set()).add(neuron_id)

    def refresh_neuron_metadata(self, neuron_id: str) -> None:
        """Re-preprocess a neuron after its content, metadata, or knowledge attributes change."""
        self._index_neuron(neuron_id)

    def rebuild_search_index(self) -> None:
        """Rebuild the in-memory inverted index after bulk knowledge edits."""
        self._token_index = {}
        for neuron_id in self.neurons:
            # _index_neuron also removes the old postings when present.
            self._index_neuron(neuron_id)

    def _index_relation(self, relation: str) -> None:
        """Preprocess canonical relation wording and aliases into a token index."""
        old_phrases = self._relation_phrases.get(relation, {})
        for phrase_tokens in old_phrases:
            for token in phrase_tokens:
                posting = self._relation_token_index.get(token)
                if posting is not None:
                    posting.discard(relation)
                    if not posting:
                        del self._relation_token_index[token]

        aliases = set(DEFAULT_RELATION_ALIASES.get(relation, ()))
        aliases.update(self._custom_relation_aliases.get(relation, set()))
        aliases.add(relation.replace("_", " "))
        phrases: dict[frozenset[str], str] = {}
        for alias in aliases:
            phrase_tokens = frozenset(_tokens(alias))
            if phrase_tokens:
                phrases[phrase_tokens] = alias
        self._relation_phrases[relation] = phrases
        for phrase_tokens in phrases:
            for token in phrase_tokens:
                self._relation_token_index.setdefault(token, set()).add(relation)

    def add_relation_alias(self, relation: str, alias: str) -> None:
        """Register a domain-specific phrase for a relationship and index it immediately."""
        if not relation or not relation.strip():
            raise ValueError("relation must not be empty")
        if not alias or not alias.strip() or not _tokens(alias):
            raise ValueError("alias must contain at least one searchable token")
        relation = relation.strip()
        self._custom_relation_aliases.setdefault(relation, set()).add(alias.strip())
        self._index_relation(relation)

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
        self._outgoing_by_relation[source_id].setdefault(edge.relation, []).append(sid)
        self._index_relation(edge.relation)
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
                          min_activation: float = 0.01, max_nodes: int = 500,
                          workers: int = 1) -> list[ActivationResult]:
        """Propagate over the active frontier only.

        Expansion is parallelized in chunks when a frontier has enough edges.
        Activation scores remain query-local; only nodes touched by this cycle or
        the previous committed cycle are updated in the public activation snapshot.
        """
        if max_hops < 0:
            raise ValueError("max_hops must be >= 0")
        if max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        if workers < 1:
            raise ValueError("workers must be >= 1")
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

        def expand_chunk(jobs: list[tuple[str, str, float, tuple[str, ...]]]
                         ) -> list[tuple[str, str, float, list[str]]]:
            expanded = []
            for source_id, sid, source_score, source_path in jobs:
                edge = self.synapses[sid]
                signal = source_score * damp * edge.effective_strength
                target_id = edge.target_id
                if signal < floor or target_id in source_path:
                    continue
                expanded.append((
                    source_id, target_id, signal, list(source_path) + [target_id]
                ))
            return expanded

        pool_context = ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
        with pool_context as pool:
            for hop in range(1, max_hops + 1):
                jobs = [
                    (source_id, sid, frontier[source_id], tuple(paths[source_id]))
                    for source_id in sorted(frontier)
                    for sid in self._outgoing.get(source_id, [])
                ]
                if not jobs:
                    break

                # Thread work is deliberately chunked to reduce scheduling overhead.
                # For small frontiers, the serial path is both simpler and often faster.
                chunk_count = min(workers, len(jobs))
                should_parallelize = pool is not None and len(jobs) >= max(16, workers * 4)
                if should_parallelize:
                    chunk_size = (len(jobs) + chunk_count - 1) // chunk_count
                    chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]
                    expanded_batches = pool.map(expand_chunk, chunks)
                else:
                    expanded_batches = (expand_chunk(jobs),)

                next_frontier: dict[str, float] = {}
                next_paths: dict[str, list[str]] = {}
                for expanded in expanded_batches:
                    for source_id, target_id, signal, candidate_path in expanded:
                        if signal > next_frontier.get(target_id, 0.0):
                            next_frontier[target_id] = signal
                            next_paths[target_id] = candidate_path

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

        # Avoid an O(total-neurons) reset. Clear only the prior touched set, then
        # publish this query's snapshot. Query result values above remain local.
        with self._activation_lock:
            for nid in self._active_neuron_ids:
                neuron = self.neurons.get(nid)
                if neuron is not None:
                    neuron.activation = 0.0
                    neuron.last_active_step = 0
            self._active_neuron_ids = set(scores)
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
        """Retrieve lexical candidates through preprocessed token postings."""
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

        # Candidate discovery jumps directly to postings for query terms.
        candidate_ids: set[str] = set()
        for token in query_terms:
            candidate_ids.update(self._token_index.get(token, ()))
        if allowed is not None:
            candidate_ids.intersection_update(allowed)

        scored = []
        for nid in candidate_ids:
            neuron = self.neurons[nid]
            overlap = len(query_terms.intersection(neuron.search_tokens)) / len(query_terms)
            if overlap > 0:
                scored.append((nid, overlap))
        scored.sort(key=lambda p: (-p[1], self.neurons[p[0]].name.casefold(), p[0]))
        return scored[:top_k]

    def search(self, query: str, *, top_k: int = 5, max_hops: int = 2,
               damping: float = 0.75, region_id: Optional[str] = None,
               workers: int = 1, record_history: bool = True) -> list[dict[str, Any]]:
        """Search preprocessed candidates, propagate activation, and favor explicit relation answers.

        Query seeding remains lexical rather than semantic language understanding.
        When a query names a relationship (for example, "A uses what?"), direct
        matching outgoing edges receive an explicit answer-ranking bonus.
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
            workers=workers,
        )
        lexical = dict(seeds)
        results: dict[str, dict[str, Any]] = {}
        for item in activation:
            neuron = self.neurons[item.neuron_id]
            ko = self.knowledge[neuron.knowledge_id]
            score = 0.7 * item.score + 0.3 * lexical.get(neuron.id, 0.0)
            results[neuron.id] = {
                "neuron_id": neuron.id, "knowledge_id": ko.id, "name": ko.name,
                "kind": ko.kind, "score": round(min(1.0, score), 6),
                "lexical_score": round(lexical.get(neuron.id, 0.0), 6),
                "activation": round(item.score, 6),
                "path": [self.neurons[nid].name for nid in item.path],
                "hops": item.hops, "attributes": dict(ko.attributes),
            }

        # Relation lookup is indexed by relation tokens; only seed neurons' matching
        # outgoing relation buckets are visited. No whole-graph edge scan is needed.
        query_terms = _tokens(query.replace("?", " ").replace("!", " ").replace(".", " "))
        relation_candidates: set[str] = set()
        for token in query_terms:
            relation_candidates.update(self._relation_token_index.get(token, ()))
        matching_relations: dict[str, tuple[frozenset[str], str]] = {}
        for relation in relation_candidates:
            phrases = self._relation_phrases.get(relation, {})
            matched = [
                (phrase_tokens, alias)
                for phrase_tokens, alias in phrases.items()
                if phrase_tokens.issubset(query_terms)
            ]
            if matched:
                matching_relations[relation] = max(
                    matched, key=lambda item: (len(item[0]), item[1].casefold())
                )

        seed_scores = dict(seeds)
        for source_id, seed_score in seeds:
            outgoing_by_relation = self._outgoing_by_relation.get(source_id, {})
            for relation in sorted(matching_relations):
                matched_phrase, matched_alias = matching_relations[relation]
                for sid in outgoing_by_relation.get(relation, ()):
                    edge = self.synapses[sid]
                    target = self.neurons[edge.target_id]
                    ko = self.knowledge[target.knowledge_id]
                    specificity_bonus = min(0.08, 0.02 * max(0, len(matched_phrase) - 1))
                    direct_score = min(
                        1.0,
                        0.6 + 0.4 * min(1.0, seed_score * edge.weight) + specificity_bonus,
                    )
                    row = results.get(target.id)
                    if row is None:
                        row = {
                            "neuron_id": target.id, "knowledge_id": ko.id, "name": ko.name,
                            "kind": ko.kind, "score": round(direct_score, 6),
                            "lexical_score": round(lexical.get(target.id, 0.0), 6),
                            "activation": round(target.activation, 6),
                            "path": [self.neurons[source_id].name, target.name],
                            "hops": 1, "attributes": dict(ko.attributes),
                        }
                        results[target.id] = row
                    elif direct_score > row["score"]:
                        row["score"] = round(direct_score, 6)
                        row["path"] = [self.neurons[source_id].name, target.name]
                        row["hops"] = 1
                    if direct_score > row.get("relation_score", -1.0):
                        row["relation_score"] = round(direct_score, 6)
                        row["matched_relation"] = relation
                        row["matched_relation_alias"] = matched_alias

        ranked = sorted(results.values(), key=lambda row: (
            -row["score"], row["hops"], row["name"].casefold(), row["neuron_id"]
        ))
        if record_history:
            for row in ranked[:top_k]:
                self.knowledge[row["knowledge_id"]].observe(
                    "retrieved", query=query, score=row["score"], path=row["path"]
                )
        return ranked[:top_k]

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
            "relation_aliases": {
                relation: sorted(aliases)
                for relation, aliases in self._custom_relation_aliases.items()
            },
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
        graph._custom_relation_aliases = {
            str(relation): set(aliases)
            for relation, aliases in payload.get("relation_aliases", {}).items()
        }
        for raw in payload.get("knowledge", []):
            item = dict(raw)
            graph.knowledge[item["id"]] = KnowledgeObject(**item)
        for raw in payload.get("neurons", []):
            item = dict(raw)
            graph.neurons[item["id"]] = Neuron(**item)
            graph._outgoing[item["id"]] = []
            graph._outgoing_by_relation[item["id"]] = {}
            graph._index_neuron(item["id"])
        for raw in payload.get("synapses", []):
            item = dict(raw)
            item["evidence"] = [Evidence(**ev) for ev in item.get("evidence", [])]
            edge = Synapse(**item)
            if edge.source_id not in graph.neurons or edge.target_id not in graph.neurons:
                raise ValueError(f"synapse {edge.id} references a missing neuron")
            graph.synapses[edge.id] = edge
            graph._outgoing[edge.source_id].append(edge.id)
            graph._outgoing_by_relation[edge.source_id].setdefault(edge.relation, []).append(edge.id)
            graph._index_relation(edge.relation)
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
