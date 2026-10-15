"""Evidence-aware verification for decisions proposed by Neuron.

This module audits a decision against graph structure, the supplied processing
trace, and explicitly recorded edge evidence. It does not run retrieval, alter
the graph, or claim that graph-local evidence proves external real-world truth.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

from .engine import NeuronGraph


_PASS = "pass"
_FAIL = "fail"
_UNRESOLVED = "unresolved"


def verify_decision(
    graph: NeuronGraph,
    trace: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    min_support: float = 1.0,
    trusted_sources: Iterable[str] | None = None,
    trusted_interpreters: Iterable[str] | None = None,
    min_relation_confidence: float = 0.75,
    min_relation_margin: float = 0.10,
) -> dict[str, Any]:
    """Verify a proposed single-edge answer without making a new decision.

    Required proposal fields:
        source_id: graph neuron ID for the subject
        relation: exact typed edge relation
        target_id: graph neuron ID for the proposed answer

    Optional proposal field:
        path: node IDs for the claimed path. For this single-edge verifier the
              only accepted path is [source_id, target_id].

    The trace should be the output of trace_processing for the same query.
    A decision is verified only when the trace uniquely supports its relation,
    the source is present in the trace, the exact graph edge exists, and enough
    explicit positive evidence exists without recorded contradiction.

    A trace may include relation_resolution from an external semantic interpreter.
    Such a resolution can disambiguate multiple candidate relations only when it is
    marked resolved, names the proposed relation, meets the configured confidence
    and top-two margin thresholds, and (when supplied) uses an allowed interpreter
    label. Without a resolution, the original conservative unique-alias rule applies.

    If trusted_sources is provided, only evidence records whose source label is in
    that allowlist count toward support or contradiction. If trusted_interpreters is
    provided, only resolution labels in that allowlist may resolve relation ambiguity.
    These allowlists check labels; they do not cryptographically authenticate sources
    or interpreter implementations.

    Returns a structured verified, ambiguous, or rejected result. The function
    is read-only with respect to the graph and trace.
    """
    if not math.isfinite(float(min_support)) or min_support <= 0:
        raise ValueError("min_support must be finite and greater than zero")
    if not math.isfinite(float(min_relation_confidence)) or not 0.0 <= min_relation_confidence <= 1.0:
        raise ValueError("min_relation_confidence must be between 0 and 1")
    if not math.isfinite(float(min_relation_margin)) or not 0.0 <= min_relation_margin <= 1.0:
        raise ValueError("min_relation_margin must be between 0 and 1")

    trusted = None if trusted_sources is None else set(trusted_sources)
    trusted_interpreter_set = (
        None if trusted_interpreters is None else set(trusted_interpreters)
    )
    checks: dict[str, dict[str, str]] = {}
    reasons: list[str] = []
    evidence_rows: list[dict[str, Any]] = []

    def add_check(name: str, status: str, detail: str) -> None:
        checks[name] = {"status": status, "detail": detail}
        if status != _PASS:
            reasons.append(detail)

    def result() -> dict[str, Any]:
        statuses = [entry["status"] for entry in checks.values()]
        if _FAIL in statuses:
            overall = "rejected"
        elif _UNRESOLVED in statuses:
            overall = "ambiguous"
        else:
            overall = "verified"
        return {
            "status": overall,
            "accepted": overall == "verified",
            "proposal": {
                "source_id": proposal.get("source_id") if isinstance(proposal, Mapping) else None,
                "relation": proposal.get("relation") if isinstance(proposal, Mapping) else None,
                "target_id": proposal.get("target_id") if isinstance(proposal, Mapping) else None,
            },
            "checks": checks,
            "reasons": reasons,
            "evidence": evidence_rows,
            "limitations": (
                "Verification checks graph structure, processing-trace consistency, "
                "and recorded evidence. It does not independently authenticate evidence "
                "sources or establish external real-world truth."
            ),
        }

    if not isinstance(trace, Mapping):
        add_check("trace_available", _FAIL, "A processing trace mapping is required.")
        return result()
    add_check("trace_available", _PASS, "Processing trace supplied.")

    if not isinstance(proposal, Mapping):
        add_check("proposal_valid", _FAIL, "Proposal must be a mapping.")
        return result()

    source_id = proposal.get("source_id")
    relation = proposal.get("relation")
    target_id = proposal.get("target_id")
    missing = [
        field for field, value in (
            ("source_id", source_id), ("relation", relation), ("target_id", target_id)
        )
        if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        add_check(
            "proposal_valid", _FAIL,
            "Proposal is missing non-empty string field(s): " + ", ".join(missing) + ".",
        )
        return result()
    add_check("proposal_valid", _PASS, "Required proposal fields are present.")

    if source_id not in graph.neurons or target_id not in graph.neurons:
        unknown = [
            value for value in (source_id, target_id) if value not in graph.neurons
        ]
        add_check(
            "node_ids_exist", _FAIL,
            "Unknown graph neuron ID(s): " + ", ".join(unknown) + ".",
        )
        return result()
    add_check("node_ids_exist", _PASS, "Source and target IDs exist in the graph.")

    # Check that the proposed subject was among the query's candidate seeds.
    lexical_seeds = trace.get("lexical_seeds", [])
    seed_ids = {
        row.get("neuron_id")
        for row in lexical_seeds
        if isinstance(row, Mapping) and isinstance(row.get("neuron_id"), str)
    } if isinstance(lexical_seeds, list) else set()

    hypotheses = trace.get("relation_hypotheses", [])
    hypothesis_relations = {
        row.get("relation")
        for row in hypotheses
        if isinstance(row, Mapping) and isinstance(row.get("relation"), str)
    } if isinstance(hypotheses, list) else set()

    edge_seed_ids = {
        edge.get("source_id")
        for hypothesis in hypotheses
        if isinstance(hypothesis, Mapping)
        for edge in (hypothesis.get("seed_edges") or [])
        if isinstance(edge, Mapping) and isinstance(edge.get("source_id"), str)
    } if isinstance(hypotheses, list) else set()

    if source_id in seed_ids or source_id in edge_seed_ids:
        add_check("subject_in_trace", _PASS, "Proposed subject is present in the processing trace.")
    else:
        add_check(
            "subject_in_trace", _UNRESOLVED,
            "The proposed subject is not present in the trace's lexical seeds or relation edges.",
        )

    resolution = trace.get("relation_resolution")
    if resolution is not None:
        if not isinstance(resolution, Mapping):
            add_check(
                "relation_interpretation", _UNRESOLVED,
                "The relation resolution is malformed; relation meaning remains unverified.",
            )
        else:
            resolution_status = resolution.get("status")
            resolved_relation = resolution.get("relation")
            interpreter = resolution.get("source")
            confidence = resolution.get("confidence")
            margin = resolution.get("margin")
            if resolution_status != "resolved":
                add_check(
                    "relation_interpretation", _UNRESOLVED,
                    f"The relation interpreter returned {resolution_status!r}; no relation was accepted.",
                )
            elif not isinstance(interpreter, str) or not interpreter.strip():
                add_check(
                    "relation_interpretation", _UNRESOLVED,
                    "The resolved relation has no interpreter provenance label.",
                )
            elif trusted_interpreter_set is not None and interpreter not in trusted_interpreter_set:
                add_check(
                    "relation_interpretation", _UNRESOLVED,
                    f"Relation interpreter {interpreter!r} is not in the trusted interpreter allowlist.",
                )
            elif (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
                or isinstance(margin, bool)
                or not isinstance(margin, (int, float))
                or not math.isfinite(float(margin))
                or not 0.0 <= float(margin) <= 1.0
            ):
                add_check(
                    "relation_interpretation", _UNRESOLVED,
                    "Resolved relations require finite confidence and top-two margin values between 0 and 1.",
                )
            elif resolved_relation != relation:
                add_check(
                    "relation_interpretation", _FAIL,
                    f"The resolver selected {resolved_relation!r}, but the proposal uses {relation!r}.",
                )
            elif relation not in hypothesis_relations:
                add_check(
                    "relation_interpretation", _FAIL,
                    "The resolver's relation is absent from the trace's relation hypotheses.",
                )
            elif float(confidence) < min_relation_confidence or float(margin) < min_relation_margin:
                add_check(
                    "relation_interpretation", _UNRESOLVED,
                    f"Relation resolution is below threshold (confidence={float(confidence):.3f}, "
                    f"margin={float(margin):.3f}; required confidence>={min_relation_confidence:.3f}, "
                    f"margin>={min_relation_margin:.3f}).",
                )
            else:
                add_check(
                    "relation_interpretation", _PASS,
                    f"{interpreter} resolved relation {relation!r} "
                    f"(confidence={float(confidence):.3f}, margin={float(margin):.3f}).",
                )
    elif not hypothesis_relations:
        add_check(
            "relation_interpretation", _UNRESOLVED,
            "The trace contains no explicit relation hypothesis; relation meaning is unverified.",
        )
    elif relation not in hypothesis_relations:
        add_check(
            "relation_interpretation", _FAIL,
            f"Proposed relation {relation!r} conflicts with trace hypotheses "
            f"{sorted(hypothesis_relations)!r}.",
        )
    elif len(hypothesis_relations) != 1:
        add_check(
            "relation_interpretation", _UNRESOLVED,
            "The proposed relation is present, but the trace contains multiple relation hypotheses.",
        )
    else:
        add_check(
            "relation_interpretation", _PASS,
            "The trace contains exactly the proposed relation hypothesis.",
        )

    matching_edges = [
        edge for sid in graph._outgoing.get(source_id, ())
        if (edge := graph.synapses[sid]).relation == relation and edge.target_id == target_id
    ]
    if not matching_edges:
        add_check(
            "edge_exists", _FAIL,
            "No direct graph edge connects the proposed source to target with the proposed relation.",
        )
        return result()
    add_check(
        "edge_exists", _PASS,
        f"Found {len(matching_edges)} matching direct graph edge(s).",
    )

    # Confirm that the same source-relation-target triple appears in the trace.
    # A direct graph edge alone is not enough if it was not part of this query's
    # interpreted/activated evidence snapshot.
    trace_relation_targets = {
        edge.get("target_id")
        for hypothesis in hypotheses
        if isinstance(hypothesis, Mapping) and hypothesis.get("relation") == relation
        for edge in (hypothesis.get("seed_edges") or [])
        if isinstance(edge, Mapping) and edge.get("source_id") == source_id
    } if isinstance(hypotheses, list) else set()

    if target_id not in trace_relation_targets:
        add_check(
            "target_in_trace", _UNRESOLVED,
            "The proposed target is not present in the trace's edges for this source and relation; "
            "refresh the trace or gather more evidence.",
        )
    elif len(trace_relation_targets) > 1:
        add_check(
            "target_in_trace", _UNRESOLVED,
            "Multiple targets are connected to this subject by the proposed relation; "
            "the verifier cannot establish a unique answer.",
        )
    else:
        add_check(
            "target_in_trace", _PASS,
            "The trace supports a unique target for this subject and relation.",
        )

    # If supplied, validate the proposed path rather than accepting an opaque path.
    path = proposal.get("path")
    if path is not None:
        if not isinstance(path, (list, tuple)) or list(path) != [source_id, target_id]:
            add_check(
                "path_integrity", _FAIL,
                "For this single-edge verifier, path must be exactly [source_id, target_id].",
            )
            return result()
    add_check("path_integrity", _PASS, "The claimed direct path is structurally consistent.")

    positive_records: list[dict[str, Any]] = []
    contradictory_records: list[dict[str, Any]] = []
    for edge in matching_edges:
        source_neuron = graph.neurons[edge.source_id]
        target_neuron = graph.neurons[edge.target_id]
        source_name = graph.knowledge[source_neuron.knowledge_id].name
        target_name = graph.knowledge[target_neuron.knowledge_id].name
        included_records = []
        for record in edge.evidence:
            source = str(getattr(record, "source", "") or "").strip()
            if not source or (trusted is not None and source not in trusted):
                continue
            row = {
                "source": source,
                "positive": bool(record.positive),
                "strength": float(record.strength),
                "note": str(getattr(record, "note", "") or ""),
            }
            included_records.append(row)
            if row["positive"]:
                positive_records.append(row)
            else:
                contradictory_records.append(row)
        evidence_rows.append({
            "synapse_id": edge.id,
            "source_id": edge.source_id,
            "source": source_name,
            "relation": edge.relation,
            "target_id": edge.target_id,
            "target": target_name,
            "weight": edge.weight,
            "confidence": edge.confidence,
            "included_evidence": included_records,
        })

    positive_strength = sum(row["strength"] for row in positive_records)
    contradictory_strength = sum(row["strength"] for row in contradictory_records)

    if positive_strength < min_support:
        source_note = (
            "from trusted sources" if trusted is not None
            else "with a non-empty source label"
        )
        add_check(
            "positive_evidence", _UNRESOLVED,
            f"Recorded positive evidence is insufficient ({positive_strength:g} < "
            f"{min_support:g}) {source_note}.",
        )
    else:
        add_check(
            "positive_evidence", _PASS,
            f"Recorded positive evidence strength is {positive_strength:g}.",
        )

    if contradictory_strength > 0:
        add_check(
            "contradictions", _UNRESOLVED,
            f"Contradictory evidence remains ({contradictory_strength:g}); "
            "the verifier will not silently override it.",
        )
    else:
        add_check("contradictions", _PASS, "No included contradictory evidence was found.")

    return result()
