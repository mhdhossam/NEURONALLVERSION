# NEURONALLVERSION — Neuron v3 core

Neuron v3 is a dependency-free, graph-first knowledge and activation core based on the strongest ideas in the research notebooks: stable knowledge identity, typed regions, relationship graphs, bounded activation, evidence-based confidence, and explainable retrieval paths. The existing notebooks remain untouched as research history.

## Why this core exists

The research notebooks contain several competing versions of Neuron, Region, and Platform. Some components are also incompatible with one another. This package provides one consistent foundation instead of continuing to grow the notebook maze.

## Architecture

- **Neuron:** an addressable memory/activation unit with one canonical identifier and a pointer to its knowledge object.
- **KnowledgeObject:** stable concept or identity with attributes, typed relationships, and observation history.
- **Synapse:** directed, typed relationship. Configured weight expresses relevance; confidence comes from explicit positive and contradictory evidence.
- **Region:** optional domain bucket that restricts initial candidate discovery.
- **Activation:** bounded transient propagation with hop limit, damping, activation floor, and node cap. Activation alone does not establish that a relationship is true.
- **Preprocessing:** searchable neuron text and tokens are prepared when a neuron is created and stored with its metadata.
- **Candidate index:** an in-memory token-to-neuron inverted index retrieves candidates from matching postings instead of rebuilding and scanning every neuron's text on each query.
- **Relationship index:** typed outgoing edges are bucketed by relation, allowing relation-directed lookup without scanning all synapses.
- **Relation aliases:** common query phrases (for example, `kind of work` → `profession`) are preprocessed into the relation index. Register project/domain-specific aliases with `add_relation_alias(relation, alias)`; aliases are explicit rules, not open-ended semantic understanding.
- **Activation:** bounded, query-local propagation with a hop limit, damping, activation floor, and node cap. Only nodes touched in the prior/current activation snapshot are updated; there is no full-network reset.
- **Parallel frontier:** propagation can process sufficiently large edge frontiers in chunks using threads when explicitly requested (`workers=4`). The default is serial (`workers=1`) because the benchmark showed Python threads slower for the tested CPU-bound frontier. Small frontiers stay serial to avoid scheduling overhead. Since pure-Python CPU work is constrained by the CPython GIL, threading is a correctness/concurrency feature whose speed benefit must be measured, not assumed.
- **Retrieval:** indexed lexical candidates followed by graph-assisted retrieval; explicit relation matches can prioritize the connected target and return an inspectable path.
- **Metadata refresh:** call `refresh_neuron_metadata(neuron_id)` after editing a neuron's content/metadata or its knowledge attributes directly. Call `rebuild_search_index()` after bulk edits.
- **Feedback:** confidence changes only when explicit evidence or feedback is recorded, not merely because an item wins retrieval.
- **Persistence:** versioned JSON save/load with reference validation and indexes rebuilt during load.

## Quick start

```python
from neuron_core import NeuronGraph

brain = NeuronGraph()
person = brain.add_concept("Mohamed", kind="person")
project = brain.add_concept("Neuron", kind="project")
edge = brain.connect(person, project, "works_on", weight=0.9)
brain.add_edge_evidence(edge, positive=True, source="verified statement")

print(brain.find_path(person, project))
print(brain.search("Mohamed", max_hops=2))
# Opt into threaded frontier expansion for experiments:
print(brain.search("Mohamed", max_hops=2, workers=4))
brain.save("memory.json")

# If source knowledge changes after creation, refresh its prepared metadata.
brain.neurons[person].content = "backend engineer"
brain.refresh_neuron_metadata(person)
```

A longer demo is available in examples/demo.py.

## Run tests

Requires Python 3.10+; no third-party packages are required for the core.

```bash
python -m unittest discover -s tests -v
python examples/demo.py
```

## Deliberate limits

This is an engineering milestone, not a claim of human-like understanding. It does not train neural weights, infer truth from correlation, or parse unrestricted natural language. Query seeding is token overlap; graph links need to be supplied or learned from validated feedback. The optional threaded path may not improve CPU-bound Python workloads because of interpreter overhead/GIL contention. The aim is to make the structural hypothesis testable and behavior inspectable.

## Next research milestone

Benchmark equivalent methods on the same fixed dataset: lexical retrieval, vector-only retrieval, graph-only traversal from known seeds, and hybrid vector-seeded graph retrieval. Report accuracy/recall, path relevance, latency, memory use, and unknown rejection. Do not compare insertion timings if one method builds links and another does not.


## Processing trace (before a decision policy)

The `trace_processing` helper exposes intermediate graph-processing state without choosing a final answer:

```python
from neuron_core import NeuronGraph, trace_processing

brain = NeuronGraph()
person = brain.add_concept("Jordan Lee", kind="person")
city = brain.add_concept("Cairo", kind="city")
brain.connect(person, city, "lives_in")
brain.add_relation_alias("lives_in", "live")

trace = trace_processing(brain, "Where does Jordan Lee live?")
print(trace["lexical_seeds"])
print(trace["relation_hypotheses"])
print(trace["activation_by_hop"])
```

The trace includes lexical seeds, matched relation phrases, generic activation grouped by hop count, paths, and evidence values on traversed or relation-matched edges. For explicitly matched relations, it also preserves separate context-conditioned exploration branches per matching source edge; alternatives remain separate and no branch is selected as the answer. Branch propagation uses `commit_snapshot=False`, so it does not overwrite the public activation snapshot. The base trace pass updates that snapshot but does not append retrieval events to knowledge history. The `profession` aliases intentionally avoid the generic one-word alias `work`, which was ambiguous with workplace relationships.

Run the context-variation probe:

```bash
python benchmarks/benchmark_context_variation.py --json-output context-processing-results.json
```

This probe compares several wordings over identical facts, including ambiguous wording, explicit residence/workplace/birthplace cues, a multi-hop chain, and contradictory evidence. It records whether relation interpretations and the active node set change. It is a behavioral diagnostic—not a literal simulation of human neurobiology.


## Metadata-guard image-classification benchmark

Run the reproducible open-set guard benchmark from the repository root:

```bash
python -m pip install numpy scikit-learn torch
python benchmarks/benchmark_metadata_guard_classification.py --json-output benchmarks/results/metadata-guard-classification-results.json

# Repeat over several fixed seeds and aggregate mean/std:
python benchmarks/benchmark_metadata_guard_multiseed.py --seeds 2022,2023,2024,2026 --json-output benchmarks/results/metadata-guard-multiseed-summary.json
```

The benchmark extracts the current `_normalise_embedding` and `find_best_in_candidates` functions and thresholds directly from `Untitled36.ipynb`. It compares the guard-enabled decision, the same similarity ranking without rejection thresholds (ablation), and a small CNN on a shared held-out test split. Digits 0–7 are known classes; digits 8–9 are withheld from training to probe unknown rejection. The known-class training pool is split again into a fit partition and a known-only calibration partition. In addition to closed-set CNN accuracy, the benchmark evaluates a CNN confidence-rejection policy whose threshold is calibrated on known-only validation confidence, without using unknown test labels. The JSON report includes accuracy, coverage, unknown rejection/false acceptance, latency, thresholds, and guard regression checks. CI uploads the report as an artifact. A four-seed summary is checked in at `benchmarks/results/metadata-guard-multiseed-summary.json`; it is still only a small-image guard-policy diagnostic, not a FaceNet evaluation.

This offline test isolates the metadata guard on small grayscale image features; it is **not** a FaceNet benchmark and does not replace a run on the original local face-recognition dataset/pretrained embeddings.

## Decision verification

A decision policy can pass its proposed subject, typed relation, target, and the
processing trace to the read-only verifier. The verifier checks that the relation
is unambiguous in the trace, the edge and optional direct path exist, the trace
supports a unique target, and recorded evidence meets the configured threshold.

```python
from neuron_core import NeuronGraph, trace_processing, verify_decision

brain = NeuronGraph()
person = brain.add_concept("Mira", kind="person")
workplace = brain.add_concept("Orion Lab", kind="organization")
edge = brain.connect(person, workplace, "works_at")
brain.add_relation_alias("works_at", "work")
brain.add_edge_evidence(edge, positive=True, source="trusted-registry")

query = "Where does Mira work?"
trace = trace_processing(brain, query)
proposal = {
    "source_id": person,
    "relation": "works_at",
    "target_id": workplace,
    "path": [person, workplace],
}
result = verify_decision(
    brain, trace, proposal,
    trusted_sources={"trusted-registry"},
)
if result["accepted"]:
    print("Decision passed graph and evidence verification")
else:
    print(result["status"], result["reasons"])
```

Statuses are `verified`, `ambiguous`, or `rejected`. Supply an explicit
`trusted_sources` allowlist for evidence labels that the application has vetted.
Without such an allowlist, non-empty evidence labels count as recorded support
but are not independently authenticated. The verifier currently handles direct
single-edge answers; it does not prove external truth or validate multi-hop
reasoning paths. Missing evidence, contradictory evidence, competing targets, or
unclear relation interpretations prevent automatic acceptance.
