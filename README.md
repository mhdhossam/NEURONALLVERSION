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

The trace includes lexical seeds, matched relation phrases, activation grouped by hop count, paths, and evidence values on traversed or relation-matched edges. It intentionally does not expose an `answer` or `decision` field; a downstream component can interpret this state according to the application's own policy. Calling the trace updates the graph's activation snapshot in the same way as a propagation pass, but does not append retrieval events to knowledge history.

Run the context-variation probe:

```bash
python benchmarks/benchmark_context_variation.py --json-output context-processing-results.json
```

This probe compares several wordings over identical facts, including ambiguous wording, explicit residence/workplace/birthplace cues, a multi-hop chain, and contradictory evidence. It records whether relation interpretations and the active node set change. It is a behavioral diagnostic—not a literal simulation of human neurobiology.
