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
- **Parallel frontier:** propagation can process sufficiently large edge frontiers in chunks using threads when explicitly requested (`workers=4`). The default is serial (`workers=1`) because the benchmark showed Python threads slower for the tested CPU-bound frontier.
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
print(brain.search("Mohamed", max_hops=2, workers=4))
brain.save("memory.json")
```

A longer demo is available in `examples/demo.py`.

## Run tests

Requires Python 3.10+; no third-party packages are required for the core.

```bash
python -m unittest discover -s tests -v
python examples/demo.py
```

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

The trace includes lexical seeds, matched relation phrases, generic activation grouped by hop count, paths, and evidence values on traversed or relation-matched edges. For explicitly matched relations, it preserves separate context-conditioned exploration branches; alternatives remain separate and no branch is selected as the answer.

## v4 reasoning experiment (isolated branch)

The experimental `explore_reasoning` function enumerates bounded graph paths from lexical query seeds. It keeps separate routes to the same endpoint, includes each edge's support and contradiction evidence, prevents cycles within a path, and reports when its path budget is exhausted. It returns `decision: None`: this is deliberately an exploration primitive, not a policy that chooses a final answer.

```python
from neuron_core import NeuronGraph, explore_reasoning

brain = NeuronGraph()
person = brain.add_concept("Mira", kind="person")
company = brain.add_concept("Orion Lab", kind="organization")
city = brain.add_concept("Cairo", kind="city")
language = brain.add_concept("Rust", kind="programming_language")
brain.connect(person, company, "works_at")
brain.connect(company, city, "located_in")
brain.connect(city, language, "uses")

trace = explore_reasoning(brain, "Mira", max_hops=3)
for path in trace["paths"]:
    print(path["nodes"], path["relations"], path["exploration_strength"])
```

Run the focused tests and path-exploration benchmark:

```bash
python -m unittest discover -s tests -p "test_reasoning.py" -v
python benchmarks/benchmark_reasoning_experiment.py
```

The benchmark tests known-path discovery as unrelated graph content grows; it is not proof of open-ended natural-language reasoning. This work lives on `neuron-v4-reasoning-experiment` so it can be evaluated without changing `neuron-v3-core` or the current retrieval policy.

## Deliberate limits

This is an engineering experiment, not a claim of human-like understanding. Query seeding is lexical; relations must be represented in the graph, and aliases are configured rules. Graph reachability does not by itself prove a fact, resolve ambiguity, infer unrestricted natural-language meaning, or select truth from contradictory evidence. The goal of this experiment is to make relational exploration bounded, inspectable, and measurable.
