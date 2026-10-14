"""Run from the repository root: python examples/demo.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuron_core import NeuronGraph

brain = NeuronGraph()
mohamed = brain.add_concept("Mohamed", kind="person", attributes={"role": "developer"})
neuron = brain.add_concept(
    "Neuron", kind="project", content="graph-first knowledge and reasoning"
)
graphs = brain.add_concept("Graph processing", kind="concept")

works_on = brain.connect(mohamed, neuron, "works_on", weight=0.9)
brain.connect(neuron, graphs, "uses", weight=0.8)
brain.add_edge_evidence(
    works_on, positive=True, source="verified statement",
    note="The project maintainer confirmed it.",
)

print("Graph-assisted retrieval for 'Mohamed':")
for result in brain.search("Mohamed", top_k=5, max_hops=2):
    print(f"- {result['name']}: score={result['score']:.3f}; path={' -> '.join(result['path'])}")

print("\nExplicit path to graph processing:")
path = brain.find_path(mohamed, graphs, max_depth=3)
for step in path or []:
    source = brain.knowledge[brain.neurons[step["source_id"]].knowledge_id].name
    target = brain.knowledge[brain.neurons[step["target_id"]].knowledge_id].name
    print(f"- {source} --{step['relation']}--> {target}")

brain.save("neuron_memory.json")
print("\nSaved memory to neuron_memory.json")
