import tempfile
import unittest
from pathlib import Path

from neuron_core import NeuronGraph


class NeuronGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = NeuronGraph()
        self.mohamed = self.graph.add_concept(
            "Mohamed", kind="person", attributes={"role": "developer", "city": "Cairo"}
        )
        self.neuron = self.graph.add_concept("Neuron", kind="project", content="graph-first memory system")
        self.graph_project = self.graph.add_concept("Graph processing", kind="concept")
        self.edge_works = self.graph.connect(self.mohamed, self.neuron, "works_on", weight=0.9)
        self.edge_uses = self.graph.connect(self.neuron, self.graph_project, "uses", weight=0.8)

    def test_one_canonical_id_for_neuron_and_knowledge(self):
        neuron = self.graph.neurons[self.mohamed]
        self.assertEqual(neuron.id, neuron.knowledge_id)
        self.assertEqual(self.graph.knowledge[neuron.knowledge_id].name, "Mohamed")

    def test_graph_finds_shortest_path_and_relation_filter(self):
        path = self.graph.find_path(self.mohamed, self.graph_project, max_depth=2)
        self.assertEqual([step["relation"] for step in path], ["works_on", "uses"])
        self.assertIsNone(self.graph.find_path(
            self.mohamed, self.graph_project, relation="owns", max_depth=3
        ))

    def test_activation_is_bounded_and_returns_path(self):
        results = self.graph.spread_activation([self.mohamed], max_hops=3, damping=0.8)
        by_id = {result.neuron_id: result for result in results}
        self.assertIn(self.graph_project, by_id)
        self.assertGreater(by_id[self.graph_project].score, 0)
        self.assertLessEqual(by_id[self.graph_project].score, 1)
        self.assertEqual(by_id[self.graph_project].path,
                         [self.mohamed, self.neuron, self.graph_project])
        self.assertEqual(self.graph.neurons[self.graph_project].activation,
                         by_id[self.graph_project].score)

    def test_cycles_terminate(self):
        self.graph.connect(self.graph_project, self.mohamed, "relates_to")
        results = self.graph.spread_activation([self.mohamed], max_hops=20, max_nodes=10)
        self.assertLessEqual(len(results), 3)

    def test_search_does_not_learn_from_activation(self):
        edge = self.graph.synapses[self.edge_works]
        before = edge.confidence
        self.graph.spread_activation([self.mohamed])
        self.assertEqual(edge.confidence, before)

    def test_explicit_evidence_changes_confidence(self):
        edge = self.graph.synapses[self.edge_works]
        before = edge.confidence
        positive = self.graph.add_edge_evidence(
            self.edge_works, positive=True, source="verified fact"
        )
        negative = self.graph.add_edge_evidence(
            self.edge_works, positive=False, source="correction"
        )
        self.assertGreater(positive, before)
        self.assertLess(negative, positive)

    def test_neuron_feedback_is_explicit(self):
        before = self.graph.neurons[self.mohamed].confidence
        after = self.graph.record_feedback(self.mohamed, positive=True, source="confirmed")
        self.assertGreater(after, before)
        events = [item["event"] for item in self.graph.knowledge[self.mohamed].history]
        self.assertEqual(events.count("feedback"), 1)

    def test_search_uses_graph_context(self):
        rows = self.graph.search("Mohamed", top_k=3, max_hops=2)
        ids = {row["neuron_id"] for row in rows}
        self.assertIn(self.mohamed, ids)
        self.assertIn(self.neuron, ids)
        self.assertTrue(all("path" in row for row in rows))

    def test_regions_limit_candidate_seeds(self):
        region = self.graph.add_region("people", domain="identity")
        self.graph.assign_to_region(self.mohamed, region)
        self.graph.assign_to_region(self.neuron, region)
        self.assertEqual(self.graph.find_candidates("Cairo", region_id=region),
                         [(self.mohamed, 1.0)])

    def test_json_round_trip_preserves_relationships_and_evidence(self):
        self.graph.add_edge_evidence(
            self.edge_works, positive=True, source="test source", note="confirmed"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brain.json"
            self.graph.save(path)
            loaded = NeuronGraph.load(path)
        self.assertEqual(len(loaded.neurons), len(self.graph.neurons))
        self.assertEqual(loaded.find_path(
            self.mohamed, self.graph_project, max_depth=2
        )[1]["relation"], "uses")
        self.assertEqual(loaded.synapses[self.edge_works].evidence[0].source, "test source")
        self.assertEqual(loaded.knowledge[self.mohamed].relationships["works_on"], [self.neuron])

    def test_invalid_ids_and_weights_fail_loudly(self):
        with self.assertRaises(KeyError):
            self.graph.connect("missing", self.neuron, "uses")
        with self.assertRaises(ValueError):
            self.graph.connect(self.mohamed, self.neuron, "uses", weight=1.5)

    def test_unknown_query_returns_no_guess(self):
        self.assertEqual(self.graph.search("nonexistent material", top_k=5), [])
        self.assertIsNone(self.graph.find_path(
            self.mohamed, self.graph_project, relation="is_identical_to"
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
