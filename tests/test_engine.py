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
        self.graph.add_relation_alias("works_on", "project")
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
        rows = loaded.search(
            "What project does Mohamed focus on?",
            top_k=3, max_hops=2, workers=1, record_history=False,
        )
        self.assertEqual(rows[0]["neuron_id"], self.neuron)
        self.assertEqual(rows[0]["matched_relation_alias"], "project")

    def test_invalid_ids_and_weights_fail_loudly(self):
        with self.assertRaises(KeyError):
            self.graph.connect("missing", self.neuron, "uses")
        with self.assertRaises(ValueError):
            self.graph.connect(self.mohamed, self.neuron, "uses", weight=1.5)

    def test_metadata_is_preprocessed_and_refreshable(self):
        neuron = self.graph.neurons[self.neuron]
        self.assertIn("memory", neuron.search_tokens)
        self.assertIn(self.neuron, self.graph._token_index["memory"])

        neuron.content = "stardust-only content"
        self.graph.refresh_neuron_metadata(self.neuron)

        self.assertNotIn(self.neuron, self.graph._token_index.get("memory", set()))
        self.assertEqual(self.graph.find_candidates("stardust", top_k=3),
                         [(self.neuron, 1.0)])

    def test_unvisited_neurons_are_not_reset(self):
        unrelated = self.graph.add_concept("Unvisited concept")
        self.graph.neurons[unrelated].activation = 0.37

        self.graph.spread_activation([self.mohamed], max_hops=1, workers=1)

        self.assertEqual(self.graph.neurons[unrelated].activation, 0.37)

    def test_previous_touched_activation_is_cleared_without_global_reset(self):
        self.graph.spread_activation([self.graph_project], max_hops=0, workers=1)
        self.assertEqual(self.graph.neurons[self.graph_project].activation, 1.0)

        self.graph.spread_activation([self.mohamed], max_hops=0, workers=1)
        self.assertEqual(self.graph.neurons[self.graph_project].activation, 0.0)
        self.assertEqual(self.graph.neurons[self.mohamed].activation, 1.0)

    def test_threaded_frontier_matches_serial_results(self):
        graph = NeuronGraph()
        seed_ids = []
        for index in range(32):
            source = graph.add_concept(f"source-{index}")
            target = graph.add_concept(f"target-{index}")
            graph.connect(source, target, "points_to")
            seed_ids.append(source)

        serial = graph.spread_activation(seed_ids, max_hops=1, workers=1)
        threaded = graph.spread_activation(seed_ids, max_hops=1, workers=4)
        self.assertEqual(serial, threaded)

    def test_explicit_relation_query_ranks_target_first(self):
        rows = self.graph.search("Neuron uses what?", top_k=3, max_hops=2, workers=1)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["neuron_id"], self.graph_project)
        self.assertEqual(rows[0]["matched_relation"], "uses")
        self.assertEqual(rows[0]["path"], ["Neuron", "Graph processing"])

    def test_multiword_relation_query_uses_preprocessed_relation_index(self):
        rows = self.graph.search(
            "Mohamed works on what?", top_k=3, max_hops=2, workers=1
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["neuron_id"], self.neuron)
        self.assertEqual(rows[0]["matched_relation"], "works_on")
        self.assertEqual(rows[0]["path"], ["Mohamed", "Neuron"])

    def test_search_can_skip_history_for_clean_benchmarking(self):
        before = {
            nid: len(knowledge.history)
            for nid, knowledge in self.graph.knowledge.items()
        }
        rows = self.graph.search(
            "Mohamed works_on what", top_k=3, workers=1, record_history=False
        )
        self.assertTrue(rows)
        after = {
            nid: len(knowledge.history)
            for nid, knowledge in self.graph.knowledge.items()
        }
        self.assertEqual(after, before)

    def test_preprocessed_relation_alias_handles_paraphrase(self):
        graph = NeuronGraph()
        ada = graph.add_concept("Ada Lovelace", kind="person")
        answer = graph.add_concept("Computational mathematics", kind="concept")
        graph.connect(ada, answer, "profession")

        rows = graph.search(
            "What kind of work does Ada Lovelace do?",
            top_k=3, max_hops=2, workers=1, record_history=False,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["neuron_id"], answer)
        self.assertEqual(rows[0]["matched_relation"], "profession")
        self.assertEqual(rows[0]["matched_relation_alias"], "kind of work")

    def test_unknown_query_returns_no_guess(self):
        self.assertEqual(self.graph.search("nonexistent material", top_k=5), [])
        self.assertIsNone(self.graph.find_path(
            self.mohamed, self.graph_project, relation="is_identical_to"
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
