import unittest

from neuron_core import NeuronGraph, explore_reasoning


class NeuronReasoningExperimentTests(unittest.TestCase):
    def setUp(self):
        self.graph = NeuronGraph()
        self.mira = self.graph.add_concept("Mira", kind="person")
        self.orion = self.graph.add_concept("Orion Lab", kind="organization")
        self.cairo = self.graph.add_concept("Cairo", kind="city")
        self.alexandria = self.graph.add_concept("Alexandria", kind="city")
        self.employment = self.graph.connect(self.mira, self.orion, "works_at")
        self.graph.connect(self.orion, self.cairo, "located_in")
        self.graph.connect(self.orion, self.alexandria, "also_operates_in")

    def test_explores_distinct_two_hop_routes_without_selecting_answer(self):
        result = explore_reasoning(
            self.graph, "Mira", max_hops=2, seed_limit=1, min_strength=0.0
        )
        two_hop = [path for path in result["paths"] if path["hops"] == 2]
        endpoints = {path["end"] for path in two_hop}
        self.assertEqual(endpoints, {"Cairo", "Alexandria"})
        self.assertTrue(all(len(path["edges"]) == path["hops"] for path in two_hop))
        self.assertIsNone(result["decision"])
        self.assertEqual(result["paths_explored"], len(result["paths"]))

    def test_preserves_support_and_contradiction_without_winner_selection(self):
        engineer = self.graph.add_concept("Senior Engineer", kind="profession")
        consultant = self.graph.add_concept("Independent Consultant", kind="profession")
        supported_edge = self.graph.connect(self.mira, engineer, "profession")
        contradicted_edge = self.graph.connect(self.mira, consultant, "profession")
        self.graph.add_edge_evidence(
            supported_edge, positive=True, source="verified profile", note="confirmed"
        )
        self.graph.add_edge_evidence(
            contradicted_edge, positive=False, source="correction", note="disputed"
        )

        result = explore_reasoning(
            self.graph, "Mira", max_hops=1, seed_limit=1, min_strength=0.0
        )
        by_end = {path["end"]: path for path in result["paths"]}
        self.assertIn("Senior Engineer", by_end)
        self.assertIn("Independent Consultant", by_end)
        self.assertGreater(by_end["Senior Engineer"]["edges"][0]["support"], 0)
        self.assertGreater(by_end["Independent Consultant"]["edges"][0]["contradiction"], 0)
        self.assertIsNone(result["decision"])

    def test_relation_filter_limits_paths(self):
        result = explore_reasoning(
            self.graph, "Mira", max_hops=2, seed_limit=1,
            allowed_relations={"works_at"}, min_strength=0.0,
        )
        self.assertTrue(result["paths"])
        self.assertTrue(all(path["relations"] == ["works_at"] for path in result["paths"]))
        self.assertEqual(result["relation_filter"], ["works_at"])

    def test_cycles_do_not_repeat_a_node_within_a_path(self):
        self.graph.connect(self.orion, self.mira, "loops_to")
        result = explore_reasoning(
            self.graph, "Mira", max_hops=5, seed_limit=1, min_strength=0.0
        )
        for path in result["paths"]:
            self.assertEqual(len(path["nodes"]), len(set(path["nodes"])))

    def test_path_budget_is_explicit(self):
        result = explore_reasoning(
            self.graph, "Mira", max_hops=3, seed_limit=1,
            max_paths=1, min_strength=0.0,
        )
        self.assertEqual(result["paths_explored"], 1)
        self.assertTrue(result["paths_truncated"])

    def test_invalid_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            explore_reasoning(self.graph, "Mira", max_hops=-1)
        with self.assertRaises(ValueError):
            explore_reasoning(self.graph, "Mira", max_paths=0)


if __name__ == "__main__":
    unittest.main()
