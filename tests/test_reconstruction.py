import unittest

from neuron_core import NeuronGraph
from neuron_core.reconstruction import ReconstructedNeuron


class ReconstructionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.graph = NeuronGraph()
        self.person = self.graph.add_concept(
            "Mohamed", kind="person", attributes={"role": "developer"}
        )
        self.project = self.graph.add_concept(
            "Neuron", kind="project", content="graph-first reasoning system"
        )
        self.tool = self.graph.add_concept("Graph traversal", kind="concept")
        self.graph.connect(self.person, self.project, "works_on", weight=0.9)
        self.graph.connect(self.project, self.tool, "uses", weight=0.8)
        self.policy = ReconstructedNeuron(self.graph)

    def test_policy_explores_graph_paths_not_only_final_candidate_order(self):
        rows = self.policy.search("Mohamed", top_k=5, max_hops=2)
        by_id = {row["neuron_id"]: row for row in rows}
        self.assertIn(self.project, by_id)
        self.assertIn(self.tool, by_id)
        self.assertEqual(by_id[self.tool]["path"], ["Mohamed", "Neuron", "Graph traversal"])
        self.assertEqual(by_id[self.tool]["hops"], 2)
        self.assertEqual(len(by_id[self.tool]["reconstruction_trace"]["edge_ids"]), 2)

    def test_verified_feedback_updates_future_activation_policy(self):
        before = list(self.policy.activation_weights)
        rows = self.policy.search("Mohamed", top_k=5, max_hops=2)
        self.policy.observe("Mohamed", self.project, 1.0)
        self.assertNotEqual(before, self.policy.activation_weights)
        self.assertEqual(self.policy.feedback_count, 1)
        self.assertEqual(rows[0]["neuron_id"], self.person)

    def test_feedback_updates_traversal_policy_too(self):
        before = list(self.policy.path_weights)
        self.policy.search("Mohamed", top_k=5, max_hops=2)
        self.policy.observe("Mohamed", self.tool, 1.0)
        self.assertNotEqual(before, self.policy.path_weights)

    def test_negative_feedback_reduces_used_path_weights(self):
        before = list(self.policy.path_weights)
        self.policy.search("Mohamed", top_k=5, max_hops=2)
        self.policy.observe("Mohamed", self.tool, -1.0)
        self.assertTrue(any(after < prior for prior, after in zip(before, self.policy.path_weights)))

    def test_feedback_requires_a_real_result_from_latest_matching_query(self):
        self.policy.search("Mohamed", top_k=5, max_hops=2)
        with self.assertRaises(ValueError):
            self.policy.observe("different query", self.project, 1.0)
        with self.assertRaises(ValueError):
            self.policy.observe("Mohamed", "not-a-result", 1.0)

    def test_policy_state_round_trip(self):
        self.policy.search("Mohamed", top_k=5, max_hops=2)
        self.policy.observe("Mohamed", self.tool, 1.0)
        restored = ReconstructedNeuron(self.graph)
        restored.load_state_dict(self.policy.state_dict())
        self.assertEqual(restored.state_dict(), self.policy.state_dict())

    def test_unknown_query_does_not_guess(self):
        self.assertEqual(self.policy.search("zorbax not known", top_k=5), [])

    def test_bad_rewards_are_rejected(self):
        self.policy.search("Mohamed", top_k=5, max_hops=2)
        with self.assertRaises(ValueError):
            self.policy.observe("Mohamed", self.project, 2.0)


if __name__ == "__main__":
    unittest.main()
