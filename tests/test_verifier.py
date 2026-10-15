import unittest

from neuron_core import NeuronGraph, trace_processing, verify_decision


class DecisionVerifierTests(unittest.TestCase):
    def setUp(self):
        self.graph = NeuronGraph()
        self.mira = self.graph.add_concept("Mira", kind="person")
        self.orion = self.graph.add_concept("Orion Lab", kind="organization")
        self.edge = self.graph.connect(self.mira, self.orion, "works_at", weight=1.0)
        self.graph.add_relation_alias("works_at", "work")
        self.query = "Where does Mira work?"
        self.trace = trace_processing(self.graph, self.query)
        self.proposal = {
            "source_id": self.mira,
            "relation": "works_at",
            "target_id": self.orion,
            "path": [self.mira, self.orion],
        }

    def test_verified_when_trace_edge_and_trusted_evidence_agree(self):
        self.graph.add_edge_evidence(
            self.edge, positive=True, source="trusted-db", note="record exists"
        )
        result = verify_decision(
            self.graph, self.trace, self.proposal, trusted_sources={"trusted-db"}
        )
        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["checks"]["edge_exists"]["status"], "pass")
        self.assertEqual(result["checks"]["positive_evidence"]["status"], "pass")

    def test_missing_recorded_evidence_is_ambiguous_not_verified(self):
        result = verify_decision(self.graph, self.trace, self.proposal)
        self.assertEqual(result["status"], "ambiguous")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["checks"]["positive_evidence"]["status"], "unresolved")

    def test_untrusted_evidence_does_not_satisfy_allowlist(self):
        self.graph.add_edge_evidence(
            self.edge, positive=True, source="unreviewed-note"
        )
        result = verify_decision(
            self.graph, self.trace, self.proposal, trusted_sources={"trusted-db"}
        )
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["checks"]["positive_evidence"]["status"], "unresolved")

    def test_contradictory_evidence_blocks_automatic_acceptance(self):
        self.graph.add_edge_evidence(
            self.edge, positive=True, source="trusted-db"
        )
        self.graph.add_edge_evidence(
            self.edge, positive=False, source="trusted-db", note="conflicting record"
        )
        result = verify_decision(
            self.graph, self.trace, self.proposal, trusted_sources={"trusted-db"}
        )
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["checks"]["contradictions"]["status"], "unresolved")

    def test_missing_edge_rejects_proposal(self):
        wrong_target = self.graph.add_concept("Unknown Lab", kind="organization")
        proposal = {**self.proposal, "target_id": wrong_target}
        result = verify_decision(self.graph, self.trace, proposal)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["checks"]["edge_exists"]["status"], "fail")

    def test_relation_conflicting_with_trace_is_rejected(self):
        home = self.graph.add_concept("Cairo", kind="city")
        self.graph.connect(self.mira, home, "lives_in")
        proposal = {**self.proposal, "relation": "lives_in", "target_id": home}
        result = verify_decision(self.graph, self.trace, proposal)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["checks"]["relation_interpretation"]["status"], "fail")

    def test_invalid_path_is_rejected(self):
        proposal = {**self.proposal, "path": [self.mira, self.orion, self.mira]}
        result = verify_decision(self.graph, self.trace, proposal)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["checks"]["path_integrity"]["status"], "fail")

    def test_multiple_relation_hypotheses_remain_ambiguous(self):
        second_target = self.graph.add_concept("Another Lab", kind="organization")
        self.graph.connect(self.mira, second_target, "employment")
        self.graph.add_relation_alias("employment", "work")
        self.trace = trace_processing(self.graph, self.query)
        self.graph.add_edge_evidence(self.edge, positive=True, source="trusted-db")
        result = verify_decision(
            self.graph, self.trace, self.proposal, trusted_sources={"trusted-db"}
        )
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["checks"]["relation_interpretation"]["status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
