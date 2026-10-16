import unittest
from neuron_core.decision import ContextualBanditRanker, candidate_features

class ContextualBanditRankerTests(unittest.TestCase):
    def test_feature_vector_is_small_and_deterministic(self):
        item = {"name": "Cairo", "score": 0.8, "lexical_score": 0.5, "hops": 1}
        self.assertEqual(candidate_features("Where is Cairo?", item), candidate_features("Where is Cairo?", item))
        self.assertEqual(len(candidate_features("Where is Cairo?", item)), 8)

    def test_feedback_changes_rankings_and_state_round_trips(self):
        ranker = ContextualBanditRanker(exploration=0.0)
        good = {"name": "Cairo", "score": 0.8, "lexical_score": 0.8, "hops": 1}
        bad = {"name": "Amman", "score": 0.7, "lexical_score": 0.1, "hops": 2}
        before = [row["name"] for row in ranker.rank("Where is Cairo?", [bad, good])]
        for _ in range(8):
            ranker.observe("Where is Cairo?", good, 1.0)
            ranker.observe("Where is Cairo?", bad, -1.0)
        after = [row["name"] for row in ranker.rank("Where is Cairo?", [bad, good])]
        self.assertEqual(ranker.observations, 16)
        self.assertNotEqual(before, after)
        restored = ContextualBanditRanker(exploration=0.0)
        restored.load_state_dict(ranker.state_dict())
        self.assertEqual(restored.rank("Where is Cairo?", [bad, good]), ranker.rank("Where is Cairo?", [bad, good]))

    def test_empty_candidates_and_invalid_rewards(self):
        ranker = ContextualBanditRanker()
        self.assertIsNone(ranker.choose("unknown", []))
        with self.assertRaises(ValueError):
            ranker.observe("q", {"name": "x"}, 2.0)

    def test_negative_evidence_feature_is_lower(self):
        supported = candidate_features("job", {"name": "job", "support": 3, "contradiction": 0})
        disputed = candidate_features("job", {"name": "job", "support": 0, "contradiction": 3})
        self.assertGreater(supported[-1], disputed[-1])

if __name__ == "__main__":
    unittest.main()
