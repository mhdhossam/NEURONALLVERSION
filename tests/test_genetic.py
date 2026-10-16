import unittest
from neuron_core.genetic import GeneticRanker


class GeneticRankerTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"name": "wrong", "score": 0.9, "lexical_score": 0.1, "hops": 3},
            {"name": "right", "score": 0.3, "lexical_score": 0.9, "hops": 1},
        ]

    def test_first_attempt_has_no_feedback_examples(self):
        ranker = GeneticRanker(seed=1, population_size=12, generations=4)
        self.assertEqual(ranker.examples, [])
        self.assertEqual(len(ranker.rank("query", self.candidates)), 2)
        self.assertEqual(ranker.evolution_steps, 0)

    def test_feedback_records_example_and_evolves(self):
        ranker = GeneticRanker(seed=2, population_size=20, generations=8)
        before = list(ranker.genome)
        result = ranker.learn_from_mistake("query", self.candidates, ["right"])
        self.assertEqual(result["examples"], 1)
        self.assertEqual(ranker.evolution_steps, 1)
        self.assertNotEqual(before, ranker.genome)

    def test_missing_answer_cannot_be_learned_by_reranking(self):
        ranker = GeneticRanker(seed=3, population_size=8, generations=2)
        with self.assertRaises(ValueError):
            ranker.learn_from_mistake("query", self.candidates, ["missing"])

    def test_state_round_trip(self):
        ranker = GeneticRanker(seed=4, population_size=10, generations=3)
        ranker.learn_from_mistake("query", self.candidates, ["right"])
        restored = GeneticRanker(seed=5)
        restored.load_state_dict(ranker.state_dict())
        self.assertEqual(restored.genome, ranker.genome)
        self.assertEqual(restored.examples, ranker.examples)
        self.assertEqual(restored.evolution_steps, 1)


if __name__ == "__main__":
    unittest.main()
