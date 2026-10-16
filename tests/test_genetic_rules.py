import unittest
from neuron_core.genetic_rules import GeneticRuleRanker


class GeneticRuleRankerTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"name": "wrong", "score": 0.9, "lexical_score": 0.1, "hops": 3},
            {"name": "right", "score": 0.3, "lexical_score": 0.9, "hops": 1},
        ]

    def test_starts_without_learned_rules(self):
        learner = GeneticRuleRanker(seed=1, population_size=12, generations=4)
        self.assertEqual(learner.rules, [])
        self.assertEqual(learner.evolution_steps, 0)
        self.assertEqual(len(learner.rank("query", self.candidates)), 2)

    def test_feedback_evolves_inspectable_rules(self):
        learner = GeneticRuleRanker(seed=2, population_size=24, generations=8)
        result = learner.learn_from_mistake("query", self.candidates, ["right"])
        self.assertEqual(result["examples"], 1)
        self.assertEqual(learner.evolution_steps, 1)
        self.assertTrue(all({"feature", "threshold", "direction", "delta"} <= set(r) for r in learner.rules))

    def test_missing_target_is_rejected(self):
        learner = GeneticRuleRanker(seed=3, population_size=8, generations=2)
        with self.assertRaises(ValueError):
            learner.learn_from_mistake("query", self.candidates, ["missing"])

    def test_state_round_trip(self):
        learner = GeneticRuleRanker(seed=4, population_size=10, generations=3)
        learner.learn_from_mistake("query", self.candidates, ["right"])
        restored = GeneticRuleRanker(seed=5)
        restored.load_state_dict(learner.state_dict())
        self.assertEqual(restored.rules, learner.rules)
        self.assertEqual(restored.examples, learner.examples)


if __name__ == "__main__":
    unittest.main()
