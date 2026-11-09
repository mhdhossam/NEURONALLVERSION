"""Smoke tests for the graph-native Phase-0 renderer."""
import unittest

from vision_research.neuron_vision import compile_prompt, render_svg


class NeuronVisionTests(unittest.TestCase):
    def test_prompt_builds_typed_graph(self):
        graph, scene, trace = compile_prompt("a red car beside a tree")
        self.assertIn("scene", graph.nodes)
        self.assertEqual([obj["kind"] for obj in scene["objects"]], ["vehicle", "tree"])
        self.assertTrue(any(edge.relation == "contains" for edge in graph.edges))
        self.assertTrue(any(edge.relation == "beside" for edge in graph.edges))
        self.assertEqual(trace[0]["active"], {"scene": 1.0})

    def test_graph_activation_reaches_objects(self):
        graph, _, trace = compile_prompt("a house and a sun")
        self.assertEqual(set(trace[1]["active"]), {"object_0", "object_1"})
        self.assertAlmostEqual(trace[1]["active"]["object_0"], 0.65)

    def test_renderer_emits_svg(self):
        _, scene, _ = compile_prompt("a blue house under a cloud")
        svg = render_svg(scene)
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        self.assertIn("<rect", svg)

    def test_unknown_prompt_has_safe_fallback(self):
        _, scene, _ = compile_prompt("something not in vocabulary")
        self.assertTrue(scene["objects"])


if __name__ == "__main__":
    unittest.main()
