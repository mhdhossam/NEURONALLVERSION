import math
import unittest

from transformer_research.graph_transformer import (
    GraphCausalAttention,
    GraphLinear,
    GraphTransformerBlock,
    WeightedEdge,
)


class GraphOperationTests(unittest.TestCase):
    def test_graph_linear_matches_same_sparse_weight_reference(self):
        edges = [
            WeightedEdge(0, 0, 0.5),
            WeightedEdge(2, 0, -0.25),
            WeightedEdge(1, 1, 0.75),
            WeightedEdge(2, 1, 0.1),
        ]
        bias = [0.2, -0.1]
        layer = GraphLinear(3, 2, edges, bias)
        x = [2.0, 3.0, 4.0]
        actual = layer.forward(x)

        # Dense reference is deliberately test-only; runtime graph layer stores
        # and executes only the edge list.
        dense = [[0.0] * 3 for _ in range(2)]
        for edge in edges:
            dense[edge.target][edge.source] = edge.weight
        expected = [
            bias[row] + sum(dense[row][col] * x[col] for col in range(3))
            for row in range(2)
        ]
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=12)

    def test_attention_is_causal(self):
        attention = GraphCausalAttention(2)
        q = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        k = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        v1 = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        v2 = [[1.0, 0.0], [2.0, 0.0], [300.0, 0.0]]
        out1, _ = attention.forward(q, k, v1)
        out2, _ = attention.forward(q, k, v2)
        self.assertEqual(out1[0], out2[0])
        self.assertEqual(out1[1], out2[1])
        self.assertNotEqual(out1[2], out2[2])

    def test_local_attention_builds_only_window_edges(self):
        attention = GraphCausalAttention(4, window=2)
        q = [[1.0] * 4 for _ in range(5)]
        out, edges = attention.forward(q, q, q)
        self.assertEqual(edges, 9)  # 1 + 2 + 2 + 2 + 2
        self.assertEqual(len(out), 5)

    def test_full_graph_decoder_block_returns_finite_hidden_states(self):
        block = GraphTransformerBlock(
            hidden_size=8,
            intermediate_size=16,
            projection_fan_in=2,
            attention_window=3,
            seed=42,
        )
        tokens = [[(i + j + 1) / 10 for j in range(8)] for i in range(5)]
        result = block.forward(tokens)
        self.assertEqual(len(result["tokens"]), 5)
        self.assertTrue(all(len(row) == 8 for row in result["tokens"]))
        self.assertEqual(result["attention_edges"], 12)  # 1 + 2 + 3 + 3 + 3
        self.assertEqual(result["parameter_edges"], 2 * (8 * 8 + 8 * 16 + 16 * 8))
        self.assertTrue(all(
            math.isfinite(value)
            for row in result["tokens"]
            for value in row
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
