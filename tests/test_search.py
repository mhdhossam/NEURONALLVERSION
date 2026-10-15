import tempfile
import unittest
from pathlib import Path

from neuron_core import NeuronGraph
from neuron_core.search import (
    SearchResult,
    SQLiteCachedSearchProvider,
    think_with_search,
)


class StubSearchProvider:
    cache_key = "stub-v1"

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def search(self, query, *, limit=5):
        self.calls.append((query, limit))
        return self.results[:limit]


class NeuronSearchTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            SearchResult(
                title="Quantum entanglement",
                url="https://example.org/quantum",
                snippet="Quantum entanglement is a physical phenomenon involving linked states.",
                provider="stub-v1",
            ),
            SearchResult(
                title="Quantum information",
                url="https://example.org/info",
                snippet="Information-theoretic view of quantum systems.",
                provider="stub-v1",
            ),
        ]
        self.provider = StubSearchProvider(self.results)
        self.graph = NeuronGraph()

    def test_unknown_query_triggers_search_and_ingests_sources(self):
        result = think_with_search(self.graph, "quantum entanglement", self.provider)
        self.assertTrue(result["search_triggered"])
        self.assertEqual(result["search_reason"], "no_local_lexical_seed")
        self.assertEqual(len(result["search_results"]), 2)
        self.assertEqual(len(result["source_node_ids"]), 2)
        self.assertIsNone(result["decision"])
        self.assertTrue(all(
            self.graph.neurons[nid].kind == "external_source"
            for nid in result["source_node_ids"]
        ))
        self.assertEqual(len(self.graph.synapses), 2)
        self.assertTrue(all(
            edge.relation == "found_by_search" for edge in self.graph.synapses.values()
        ))

    def test_existing_local_candidate_skips_search_unless_forced(self):
        self.graph.add_concept("Quantum entanglement", kind="local_topic")
        result = think_with_search(self.graph, "quantum entanglement", self.provider)
        self.assertFalse(result["search_triggered"])
        self.assertEqual(self.provider.calls, [])

        forced = think_with_search(
            self.graph, "quantum entanglement", self.provider, force_search=True
        )
        self.assertTrue(forced["search_triggered"])
        self.assertEqual(len(self.provider.calls), 1)

    def test_repeated_search_reuses_source_nodes_and_edges(self):
        first = think_with_search(
            self.graph, "quantum entanglement", self.provider, force_search=True
        )
        neuron_count = len(self.graph.neurons)
        edge_count = len(self.graph.synapses)
        second = think_with_search(
            self.graph, "quantum entanglement", self.provider, force_search=True
        )
        self.assertEqual(len(self.graph.neurons), neuron_count)
        self.assertEqual(len(self.graph.synapses), edge_count)
        self.assertEqual(first["source_node_ids"], second["source_node_ids"])

    def test_search_cache_avoids_repeat_provider_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = SQLiteCachedSearchProvider(
                self.provider, Path(directory) / "cache.sqlite3",
                ttl_seconds=60, max_entries=10,
            )
            first = cache.search("  Quantum   Entanglement ", limit=2)
            second = cache.search("quantum entanglement", limit=2)
        self.assertEqual(first, second)
        self.assertEqual(len(self.provider.calls), 1)

    def test_empty_query_and_invalid_limits_fail(self):
        with self.assertRaises(ValueError):
            think_with_search(self.graph, " ", self.provider)
        with self.assertRaises(ValueError):
            think_with_search(self.graph, "unknown", self.provider, result_limit=21)

    def test_sources_are_not_marked_as_verified_fact_evidence(self):
        think_with_search(self.graph, "quantum entanglement", self.provider)
        self.assertTrue(all(
            edge.support == 0 and edge.contradiction == 0 and not edge.evidence
            for edge in self.graph.synapses.values()
        ))


if __name__ == "__main__":
    unittest.main()


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit=-1):
        return self.payload[:limit]


class SearchProviderAdapterTests(unittest.TestCase):
    def test_duckduckgo_html_parsing_works_without_network(self):
        from unittest.mock import patch
        from neuron_core.search import DuckDuckGoHTMLProvider

        html = b"""
        <html><body>
          <a class="result__a" href="https://example.org/one">First <b>Source</b></a>
          <a class="result__snippet" href="https://example.org/one">A useful <b>snippet</b>.</a>
          <a class="result__a" href="https://example.org/two">Second Source</a>
          <a class="result__snippet" href="https://example.org/two">Another snippet.</a>
        </body></html>
        """
        with patch("neuron_core.search.urlopen", return_value=FakeResponse(html)):
            results = DuckDuckGoHTMLProvider().search("test query", limit=2)

        self.assertEqual([item.title for item in results], ["First Source", "Second Source"])
        self.assertIn("useful snippet", results[0].snippet)
        self.assertEqual(results[1].url, "https://example.org/two")

    def test_wikipedia_json_mapping_works_without_network(self):
        from unittest.mock import patch
        from neuron_core.search import WikipediaSearchProvider

        payload = b'{"pages":[{"title":"Earth","key":"Earth","description":"Our planet","excerpt":"Planetary details"}]}'
        with patch("neuron_core.search.urlopen", return_value=FakeResponse(payload)):
            results = WikipediaSearchProvider().search("Earth", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Earth")
        self.assertEqual(results[0].url, "https://en.wikipedia.org/wiki/Earth")
        self.assertEqual(results[0].snippet, "Our planet")
