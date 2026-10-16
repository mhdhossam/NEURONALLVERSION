"""Neuron v3: graph-first knowledge and bounded activation core."""
from .engine import ActivationResult, Evidence, KnowledgeObject, Neuron, NeuronGraph, Region, Synapse
from .processing import trace_processing
from .reconstruction import ReconstructedNeuron
from .reasoning import explore_reasoning
from .search import (
    DuckDuckGoHTMLProvider,
    FallbackSearchProvider,
    SearchError,
    SearchResult,
    SQLiteCachedSearchProvider,
    WikipediaSearchProvider,
    think_with_search,
)

__all__ = [
    "ActivationResult", "Evidence", "KnowledgeObject", "Neuron",
    "NeuronGraph", "Region", "Synapse", "trace_processing",
    "explore_reasoning", "DuckDuckGoHTMLProvider", "FallbackSearchProvider",
    "SearchError", "SearchResult", "SQLiteCachedSearchProvider",
    "WikipediaSearchProvider", "think_with_search",
]
