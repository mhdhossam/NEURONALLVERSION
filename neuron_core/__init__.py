"""Neuron v3: graph-first knowledge and bounded activation core."""
from .engine import ActivationResult, Evidence, KnowledgeObject, Neuron, NeuronGraph, Region, Synapse
from .processing import trace_processing
from .reasoning import explore_reasoning

__all__ = [
    "ActivationResult", "Evidence", "KnowledgeObject", "Neuron",
    "NeuronGraph", "Region", "Synapse", "trace_processing",
    "explore_reasoning",
]
