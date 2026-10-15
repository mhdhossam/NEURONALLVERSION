"""Neuron v3: graph-first knowledge and bounded activation core."""
from .engine import ActivationResult, Evidence, KnowledgeObject, Neuron, NeuronGraph, Region, Synapse
from .processing import trace_processing
from .verifier import verify_decision

__all__ = [
    "ActivationResult", "Evidence", "KnowledgeObject", "Neuron",
    "NeuronGraph", "Region", "Synapse", "trace_processing", "verify_decision",
]
