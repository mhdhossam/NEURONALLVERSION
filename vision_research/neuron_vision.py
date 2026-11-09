"""NEURON Vision research prototype: graph-native, non-neural scene synthesis.

This is a deliberately small Phase-0 experiment. It converts a constrained text
prompt into a typed scene graph and renders that graph as SVG. It uses no
PyTorch, NumPy, embeddings, learned weight matrices, diffusion, or external model.

It is NOT a text-to-image foundation model and does not learn photorealistic
visual concepts. The purpose is to test graph construction, relationship
constraints, deterministic visual decoding, and inspectable generation traces.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class VisualNode:
    node_id: str
    kind: str
    label: str
    attributes: dict = field(default_factory=dict)
    activation: float = 0.0


@dataclass
class VisualEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


class VisualGraph:
    """Small typed graph with bounded, explainable activation propagation."""

    def __init__(self) -> None:
        self.nodes: dict[str, VisualNode] = {}
        self.edges: list[VisualEdge] = []

    def add_node(self, node: VisualNode) -> None:
        self.nodes[node.node_id] = node

    def connect(self, source: str, target: str, relation: str, weight: float = 1.0) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("Both graph endpoints must exist before connecting them")
        self.edges.append(VisualEdge(source, target, relation, weight))

    def activate(self, seeds: Iterable[str], hops: int = 2, damping: float = 0.65) -> list[dict]:
        frontier: dict[str, float] = {}
        for node_id in seeds:
            if node_id in self.nodes:
                frontier[node_id] = 1.0
        trace = []
        for depth in range(hops + 1):
            for node_id, value in frontier.items():
                self.nodes[node_id].activation = max(self.nodes[node_id].activation, value)
            trace.append({"hop": depth, "active": dict(sorted(frontier.items()))})
            next_frontier: dict[str, float] = {}
            for edge in self.edges:
                if edge.source in frontier:
                    value = frontier[edge.source] * edge.weight * damping
                    if value > next_frontier.get(edge.target, 0.0):
                        next_frontier[edge.target] = value
            frontier = {key: value for key, value in next_frontier.items() if value >= 0.08}
        return trace


OBJECTS = {
    "car": "vehicle", "sports car": "vehicle", "tree": "tree",
    "sun": "sun", "mountain": "mountain", "mountains": "mountain",
    "house": "house", "chair": "chair", "cloud": "cloud",
}
COLORS = {
    "red": "#c93636", "blue": "#3478c8", "green": "#3f9858",
    "yellow": "#f2c94c", "black": "#242424", "white": "#f5f5f0",
    "wooden": "#98643b", "orange": "#e78b32", "purple": "#8256a8",
}
ALIASES = {"automobile": "car", "vehicle": "car", "mountainous": "mountain"}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def compile_prompt(prompt: str) -> tuple[VisualGraph, dict, list[dict]]:
    """Compile a small natural-language subset into a typed scene graph."""
    words = tokens(prompt)
    normalized = [ALIASES.get(word, word) for word in words]
    graph = VisualGraph()
    scene = {"background": "#b9def5", "ground": "#789b57", "objects": []}
    concepts = []
    for word in normalized:
        if word in COLORS and word not in concepts:
            concepts.append(word)
    found: list[str] = []
    phrase = prompt.lower()
    for phrase_name, kind in sorted(OBJECTS.items(), key=lambda item: -len(item[0])):
        if re.search(r"\b" + re.escape(phrase_name) + r"\b", phrase) and kind not in found:
            found.append(kind)
    if not found:
        found = ["mountain", "sun", "tree"]

    graph.add_node(VisualNode("scene", "scene", "scene"))
    for index, kind in enumerate(found):
        node_id = f"object_{index}"
        color = next((COLORS[c] for c in concepts if c in phrase), None)
        if kind == "vehicle" and "red" in phrase:
            color = COLORS["red"]
        elif kind == "tree":
            color = COLORS["green"]
        elif kind == "sun":
            color = COLORS["yellow"]
        elif kind == "mountain":
            color = "#718096"
        elif kind == "house":
            color = "#d7a66b"
        elif kind == "chair":
            color = COLORS["wooden"] if "wooden" in phrase else "#3478c8"
        graph.add_node(VisualNode(node_id, "object", kind, {"color": color or "#777777"}))
        graph.connect("scene", node_id, "contains", 1.0)
        scene["objects"].append({"id": node_id, "kind": kind, "color": color or "#777777"})

    # Explicit relations are represented as graph edges, not hidden in embeddings.
    relation = "beside" if "beside" in words or "next" in words else None
    if relation and len(found) >= 2:
        graph.connect("object_0", "object_1", "beside", 1.0)
    elif "above" in words and len(found) >= 2:
        graph.connect("object_0", "object_1", "above", 1.0)
    elif "under" in words and len(found) >= 2:
        graph.connect("object_0", "object_1", "under", 1.0)

    trace = graph.activate(["scene"], hops=2)
    return graph, scene, trace


def layout_objects(kinds: list[str]) -> dict[str, tuple[int, int]]:
    """Deterministic composition rules; coordinates are scene-graph attributes."""
    defaults = {
        "sun": (54, 45), "mountain": (128, 115), "tree": (210, 155),
        "vehicle": (126, 190), "house": (205, 170), "chair": (128, 185),
        "cloud": (190, 65),
    }
    positions = {}
    for index, kind in enumerate(kinds):
        positions[kind] = defaults.get(kind, (42 + index * 35, 150))
    return positions


def render_svg(scene: dict, width: int = 256, height: int = 256) -> str:
    """Render graph-decoded primitives; SVG is the deterministic pixel decoder."""
    objects = scene["objects"]
    kinds = [item["kind"] for item in objects]
    positions = layout_objects(kinds)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{scene["background"]}"/>',
        f'<rect y="174" width="{width}" height="82" fill="{scene["ground"]}"/>',
    ]
    for item in objects:
        kind, color = item["kind"], item["color"]
        x, y = positions[kind]
        if kind == "sun":
            out.append(f'<circle cx="{x}" cy="{y}" r="23" fill="{color}"/>')
        elif kind == "mountain":
            out.append(f'<path d="M 30 180 L {x} 48 L 225 180 Z" fill="{color}"/>')
            out.append(f'<path d="M {x-18} 82 L {x} 48 L {x+18} 82 L {x+7} 76 L {x} 84 L {x-8} 76 Z" fill="#f4f5f7"/>')
        elif kind == "tree":
            out.append(f'<rect x="{x-6}" y="{y}" width="12" height="39" fill="#765033"/>')
            out.append(f'<circle cx="{x}" cy="{y-13}" r="25" fill="{color}"/>')
        elif kind == "vehicle":
            out.append(f'<rect x="{x-48}" y="{y-20}" width="96" height="29" rx="8" fill="{color}"/>')
            out.append(f'<path d="M {x-27} {y-20} L {x-12} {y-40} L {x+22} {y-40} L {x+38} {y-20} Z" fill="{color}"/>')
            out.append(f'<circle cx="{x-28}" cy="{y+10}" r="10" fill="#20252b"/><circle cx="{x+29}" cy="{y+10}" r="10" fill="#20252b"/>')
            out.append(f'<circle cx="{x-28}" cy="{y+10}" r="4" fill="#cbd5e0"/><circle cx="{x+29}" cy="{y+10}" r="4" fill="#cbd5e0"/>')
        elif kind == "house":
            out.append(f'<rect x="{x-27}" y="{y-35}" width="54" height="48" fill="{color}"/>')
            out.append(f'<path d="M {x-35} {y-35} L {x} {y-65} L {x+35} {y-35} Z" fill="#8b4141"/>')
            out.append(f'<rect x="{x-7}" y="{y-7}" width="14" height="20" fill="#69452e"/>')
        elif kind == "chair":
            out.append(f'<rect x="{x-24}" y="{y-34}" width="48" height="10" fill="{color}"/>')
            out.append(f'<rect x="{x-24}" y="{y-58}" width="10" height="27" fill="{color}"/>')
            out.append(f'<path d="M {x-19} {y-24} L {x-24} {y+15} M {x+19} {y-24} L {x+24} {y+15}" stroke="{color}" stroke-width="6"/>')
        elif kind == "cloud":
            out.append(f'<path d="M {x-32} {y+8} a16 16 0 0 1 8-30 a21 21 0 0 1 40 4 a14 14 0 0 1 5 26 Z" fill="#ffffff"/>')
    out.append("</svg>")
    return "\n".join(out)


def generate(prompt: str, output: Path) -> dict:
    graph, scene, activation_trace = compile_prompt(prompt)
    svg = render_svg(scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    report = {
        "prompt": prompt,
        "method": "typed graph construction + deterministic graph-to-SVG decoding",
        "neural_backbone": None,
        "matrix_based_learned_inference": False,
        "learned_from_dataset": False,
        "nodes": [asdict(node) for node in graph.nodes.values()],
        "edges": [asdict(edge) for edge in graph.edges],
        "activation_trace": activation_trace,
        "output": str(output),
        "limitations": [
            "constrained vocabulary and hand-authored object renderers",
            "no learned visual concepts or photorealistic texture synthesis",
            "not a replacement for Qwen or a general text-to-image model",
        ],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="A constrained scene description")
    parser.add_argument("--output", default="neuron_vision.svg")
    args = parser.parse_args()
    report = generate(args.prompt, Path(args.output))
    print(json.dumps({key: report[key] for key in ("method", "neural_backbone", "matrix_based_learned_inference", "output", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
