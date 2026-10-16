"""Small genetic learner for Neuron candidate ranking.

This isolated experiment evolves scoring genomes through selection, crossover, and
mutation against verified examples. Retrieval and stored graph facts are unchanged.
"""
from __future__ import annotations
import math
import random
from typing import Any, Mapping, Sequence
from .decision import candidate_features, FEATURES


class GeneticRanker:
    """Evolve a compact candidate-ranking genome after verifier feedback."""
    VERSION = 1

    def __init__(self, *, population_size: int = 36, generations: int = 16,
                 mutation_rate: float = 0.25, mutation_scale: float = 0.35, seed: int = 17):
        if population_size < 4:
            raise ValueError("population_size must be >= 4")
        if generations < 1:
            raise ValueError("generations must be >= 1")
        if not 0 < mutation_rate <= 1 or mutation_scale <= 0:
            raise ValueError("invalid mutation settings")
        self.population_size, self.generations = int(population_size), int(generations)
        self.mutation_rate, self.mutation_scale = float(mutation_rate), float(mutation_scale)
        self._rng = random.Random(seed)
        # Begin with graph-score-first behavior rather than a randomly initialized model.
        self.genome = [0.0, 1.0, 0.35, 0.15, 0.25, 0.15, 0.10, 0.10]
        self.examples: list[dict[str, Any]] = []
        self.evolution_steps = 0
        self._population = [list(self.genome)]
        while len(self._population) < self.population_size:
            self._population.append(self._mutate(self.genome))

    def score(self, query: str, candidate: Mapping[str, Any],
              genome: Sequence[float] | None = None) -> float:
        weights = self.genome if genome is None else genome
        return sum(float(w) * x for w, x in zip(weights, candidate_features(query, candidate)))

    def rank(self, query: str, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for i, candidate in enumerate(candidates):
            row = dict(candidate)
            row["genetic_score"] = round(self.score(query, row), 8)
            rows.append((i, row))
        rows.sort(key=lambda pair: (-pair[1]["genetic_score"], pair[0]))
        return [row for _, row in rows]

    def _mutate(self, genome: Sequence[float]) -> list[float]:
        child = list(genome)
        for i in range(len(child)):
            if self._rng.random() < self.mutation_rate:
                child[i] = max(-3.0, min(3.0, child[i] + self._rng.gauss(0, self.mutation_scale)))
        return child

    def _fitness(self, genome: Sequence[float]) -> float:
        if not self.examples:
            return -0.001 * sum(w * w for w in genome)
        correct = 0
        for example in self.examples:
            candidates = example["candidates"]
            if not candidates:
                correct += int(not example["expected"])
                continue
            best = max(range(len(candidates)),
                       key=lambda i: (self.score(example["query"], candidates[i], genome), -i))
            correct += int(candidates[best].get("name") in example["expected"])
        return correct / len(self.examples) - 0.0005 * sum(w * w for w in genome)

    def learn_from_mistake(self, query: str, candidates: Sequence[Mapping[str, Any]],
                           expected_names: Sequence[str]) -> dict[str, Any]:
        """Record post-attempt verifier feedback, then evolve the ranking genome.

        Expected labels must never be passed to rank before the measured first attempt.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        expected = {str(name) for name in expected_names}
        names = {str(c.get("name", "")) for c in candidates}
        if expected and not names.intersection(expected):
            raise ValueError("expected answer is absent from candidates; ranking cannot recover it")
        self.examples.append({"query": query, "candidates": [dict(c) for c in candidates],
                              "expected": sorted(expected)})
        population = list(self._population) + [list(self.genome)]
        while len(population) < self.population_size:
            population.append(self._mutate(self.genome))
        for _ in range(self.generations):
            population.sort(key=self._fitness, reverse=True)
            elites = population[:max(2, self.population_size // 6)]
            next_population = [list(g) for g in elites]
            while len(next_population) < self.population_size:
                a, b = self._rng.choice(elites), self._rng.choice(elites)
                child = [a[i] if self._rng.random() < 0.5 else b[i] for i in range(len(a))]
                next_population.append(self._mutate(child))
            population = next_population
        population.sort(key=self._fitness, reverse=True)
        self._population, self.genome = population, list(population[0])
        self.evolution_steps += 1
        return {"examples": len(self.examples), "fitness": round(self._fitness(self.genome), 6),
                "genome": list(self.genome), "evolution_steps": self.evolution_steps}

    def state_dict(self) -> dict[str, Any]:
        return {"version": self.VERSION, "feature_names": list(FEATURES),
                "population_size": self.population_size, "generations": self.generations,
                "mutation_rate": self.mutation_rate, "mutation_scale": self.mutation_scale,
                "genome": list(self.genome),
                "examples": [{"query": e["query"], "candidates": e["candidates"],
                              "expected": list(e["expected"])} for e in self.examples],
                "evolution_steps": self.evolution_steps}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != self.VERSION or tuple(state.get("feature_names", ())) != FEATURES:
            raise ValueError("incompatible genetic ranker state")
        genome = [float(x) for x in state["genome"]]
        if len(genome) != len(FEATURES) or not all(math.isfinite(x) for x in genome):
            raise ValueError("invalid genome")
        self.population_size, self.generations = int(state["population_size"]), int(state["generations"])
        self.mutation_rate, self.mutation_scale = float(state["mutation_rate"]), float(state["mutation_scale"])
        self.genome = genome
        self.examples = [{"query": str(e["query"]), "candidates": [dict(c) for c in e["candidates"]],
                          "expected": sorted(str(x) for x in e["expected"])} for e in state["examples"]]
        self.evolution_steps = int(state["evolution_steps"])
        self._population = [list(genome)]
        while len(self._population) < self.population_size:
            self._population.append(self._mutate(genome))
