"""Inspectable genetic evolution of small candidate-selection rules.

This is a research baseline, not a claim of brain-like understanding. It evolves
short symbolic rules over existing candidate features; it never changes graph facts
or invents candidates. Feedback labels are external verifier labels.
"""
from __future__ import annotations
import random
from typing import Any, Mapping, Sequence
from .decision import candidate_features, FEATURES

Rule = dict[str, float | int]


class GeneticRuleRanker:
    VERSION = 1

    def __init__(self, *, population_size: int = 32, generations: int = 12,
                 max_rules: int = 6, seed: int = 29):
        if population_size < 4 or generations < 1 or max_rules < 1:
            raise ValueError("population_size >= 4, generations >= 1, max_rules >= 1 required")
        self.population_size, self.generations = population_size, generations
        self.max_rules = max_rules
        self.rng = random.Random(seed)
        self.rules: list[Rule] = []
        self.examples: list[dict[str, Any]] = []
        self.evolution_steps = 0
        self.population: list[list[Rule]] = [[]]
        while len(self.population) < population_size:
            self.population.append(self._mutate([]))

    def _new_rule(self) -> Rule:
        return {"feature": self.rng.randrange(len(FEATURES)),
                "threshold": round(self.rng.uniform(0.1, 0.9), 2),
                "direction": self.rng.choice([-1, 1]),
                "delta": round(self.rng.uniform(-1.0, 1.0), 2)}

    def _mutate(self, genome: Sequence[Rule]) -> list[Rule]:
        child = [dict(r) for r in genome]
        action = self.rng.random()
        if not child or (action < 0.45 and len(child) < self.max_rules):
            child.append(self._new_rule())
        elif action < 0.7 and child:
            child.pop(self.rng.randrange(len(child)))
        elif child:
            rule = child[self.rng.randrange(len(child))]
            field = self.rng.choice(["feature", "threshold", "direction", "delta"])
            if field == "feature":
                rule[field] = self.rng.randrange(len(FEATURES))
            elif field == "threshold":
                rule[field] = round(max(0.0, min(1.0, float(rule[field]) + self.rng.uniform(-0.25, 0.25))), 2)
            elif field == "direction":
                rule[field] = -int(rule[field])
            else:
                rule[field] = round(max(-2.0, min(2.0, float(rule[field]) + self.rng.uniform(-0.5, 0.5))), 2)
        return child[:self.max_rules]

    def score(self, query: str, candidate: Mapping[str, Any],
              genome: Sequence[Rule] | None = None) -> float:
        features = candidate_features(query, candidate)
        # Keep the existing graph score as the stable backbone.
        score = features[1] + 0.15 * features[2] + 0.1 * features[3]
        for rule in (self.rules if genome is None else genome):
            value = features[int(rule["feature"])]
            threshold = float(rule["threshold"])
            matches = value >= threshold if int(rule["direction"]) > 0 else value <= threshold
            if matches:
                score += float(rule["delta"])
        return score

    def rank(self, query: str, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for i, candidate in enumerate(candidates):
            row = dict(candidate)
            row["rule_score"] = round(self.score(query, row), 8)
            rows.append((i, row))
        rows.sort(key=lambda item: (-item[1]["rule_score"], item[0]))
        return [row for _, row in rows]

    def _fitness(self, genome: Sequence[Rule]) -> float:
        if not self.examples:
            return -0.01 * len(genome)
        correct = 0
        for e in self.examples:
            candidates = e["candidates"]
            if not candidates:
                correct += int(not e["expected"])
                continue
            best = max(range(len(candidates)), key=lambda i: (self.score(e["query"], candidates[i], genome), -i))
            correct += int(candidates[best].get("name") in e["expected"])
        # Prefer simpler rule sets when equally accurate.
        return correct / len(self.examples) - 0.003 * len(genome)

    def learn_from_mistake(self, query: str, candidates: Sequence[Mapping[str, Any]],
                           expected_names: Sequence[str]) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        expected = {str(n) for n in expected_names}
        if expected and not expected.intersection(str(c.get("name", "")) for c in candidates):
            raise ValueError("expected answer is absent from candidates; rules cannot recover it")
        self.examples.append({"query": query, "candidates": [dict(c) for c in candidates],
                              "expected": sorted(expected)})
        population = [[dict(r) for r in g] for g in self.population] + [[dict(r) for r in self.rules]]
        while len(population) < self.population_size:
            population.append(self._mutate(self.rules))
        elite_count = max(2, self.population_size // 5)
        for _ in range(self.generations):
            population.sort(key=self._fitness, reverse=True)
            elites = population[:elite_count]
            nxt = [[dict(r) for r in g] for g in elites]
            while len(nxt) < self.population_size:
                a, b = self.rng.choice(elites), self.rng.choice(elites)
                cut_a = self.rng.randrange(len(a) + 1)
                cut_b = self.rng.randrange(len(b) + 1)
                child = [dict(r) for r in a[:cut_a] + b[cut_b:]][:self.max_rules]
                nxt.append(self._mutate(child))
            population = nxt
        population.sort(key=self._fitness, reverse=True)
        self.population = population
        self.rules = [dict(r) for r in population[0]]
        self.evolution_steps += 1
        return {"examples": len(self.examples), "fitness": round(self._fitness(self.rules), 6),
                "rules": [dict(r) for r in self.rules], "evolution_steps": self.evolution_steps}

    def state_dict(self) -> dict[str, Any]:
        return {"version": self.VERSION, "feature_names": list(FEATURES),
                "population_size": self.population_size, "generations": self.generations,
                "max_rules": self.max_rules, "rules": [dict(r) for r in self.rules],
                "examples": [{"query": e["query"], "candidates": e["candidates"],
                              "expected": list(e["expected"])} for e in self.examples],
                "evolution_steps": self.evolution_steps}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != self.VERSION or tuple(state.get("feature_names", ())) != FEATURES:
            raise ValueError("incompatible rule-ranker state")
        self.population_size, self.generations = int(state["population_size"]), int(state["generations"])
        self.max_rules = int(state["max_rules"])
        self.rules = [dict(r) for r in state["rules"]]
        self.examples = [{"query": str(e["query"]), "candidates": [dict(c) for c in e["candidates"]],
                          "expected": sorted(str(x) for x in e["expected"])} for e in state["examples"]]
        self.evolution_steps = int(state["evolution_steps"])
        self.population = [[dict(r) for r in self.rules]]
        while len(self.population) < self.population_size:
            self.population.append(self._mutate(self.rules))
