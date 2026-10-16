"""Optional lightweight contextual-bandit decision policy for Neuron.

This module does not alter graph retrieval. It learns to rank candidates already
found by retrieval from explicit reward feedback. A missing answer cannot be
recovered by a ranker; callers should keep retrieval/search fallbacks.
No third-party dependencies are required.
"""
from __future__ import annotations
import math
import random
import re
from typing import Any, Mapping, Sequence

FEATURES = ("bias", "graph_score", "lexical_score", "activation",
            "relation_score", "query_overlap", "path_compactness", "evidence_balance")

def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[^\W_]+", text, flags=re.UNICODE)}

def _clip(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.0

def candidate_features(query: str, candidate: Mapping[str, Any]) -> tuple[float, ...]:
    """Build a small, inspectable feature vector from a Neuron search result."""
    query_tokens = _tokens(query)
    name_tokens = _tokens(str(candidate.get("name", "")))
    overlap = len(query_tokens & name_tokens) / max(1, len(query_tokens))
    hops = max(0.0, float(candidate.get("hops", 0) or 0))
    support = max(0.0, float(candidate.get("support", candidate.get("positive_evidence", 0)) or 0))
    contradiction = max(0.0, float(candidate.get("contradiction", candidate.get("negative_evidence", 0)) or 0))
    balance = (support - contradiction) / max(1.0, support + contradiction)
    return (1.0, _clip(candidate.get("score", 0.0)),
            _clip(candidate.get("lexical_score", 0.0)),
            _clip(candidate.get("activation", 0.0)),
            _clip(candidate.get("relation_score", 0.0)), _clip(overlap),
            1.0 / (1.0 + hops), _clip(balance, -1.0, 1.0))

class ContextualBanditRanker:
    """Diagonal LinUCB-style ranker with explicit chosen-action rewards.

    Call observe only with verified feedback for the selected candidate.
    Positive reward means correct; negative reward means incorrect/unsupported.
    State and updates are O(number of features), with no external dependencies.
    """
    def __init__(self, *, exploration: float = 0.25, seed: int = 7):
        if exploration < 0:
            raise ValueError("exploration must be >= 0")
        self.exploration = float(exploration)
        self._precision = [1.0] * len(FEATURES)
        self._reward_sum = [0.0] * len(FEATURES)
        self.observations = 0
        self._rng = random.Random(seed)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURES

    def score(self, query: str, candidate: Mapping[str, Any], *, explore: bool = False) -> float:
        x = candidate_features(query, candidate)
        mean = sum((self._reward_sum[i] / self._precision[i]) * x[i] for i in range(len(x)))
        uncertainty = math.sqrt(sum(x[i] * x[i] / self._precision[i] for i in range(len(x))))
        return mean + (self.exploration * uncertainty if explore else 0.0)

    def rank(self, query: str, candidates: Sequence[Mapping[str, Any]], *, explore: bool = False) -> list[dict[str, Any]]:
        ranked = []
        for index, candidate in enumerate(candidates):
            row = dict(candidate)
            row["policy_score"] = round(self.score(query, row, explore=explore), 8)
            row["_stable_index"] = index
            ranked.append(row)
        ranked.sort(key=lambda row: (-row["policy_score"], row["_stable_index"]))
        for row in ranked:
            row.pop("_stable_index", None)
        return ranked

    def choose(self, query: str, candidates: Sequence[Mapping[str, Any]], *, explore: bool = True) -> int | None:
        if not candidates:
            return None
        if explore and self._rng.random() < min(1.0, self.exploration):
            return self._rng.randrange(len(candidates))
        ranked = self.rank(query, candidates, explore=explore)
        chosen_name = ranked[0].get("name")
        for index, candidate in enumerate(candidates):
            if candidate.get("name") == chosen_name:
                return index
        return 0

    def observe(self, query: str, candidate: Mapping[str, Any], reward: float) -> None:
        """Update from explicit feedback on the selected action."""
        reward = float(reward)
        if not math.isfinite(reward) or not -1.0 <= reward <= 1.0:
            raise ValueError("reward must be finite and between -1 and 1")
        x = candidate_features(query, candidate)
        for i, value in enumerate(x):
            self._precision[i] += value * value
            self._reward_sum[i] += value * reward
        self.observations += 1

    def choose_and_observe(self, query: str, candidates: Sequence[Mapping[str, Any]],
                           reward_for: Any, *, explore: bool = True) -> tuple[int | None, float | None]:
        """Use an external verified-reward oracle in controlled experiments only."""
        index = self.choose(query, candidates, explore=explore)
        if index is None:
            return None, None
        reward = float(reward_for(candidates[index]))
        self.observe(query, candidates[index], reward)
        return index, reward

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "exploration": self.exploration,
                "feature_names": list(FEATURES), "precision": list(self._precision),
                "reward_sum": list(self._reward_sum), "observations": self.observations}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1 or tuple(state.get("feature_names", ())) != FEATURES:
            raise ValueError("incompatible ranker state")
        precision, reward_sum = list(state["precision"]), list(state["reward_sum"])
        if len(precision) != len(FEATURES) or len(reward_sum) != len(FEATURES):
            raise ValueError("invalid ranker state dimensions")
        if any(float(value) <= 0 for value in precision):
            raise ValueError("precision values must be positive")
        self.exploration = float(state["exploration"])
        self._precision = [float(value) for value in precision]
        self._reward_sum = [float(value) for value in reward_sum]
        self.observations = int(state["observations"])
