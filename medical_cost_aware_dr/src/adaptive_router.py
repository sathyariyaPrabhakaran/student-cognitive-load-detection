from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score


@dataclass
class RouterDecision:
    escalate: np.ndarray
    score: np.ndarray
    threshold: float


def uncertainty_features(probabilities: np.ndarray, quality: np.ndarray | None = None) -> np.ndarray:
    """Build compact, model-agnostic routing features from class probabilities."""
    p = np.clip(probabilities, 1e-8, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1)
    sorted_p = np.sort(p, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2] if p.shape[1] > 1 else sorted_p[:, -1]
    max_prob = p.max(axis=1)
    feats = [max_prob, entropy, margin]
    if quality is not None:
        feats.append(np.asarray(quality).reshape(-1))
    return np.column_stack(feats)


class AdaptiveRouter:
    """Learn whether a lightweight prediction should be escalated.

    The router learns from validation-set outcomes: escalation is useful when
    the lightweight model is wrong and the expert model is correct. A separate
    operating threshold is then selected to satisfy a minimum sensitivity.
    """

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
        self.threshold = 0.5

    def fit(self, light_prob: np.ndarray, light_pred: np.ndarray,
            expert_pred: np.ndarray, y_true: np.ndarray,
            quality: np.ndarray | None = None,
            min_sensitivity: float = 0.90):
        x = uncertainty_features(light_prob, quality)
        # Target 1 means escalation is beneficial: the expert is right while
        # the lightweight model is wrong. If both agree, escalation is not useful.
        useful = ((light_pred != y_true) & (expert_pred == y_true)).astype(int)
        self.model.fit(x, useful)
        score = self.model.predict_proba(x)[:, 1]
        candidates = np.unique(np.r_[np.linspace(0.05, 0.95, 91), score])
        best = 0.5
        best_escalation = 1.0
        for threshold in candidates:
            escalate = score >= threshold
            final = np.where(escalate, expert_pred, light_pred)
            sensitivity = recall_score(y_true, final, average="macro", zero_division=0)
            if sensitivity >= min_sensitivity and escalate.mean() < best_escalation:
                best = float(threshold)
                best_escalation = float(escalate.mean())
        self.threshold = best
        return self

    def decide(self, light_prob: np.ndarray, quality: np.ndarray | None = None) -> RouterDecision:
        x = uncertainty_features(light_prob, quality)
        score = self.model.predict_proba(x)[:, 1]
        return RouterDecision(score >= self.threshold, score, self.threshold)
