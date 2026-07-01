"""Sequential Probability Ratio Test (SPRT) for strength gates.

Uses the Fishtest-style Gaussian GSPRT on win/draw/loss results (scored
1 / 0.5 / 0) with empirical variance, matching common chess engine gate
practice (github.com/official-stockfish/fishtest). The gate also reports a
logistic Elo estimate with a 95% confidence interval so a "pass" says *how
much* stronger the candidate is, not just that it crossed a threshold.
"""

from __future__ import annotations

import math
from statistics import NormalDist

# Shared defaults for auto and manual gates.
ELO0 = 0.0
ELO1 = 25.0
ALPHA = 0.05
BETA = 0.05

# Two-sided z for the reported Elo confidence interval (95% by default).
CONFIDENCE = 0.95

_NORM = NormalDist()


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    """Inverse of elo_to_score: logistic Elo for a [0,1] score."""
    if score <= 0.0:
        return -float("inf")
    if score >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


def sprt_bounds(alpha: float = ALPHA, beta: float = BETA) -> tuple[float, float]:
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    return lower, upper


def _score_mean_var(wins: int, draws: int, losses: int) -> tuple[int, float, float]:
    """Return (n, mean_score, empirical_variance) for a W/D/L trinomial."""
    n = wins + draws + losses
    if n == 0:
        return 0, 0.0, 0.0
    total_score = wins + 0.5 * draws
    mean = total_score / n
    var = (
        wins * (1.0 - mean) ** 2
        + draws * (0.5 - mean) ** 2
        + losses * (0.0 - mean) ** 2
    ) / n
    return n, mean, var


def sprt_llr(
    wins: int,
    draws: int,
    losses: int,
    elo0: float = ELO0,
    elo1: float = ELO1,
) -> float:
    """Fishtest-style Gaussian GSPRT log-likelihood ratio.

    LLR = n / (2 * var) * [ (mean - mu0)^2 - (mean - mu1)^2 ]

    where mean is the empirical score, var its empirical variance, and
    mu0/mu1 are the scores implied by elo0/elo1. Positive LLR favours H1
    (candidate is elo1 stronger); it crosses the upper bound to accept and
    the lower bound to reject.
    """
    n, mean, var = _score_mean_var(wins, draws, losses)
    if n == 0:
        return 0.0

    mu0 = elo_to_score(elo0)
    mu1 = elo_to_score(elo1)

    if var < 1e-12:
        # Empirical variance vanishes when all scores are identical; use a
        # Bernoulli-style floor so decisive runs still accumulate LLR.
        var = max(mu0 * (1.0 - mu0), mu1 * (1.0 - mu1), 0.01)

    return n / (2.0 * var) * ((mean - mu0) ** 2 - (mean - mu1) ** 2)


def sprt_elo(
    wins: int,
    draws: int,
    losses: int,
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float, float]:
    """Logistic Elo estimate with a confidence interval and LOS.

    Returns (elo, elo_lower, elo_upper, los):
      - elo: point estimate of the candidate's Elo advantage.
      - elo_lower / elo_upper: two-sided ``confidence`` interval (95% default),
        so elo_lower is the improvement we are (100*confidence)% sure exceeds.
      - los: likelihood of superiority, P(elo > 0), normal approximation.
    """
    n, mean, var = _score_mean_var(wins, draws, losses)
    if n == 0:
        return 0.0, float("nan"), float("nan"), 0.5

    se = math.sqrt(var / n) if var > 0.0 else 0.0
    elo = score_to_elo(mean)

    if se == 0.0:
        los = 1.0 if mean > 0.5 else 0.0 if mean < 0.5 else 0.5
        return elo, elo, elo, los

    z = _NORM.inv_cdf(0.5 + confidence / 2.0)
    lower_score = min(max(mean - z * se, 0.0), 1.0)
    upper_score = min(max(mean + z * se, 0.0), 1.0)
    los = _NORM.cdf((mean - 0.5) / se)
    return elo, score_to_elo(lower_score), score_to_elo(upper_score), los


def sprt_decision(llr: float, lower: float, upper: float) -> str:
    if llr >= upper:
        return "accept"
    if llr <= lower:
        return "reject"
    return "continue"


def sprt_verdict_label(decision: str) -> str:
    if decision == "accept":
        return "PASS"
    if decision == "reject":
        return "FAIL"
    return "INCONCLUSIVE"
