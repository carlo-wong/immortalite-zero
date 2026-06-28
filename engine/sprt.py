"""Sequential Probability Ratio Test (SPRT) for strength gates.

Uses a Gaussian score SPRT on win/draw/loss results (1 / 0.5 / 0) with
empirical variance (GSPRT-style), matching common chess engine gate practice.
"""

from __future__ import annotations

import math

# Shared defaults for auto and manual gates.
ELO0 = 0.0
ELO1 = 25.0
ALPHA = 0.05
BETA = 0.05


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_bounds(alpha: float = ALPHA, beta: float = BETA) -> tuple[float, float]:
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    return lower, upper


def sprt_llr(
    wins: int,
    draws: int,
    losses: int,
    elo0: float = ELO0,
    elo1: float = ELO1,
) -> float:
    n = wins + draws + losses
    if n == 0:
        return 0.0

    total_score = wins + 0.5 * draws
    mean = total_score / n
    mu0 = elo_to_score(elo0)
    mu1 = elo_to_score(elo1)

    var = (
        wins * (1.0 - mean) ** 2
        + draws * (0.5 - mean) ** 2
        + losses * (0.0 - mean) ** 2
    ) / n
    if var < 1e-12:
        # Empirical variance vanishes when all scores are identical; use a
        # Bernoulli-style floor so decisive runs still accumulate LLR.
        var = max(mu0 * (1.0 - mu0), mu1 * (1.0 - mu1), 0.01)

    return (
        (mu1 - mu0) / var * (2.0 * total_score - n * (mu0 + mu1))
        + n * (mu0 * mu0 - mu1 * mu1) / (2.0 * var)
    )


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
