import math

import pytest

from engine.sprt import (
    ALPHA,
    BETA,
    ELO0,
    ELO1,
    elo_to_score,
    sprt_bounds,
    sprt_decision,
    sprt_llr,
    sprt_verdict_label,
)


def test_sprt_bounds_match_formula() -> None:
    lower, upper = sprt_bounds(ALPHA, BETA)
    assert lower == pytest.approx(math.log(BETA / (1.0 - ALPHA)))
    assert upper == pytest.approx(math.log((1.0 - BETA) / ALPHA))


def test_elo_to_score_at_zero() -> None:
    assert elo_to_score(0.0) == pytest.approx(0.5)


def test_sprt_llr_all_wins_positive() -> None:
    llr = sprt_llr(50, 0, 0, elo0=ELO0, elo1=ELO1)
    lower, upper = sprt_bounds()
    assert llr > upper


def test_sprt_llr_all_losses_negative() -> None:
    llr = sprt_llr(0, 0, 50, elo0=ELO0, elo1=ELO1)
    lower, _ = sprt_bounds()
    assert llr < lower


def test_sprt_llr_empty_is_zero() -> None:
    assert sprt_llr(0, 0, 0) == 0.0


def test_sprt_decision_labels() -> None:
    lower, upper = sprt_bounds()
    assert sprt_decision(upper + 1.0, lower, upper) == "accept"
    assert sprt_decision(lower - 1.0, lower, upper) == "reject"
    assert sprt_decision(0.0, lower, upper) == "continue"


def test_sprt_verdict_label() -> None:
    assert sprt_verdict_label("accept") == "PASS"
    assert sprt_verdict_label("reject") == "FAIL"
    assert sprt_verdict_label("continue") == "INCONCLUSIVE"
