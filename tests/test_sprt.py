import math

import pytest

from engine.sprt import (
    ALPHA,
    BETA,
    ELO0,
    ELO1,
    elo_to_score,
    score_to_elo,
    sprt_bounds,
    sprt_decision,
    sprt_elo,
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


def test_sprt_llr_winning_result_not_rejected() -> None:
    # A convincing 62% score (68W-23D-37L over 128 games, ~+86 Elo) must never
    # be a FAIL. Regression guard for the earlier broken LLR formula.
    llr = sprt_llr(68, 23, 37, elo0=ELO0, elo1=ELO1)
    lower, _ = sprt_bounds()
    assert llr > lower


def test_sprt_llr_even_match_is_continue() -> None:
    llr = sprt_llr(64, 0, 64, elo0=ELO0, elo1=ELO1)
    lower, upper = sprt_bounds()
    assert lower < llr < upper


def test_score_to_elo_inverts_elo_to_score() -> None:
    assert score_to_elo(0.5) == pytest.approx(0.0)
    assert elo_to_score(score_to_elo(0.6)) == pytest.approx(0.6)


def test_sprt_elo_positive_for_winning_result() -> None:
    elo, lower, upper, los = sprt_elo(68, 23, 37)
    assert elo == pytest.approx(85.9, abs=1.0)
    assert lower < elo < upper
    assert lower > 0.0  # 95% confident the candidate is genuinely stronger
    assert los > 0.9


def test_sprt_elo_even_match_brackets_zero() -> None:
    elo, lower, upper, los = sprt_elo(64, 0, 64)
    assert elo == pytest.approx(0.0, abs=1e-6)
    assert lower < 0.0 < upper
    assert los == pytest.approx(0.5, abs=1e-6)


def test_sprt_elo_empty_is_neutral() -> None:
    elo, lower, upper, los = sprt_elo(0, 0, 0)
    assert elo == 0.0
    assert los == pytest.approx(0.5)


def test_sprt_decision_labels() -> None:
    lower, upper = sprt_bounds()
    assert sprt_decision(upper + 1.0, lower, upper) == "accept"
    assert sprt_decision(lower - 1.0, lower, upper) == "reject"
    assert sprt_decision(0.0, lower, upper) == "continue"


def test_sprt_verdict_label() -> None:
    assert sprt_verdict_label("accept") == "PASS"
    assert sprt_verdict_label("reject") == "FAIL"
    assert sprt_verdict_label("continue") == "INCONCLUSIVE"
