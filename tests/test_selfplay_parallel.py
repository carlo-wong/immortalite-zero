from engine.selfplay import _split_games


def test_split_games_even() -> None:
    assert _split_games(256, 4) == [64, 64, 64, 64]


def test_split_games_colab_two_workers() -> None:
    assert _split_games(256, 2) == [128, 128]


def test_split_games_with_remainder() -> None:
    assert sum(_split_games(10, 4)) == 10
    assert len(_split_games(10, 4)) == 4
