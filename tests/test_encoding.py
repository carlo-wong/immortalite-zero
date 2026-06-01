"""Round-trip test: every legal move must survive move_to_index -> index_to_move."""

import random

import chess

from engine.encoding import board_to_planes, index_to_move, legal_move_indices, move_to_index


def _check_position(board: chess.Board) -> None:
    planes = board_to_planes(board)
    assert planes.shape == (18, 8, 8)

    for move in board.legal_moves:
        idx = move_to_index(move, board)
        assert 0 <= idx < 64 * 73, f"index out of range for {move}"
        recovered = index_to_move(idx, board)
        assert recovered == move, f"round-trip failed: {move} -> {idx} -> {recovered}"

    # legal_move_indices must be a bijection over legal moves
    mapping = legal_move_indices(board)
    assert len(mapping) == board.legal_moves.count()


def test_random_selfplay_positions():
    random.seed(0)
    failures = 0
    for game in range(200):
        board = chess.Board()
        for _ in range(random.randint(0, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            _check_position(board)
            board.push(random.choice(moves))
    print("OK: round-trip passed on random positions")


if __name__ == "__main__":
    test_random_selfplay_positions()
