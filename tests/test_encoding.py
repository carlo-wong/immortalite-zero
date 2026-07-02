"""Round-trip test: every legal move must survive move_to_index -> index_to_move."""

import random

import chess
import numpy as np

from engine.encoding import (
    NUM_INPUT_PLANES,
    POLICY_SIZE,
    board_to_planes,
    fill_planes_batch,
    index_to_move,
    legal_move_indices,
    move_to_index,
)


def _check_position(board: chess.Board) -> None:
    planes = board_to_planes(board)
    assert planes.shape == (NUM_INPUT_PLANES, 8, 8)

    for move in board.legal_moves:
        idx = move_to_index(move, board)
        assert 0 <= idx < POLICY_SIZE, f"index out of range for {move}"
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


def test_canonical_mirror_equivalence() -> None:
    board = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        board.push_uci(uci)
    mirrored = board.mirror()

    planes = board_to_planes(board)
    mirrored_planes = board_to_planes(mirrored)
    assert np.array_equal(planes, mirrored_planes)


def test_repetition_and_halfmove_planes() -> None:
    rep2 = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        rep2.push_uci(uci)
    planes_rep2 = board_to_planes(rep2)
    assert np.all(planes_rep2[17] == 1.0)
    assert np.all(planes_rep2[18] == 0.0)

    rep3 = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        rep3.push_uci(uci)
    planes_rep3 = board_to_planes(rep3)
    assert np.all(planes_rep3[17] == 1.0)
    assert np.all(planes_rep3[18] == 1.0)

    board_99 = chess.Board("7k/8/8/8/8/8/8/KR6 w - - 99 1")
    planes_99 = board_to_planes(board_99)
    assert np.allclose(planes_99[19], 0.99)

    board_150 = chess.Board("7k/8/8/8/8/8/8/KR6 w - - 150 1")
    planes_150 = board_to_planes(board_150)
    assert np.all(planes_150[19] == 1.0)


def _board_to_planes_via_mirror(board: chess.Board) -> np.ndarray:
    """Reference encoder using board.mirror() (pre-optimization path)."""
    planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    canonical = board if board.turn == chess.WHITE else board.mirror()
    for square, piece in canonical.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        planes[plane, rank, file] = 1.0
    if canonical.has_kingside_castling_rights(chess.WHITE):
        planes[12, :, :] = 1.0
    if canonical.has_queenside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if canonical.has_kingside_castling_rights(chess.BLACK):
        planes[14, :, :] = 1.0
    if canonical.has_queenside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if canonical.ep_square is not None:
        rank = chess.square_rank(canonical.ep_square)
        file = chess.square_file(canonical.ep_square)
        planes[16, rank, file] = 1.0
    if board.is_repetition(2):
        planes[17, :, :] = 1.0
    if board.is_repetition(3):
        planes[18, :, :] = 1.0
    halfmove_norm = min(float(board.halfmove_clock) / 100.0, 1.0)
    planes[19, :, :] = halfmove_norm
    return planes


def test_board_to_planes_matches_mirror_reference() -> None:
    random.seed(1)
    for _ in range(500):
        board = chess.Board()
        for _ in range(random.randint(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        assert np.array_equal(board_to_planes(board), _board_to_planes_via_mirror(board))


def test_fill_planes_batch_matches_board_to_planes() -> None:
    random.seed(2)
    boards = [chess.Board()]
    b = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        b.push_uci(uci)
    boards.append(b)
    out = np.zeros((len(boards), NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    fill_planes_batch(boards, out)
    for i, board in enumerate(boards):
        assert np.array_equal(out[i], board_to_planes(board))


if __name__ == "__main__":
    test_random_selfplay_positions()
