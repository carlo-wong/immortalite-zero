"""Round-trip test: every legal move must survive move_to_index -> index_to_move."""

import random

import chess
import numpy as np

from engine.encoding import (
    NUM_INPUT_PLANES,
    POLICY_SIZE,
    _write_board_planes_ref,
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


def test_repetition_planes_coalesce_matches_two_separate_calls() -> None:
    """Coalesced repetition logic must match two separate is_repetition calls."""
    random.seed(42)
    for _ in range(300):
        board = chess.Board()
        for _ in range(random.randint(0, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        planes = board_to_planes(board)
        p17_ref = 1.0 if board.is_repetition(2) else 0.0
        p18_ref = 1.0 if board.is_repetition(3) else 0.0
        assert float(planes[17, 0, 0]) == p17_ref, f"plane 17 mismatch: {board.fen()}"
        assert float(planes[18, 0, 0]) == p18_ref, f"plane 18 mismatch: {board.fen()}"


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


def test_bitboard_encoding_equivalence_random() -> None:
    """Property test: bitboard impl must exactly match reference loop over ~40 games."""
    rng = random.Random(42)
    ref = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    new = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    positions_tested = 0

    for _ in range(40):
        board = chess.Board()
        for _ in range(120):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            _write_board_planes_ref(board, ref)
            board_to_planes(board)  # exercise public API too
            np.copyto(new, board_to_planes(board))
            assert np.array_equal(new, ref), (
                f"Mismatch at fen={board.fen()}\n"
                f"differing plane indices: {np.unique(np.where(new != ref)[0]).tolist()}"
            )
            positions_tested += 1
            board.push(rng.choice(moves))

    assert positions_tested > 100, f"too few positions tested: {positions_tested}"


def test_bitboard_encoding_special_positions() -> None:
    """Hand-picked edge case positions for the bitboard encoder."""
    ref = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    new = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)

    def check(board: chess.Board) -> None:
        _write_board_planes_ref(board, ref)
        np.copyto(new, board_to_planes(board))
        assert np.array_equal(new, ref), (
            f"Mismatch at {board.fen()}\n"
            f"differing planes: {np.unique(np.where(new != ref)[0]).tolist()}"
        )

    # En-passant available with black to move (after 1.e4).
    board = chess.Board()
    board.push_uci("e2e4")
    assert board.turn == chess.BLACK and board.ep_square is not None
    check(board)
    # ep square e3 must appear at its square_mirror position (e6 = rank 5, file 4).
    np.copyto(new, board_to_planes(board))
    assert new[16, 5, 4] == 1.0, f"ep plane wrong: {new[16]}"
    assert new[16].sum() == 1.0

    # En-passant available with black to move, pawn actually able to capture.
    # After 1.e4 d5 2.e5 f5: white ep target is f6, it's white to move — skip.
    # Build a FEN where it's black to move with ep reachable: after 1.d4 e5 2.d5 c5.
    b2 = chess.Board()
    for uci in ("d2d4", "e7e5", "d4d5", "c7c5"):
        b2.push_uci(uci)
    assert b2.turn == chess.WHITE  # white to move, ep = c6
    check(b2)
    # Black to move with an ep target set (planted via FEN).
    b5 = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3")
    assert b5.turn == chess.BLACK and b5.ep_square == chess.E3
    check(b5)
    np.copyto(new, board_to_planes(b5))
    # e3 mirrored -> e6 = rank 5, file 4
    assert new[16, 5, 4] == 1.0

    # Threefold repetition flags set.
    board_rep = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        board_rep.push_uci(uci)
    assert board_rep.is_repetition(3)
    check(board_rep)
    np.copyto(new, board_to_planes(board_rep))
    assert np.all(new[17] == 1.0) and np.all(new[18] == 1.0)

    # Halfmove clock > 100, clamped to 1.0.
    board_clk = chess.Board("7k/8/8/8/8/8/8/KR6 w - - 150 1")
    check(board_clk)
    np.copyto(new, board_to_planes(board_clk))
    assert np.all(new[19] == 1.0)

    # Promotion-heavy endgame (multiple queens, bishops, rooks).
    board_promo = chess.Board("1QQ5/2Q5/8/8/8/5q2/8/k1K5 w - - 0 1")
    check(board_promo)

    # Castling rights: one side lost kingside but kept queenside.
    board_cast = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1")
    check(board_cast)
    board_cast.push_uci("e8g8")  # black castles kingside -> loses both rights
    check(board_cast)


def test_bitboard_batch_equivalence_random() -> None:
    """fill_planes_batch must match per-board board_to_planes across 50 random boards."""
    rng = random.Random(99)
    boards: list[chess.Board] = []
    b = chess.Board()
    for _ in range(50):
        moves = list(b.legal_moves)
        if not moves or b.is_game_over():
            b = chess.Board()
        b = b.copy()
        b.push(rng.choice(list(b.legal_moves)))
        boards.append(b.copy())

    out = np.zeros((len(boards), NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    fill_planes_batch(boards, out)
    for i, board in enumerate(boards):
        expected = board_to_planes(board)
        assert np.array_equal(out[i], expected), f"batch mismatch at board {i}: {board.fen()}"


if __name__ == "__main__":
    test_random_selfplay_positions()
