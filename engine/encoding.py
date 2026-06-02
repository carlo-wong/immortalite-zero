"""Board and move encoding.

Input encoding: 20 planes of 8x8 in side-to-move canonical orientation.
  - 12 planes: piece type x colour (P N B R Q K us, then them)
  -  4 planes: castling rights (us-K us-Q them-K them-Q)
  -  1 plane : en-passant target square
  -  2 planes: repetition flags (>=2 occurrences, >=3 occurrences)
  -  1 plane : halfmove clock (normalized to [0, 1], broadcast)

Move encoding: the AlphaZero 8x8x73 = 4672 scheme.
  from_square * 73 + plane, where plane is one of:
    0..55  queen-like moves: 8 directions x 7 distances
    56..63 knight moves
    64..72 underpromotions: 3 file-directions x {knight, bishop, rook}
"""

from __future__ import annotations

import chess
import numpy as np

NUM_INPUT_PLANES = 20
POLICY_SIZE = 64 * 73  # 4672
ENCODING_VERSION = 2

# 8 sliding directions as (d_rank, d_file), ordered N, NE, E, SE, S, SW, W, NW.
_QUEEN_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_QUEEN_DIR_INDEX = {d: i for i, d in enumerate(_QUEEN_DIRS)}

# 8 knight offsets as (d_rank, d_file).
_KNIGHT_OFFSETS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
_KNIGHT_INDEX = {d: i for i, d in enumerate(_KNIGHT_OFFSETS)}

# Underpromotion pieces in fixed order.
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_UNDERPROMO_INDEX = {p: i for i, p in enumerate(_UNDERPROMO_PIECES)}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a board as a (20, 8, 8) float32 tensor."""
    planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    # Canonical orientation: side-to-move is always "white" after mirroring.
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


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Map a legal move to its index in [0, POLICY_SIZE)."""
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(move.from_square)
        to_sq = chess.square_mirror(move.to_square)
    else:
        from_sq = move.from_square
        to_sq = move.to_square
    fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)
    tr, tf = chess.square_rank(to_sq), chess.square_file(to_sq)
    d_rank, d_file = tr - fr, tf - ff

    # Underpromotions (queen promotions fall through to the sliding encoding).
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # file direction: -1, 0, +1 -> index 0, 1, 2
        dir_idx = _sign(d_file) + 1
        piece_idx = _UNDERPROMO_INDEX[move.promotion]
        plane = 64 + dir_idx * 3 + piece_idx
        return from_sq * 73 + plane

    # Knight moves.
    if (d_rank, d_file) in _KNIGHT_INDEX:
        plane = 56 + _KNIGHT_INDEX[(d_rank, d_file)]
        return from_sq * 73 + plane

    # Sliding / king / pawn / queen-promotion moves.
    step = (_sign(d_rank), _sign(d_file))
    distance = max(abs(d_rank), abs(d_file))
    dir_idx = _QUEEN_DIR_INDEX[step]
    plane = dir_idx * 7 + (distance - 1)
    return from_sq * 73 + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """Inverse of :func:`move_to_index`. Returns None if the move is illegal."""
    from_sq_canonical = index // 73
    plane = index % 73
    fr, ff = chess.square_rank(from_sq_canonical), chess.square_file(from_sq_canonical)

    promotion = None
    if plane < 56:  # sliding
        dir_idx = plane // 7
        distance = (plane % 7) + 1
        d_rank, d_file = _QUEEN_DIRS[dir_idx]
        tr, tf = fr + d_rank * distance, ff + d_file * distance
    elif plane < 64:  # knight
        d_rank, d_file = _KNIGHT_OFFSETS[plane - 56]
        tr, tf = fr + d_rank, ff + d_file
    else:  # underpromotion
        p = plane - 64
        dir_idx = p // 3
        piece_idx = p % 3
        d_file = dir_idx - 1
        # Canonical frame always has side-to-move as white.
        d_rank = 1
        tr, tf = fr + d_rank, ff + d_file
        promotion = _UNDERPROMO_PIECES[piece_idx]

    if not (0 <= tr <= 7 and 0 <= tf <= 7):
        return None

    to_sq_canonical = chess.square(tf, tr)
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq_canonical)
        to_sq = chess.square_mirror(to_sq_canonical)
    else:
        from_sq = from_sq_canonical
        to_sq = to_sq_canonical

    # Infer queen promotion for pawn pushes/captures reaching the last rank.
    if promotion is None:
        piece = board.piece_at(from_sq)
        to_rank = chess.square_rank(to_sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_rank == 7) or (
                piece.color == chess.BLACK and to_rank == 0
            ):
                promotion = chess.QUEEN

    move = chess.Move(from_sq, to_sq, promotion=promotion)
    return move if move in board.legal_moves else None


def legal_move_indices(board: chess.Board) -> dict[int, chess.Move]:
    """Map every legal move's index -> move for the given board."""
    return {move_to_index(m, board): m for m in board.legal_moves}
