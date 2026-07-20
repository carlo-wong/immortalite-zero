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


def _write_board_planes_ref(board: chess.Board, planes: np.ndarray) -> None:
    """Reference implementation (piece_map loop). Kept for equivalence tests."""
    planes.fill(0.0)
    if board.turn == chess.WHITE:
        for square, piece in board.piece_map().items():
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            planes[plane, rank, file] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[15, :, :] = 1.0
        ep_square = board.ep_square
        if ep_square is not None:
            planes[16, chess.square_rank(ep_square), chess.square_file(ep_square)] = 1.0
    else:
        # Canonical frame without board.mirror(): square_mirror + color swap.
        for square, piece in board.piece_map().items():
            canon_sq = chess.square_mirror(square)
            rank = chess.square_rank(canon_sq)
            file = chess.square_file(canon_sq)
            plane = (piece.piece_type - 1) + (0 if piece.color == chess.BLACK else 6)
            planes[plane, rank, file] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[15, :, :] = 1.0
        ep_square = board.ep_square
        if ep_square is not None:
            ep_m = chess.square_mirror(ep_square)
            planes[16, chess.square_rank(ep_m), chess.square_file(ep_m)] = 1.0

    if board.is_repetition(2):
        planes[17, :, :] = 1.0
    if board.is_repetition(3):
        planes[18, :, :] = 1.0
    halfmove_norm = min(float(board.halfmove_clock) / 100.0, 1.0)
    planes[19, :, :] = halfmove_norm


def _write_board_planes(board: chess.Board, planes: np.ndarray) -> None:
    """Write canonical side-to-move planes into ``planes`` (20, 8, 8).

    Uses bitboard operations to avoid a Python loop over piece_map().
    Layout: unpackbits(bb.to_bytes(8,'little'), bitorder='little').reshape(8,8)
    gives planes[rank, file] = 1 for each occupied square (rank*8+file = square
    index).  For black-to-move canonical frame, square_mirror is a vertical rank
    flip: arr[::-1, :].
    """
    planes.fill(0.0)

    piece_bbs = (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
    )
    occ_us = board.occupied_co[board.turn]
    occ_them = board.occupied_co[not board.turn]

    # Pack 12 bitboards into a uint64 array; view as uint8 bytes (little-endian
    # on x86) then unpack all piece planes in one np.unpackbits call.
    # Assigning uint8 piece_planes to float32 planes[] is cast in C by numpy.
    bbs = np.empty(12, dtype=np.uint64)
    for i, bb in enumerate(piece_bbs):
        bbs[i] = bb & occ_us
        bbs[i + 6] = bb & occ_them

    piece_planes = (
        np.unpackbits(bbs.view(np.uint8).reshape(12, 8), axis=1, bitorder="little")
        .reshape(12, 8, 8)
    )

    if board.turn == chess.WHITE:
        planes[:12] = piece_planes
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[12] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[13] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[14] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[15] = 1.0
        ep_square = board.ep_square
        if ep_square is not None:
            planes[16, chess.square_rank(ep_square), chess.square_file(ep_square)] = 1.0
    else:
        # square_mirror flips ranks: apply vertical flip to all 12 piece planes.
        planes[:12] = piece_planes[:, ::-1, :]
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[12] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[13] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[14] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[15] = 1.0
        ep_square = board.ep_square
        if ep_square is not None:
            ep_m = chess.square_mirror(ep_square)
            planes[16, chess.square_rank(ep_m), chess.square_file(ep_m)] = 1.0

    # is_repetition(3) implies is_repetition(2); coalesce to one history walk
    # in the common (no-rep) case and at most one when threefold holds.
    if board.is_repetition(3):
        planes[17] = 1.0
        planes[18] = 1.0
    elif board.is_repetition(2):
        planes[17] = 1.0
    halfmove_norm = min(float(board.halfmove_clock) / 100.0, 1.0)
    planes[19] = halfmove_norm


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a board as a (20, 8, 8) float32 tensor."""
    planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    _write_board_planes(board, planes)
    return planes


def fill_planes_batch(boards: list[chess.Board], out: np.ndarray) -> None:
    """Fill ``out[i]`` with planes for ``boards[i]``. ``out`` shape (N, 20, 8, 8)."""
    for i, board in enumerate(boards):
        _write_board_planes(board, out[i])


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
