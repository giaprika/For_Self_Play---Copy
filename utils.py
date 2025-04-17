import os
import numpy as np
import chess
import torch

def get_filename_without_extension(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def get_loss_from_model_name(file_name):
    try:
        loss_str = file_name.split('_')[-1]
        return float(loss_str)
    except ValueError:
        raise ValueError(f"Filename '{file_name}' does not contain valid loss value.")

def board_to_tensor(board):
    tensor = np.zeros((20, 8, 8), dtype=np.float32)
    
    # Piece channels
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            offset = 0 if piece.color == chess.WHITE else 6
            channel = piece_map[piece.piece_type] + offset
            row, col = divmod(square, 8)
            tensor[channel, row, col] = 1

    # Repetition history
    tensor[12, :, :] = float(board.is_repetition(2))
    tensor[13, :, :] = float(board.is_repetition(3))

    
    # Castling rights
    tensor[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[15, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    tensor[17, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    
    # En passant
    if board.ep_square:
        row, col = divmod(board.ep_square, 8)
        tensor[18, row, col] = 1.0

    
    # Turn
    tensor[19, :, :] = 1 if board.turn == chess.WHITE else 0
    
    return torch.tensor(tensor)

def move_to_index(move):
    if not isinstance(move, chess.Move):
        raise TypeError(f"Expected chess.Move, got {type(move)}")

    from_sq = move.from_square
    to_sq = move.to_square
    dx = chess.square_file(to_sq) - chess.square_file(from_sq)
    dy = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    
    # Queen moves (56 directions)
    if abs(dx) == abs(dy) or dx == 0 or dy == 0:
        direction = (np.sign(dx), np.sign(dy))
        distance = max(abs(dx), abs(dy)) - 1
        try:
            return 64 * 0 + distance * 8 + [(1,0), (1,1), (0,1), (-1,1), 
                                          (-1,0), (-1,-1), (0,-1), (1,-1)].index(direction)
        except ValueError:
            raise ValueError(f"Invalid queen direction: {direction}")
    
    # Knight moves (8 directions)
    if {abs(dx), abs(dy)} == {1, 2}:
        return 64 * 1 + [(2,1), (1,2), (-1,2), (-2,1),
                       (-2,-1), (-1,-2), (1,-2), (2,-1)].index((dx, dy))
    
    # Underpromotions (9 types per direction)
    if move.promotion and move.promotion != chess.QUEEN:
        promo_type = [chess.KNIGHT, chess.BISHOP, chess.ROOK].index(move.promotion)
        direction = (dx, dy)
        if direction == (0, 1):  # Forward
            return 64 * 2 + promo_type
        elif direction == (-1, 1):  # Left
            return 64 * 3 + promo_type
        elif direction == (1, 1):  # Right
            return 64 * 4 + promo_type
    
    # Queen promotions or fallback
    return 64 * 5 + chess.square_file(to_sq) + 8 * chess.square_rank(to_sq)

def index_to_move(board, index):
    for move in board.legal_moves:
        if move_to_index(move) == index:
            return move
    return None  # hoặc raise Exception nếu không tìm thấy


def get_policy_vector(board, mcts_node):
    policy = np.zeros(4672, dtype=np.float32)
    total_visits = sum(child.visit_count for child in mcts_node.children.values())
    
    for move, child in mcts_node.children.items():
        idx = move_to_index(move)
        policy[idx] = child.visit_count / total_visits
    
    return policy