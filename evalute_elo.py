import chess
import chess.engine
import torch
import random
import math
from tqdm import tqdm
from model import AlphaZeroNet
from mcts import MCTS
from utils import board_to_tensor

# Đường dẫn tới Stockfish
STOCKFISH_PATH = "./stockfish/stockfish-windows-x86-64-avx2"  # Đổi nếu cần

# Elo giả lập cho Stockfish
STOCKFISH_ELO = 1350  # Mức thấp nhất mà Stockfish hỗ trợ

# Số ván để đánh giá
NUM_GAMES = 20
TIME_LIMIT = 1.0

# Tải mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaZeroNet().to(device)
model.load_state_dict(torch.load("final_model_3.pt", map_location=device))
model.eval()

# Hàm chọn nước đi từ model
def get_model_move(board: chess.Board) -> str:
    mcts = MCTS(model, simulations=50, time_limit=TIME_LIMIT)
    move = mcts.search(board)
    return move.uci()

# Ước lượng chênh lệch Elo từ tỉ lệ thắng
def estimate_elo(score_ratio):
    if score_ratio == 1.0:
        return 1000
    if score_ratio == 0.0:
        return -1000
    return -400 * math.log10(1 / score_ratio - 1)

# Chơi 1 ván giữa model và Stockfish
def play_game(engine, model_as_white=True):
    board = chess.Board()
    while not board.is_game_over():
        if (board.turn == chess.WHITE and model_as_white) or (board.turn == chess.BLACK and not model_as_white):
            move_uci = get_model_move(board)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.05))  # Giới hạn suy nghĩ
            move_uci = result.move.uci()
        board.push_uci(move_uci)

    result = board.result()
    if result == "1-0":
        return 1 if model_as_white else 0
    elif result == "0-1":
        return 0 if model_as_white else 1
    else:
        return 0.5

# Hàm chính
def main():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # Cấu hình Stockfish yếu đi
    engine.configure({
        "Skill Level": 5,
        "UCI_LimitStrength": True,
        "UCI_Elo": STOCKFISH_ELO
    })

    score = 0
    for i in tqdm(range(NUM_GAMES)):
        model_white = (i % 2 == 0)
        result = play_game(engine, model_as_white=model_white)
        score += result

    engine.quit()

    score_ratio = score / NUM_GAMES
    elo_diff = estimate_elo(score_ratio)
    estimated_elo = STOCKFISH_ELO + elo_diff

    print(f"\n✅ Tỉ lệ thắng của model: {score_ratio * 100:.2f}%")
    print(f"📈 Chênh lệch Elo ước lượng: {elo_diff:.1f}")
    print(f"🏅 Elo ước lượng của model: {estimated_elo:.1f}")

if __name__ == "__main__":
    main()
