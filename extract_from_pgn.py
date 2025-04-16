import chess.pgn
import chess.engine
import numpy as np
import os
import torch
from concurrent.futures import ProcessPoolExecutor
from utils import board_to_tensor, move_to_index

def result_to_value(result_str):
    """Chuyển kết quả trận đấu thành giá trị số."""
    return 1 if result_str == "1-0" else -1 if result_str == "0-1" else 0

def get_stockfish_policy(board, engine, time_limit=0.5):
    """Lấy phân phối xác suất từ Stockfish cho các nước đi hợp lệ."""
    policy = np.zeros(4672, dtype=np.float32)
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return policy
    
    # engine.configure({"Threads": 2, "Hash": 2048})
    
    scores = []
    for move in legal_moves:
        temp_board = board.copy()
        temp_board.push(move)
        analysis = engine.analyse(temp_board, chess.engine.Limit(time=time_limit))
        score = analysis["score"].relative.score(mate_score=10000)
        scores.append(score)
    
    if not scores:
        return policy
    
    scores = np.array(scores, dtype=np.float32)
    scores -= np.max(scores)
    probabilities = np.exp(scores) / np.sum(np.exp(scores))
    
    for move, prob in zip(legal_moves, probabilities):
        try:
            idx = move_to_index(move)
            if 0 <= idx < 4672:
                policy[idx] = prob
        except Exception as e:
            print(f"Lỗi chuyển đổi move {move}: {e}")
    
    return policy

def process_single_game(game, stockfish_path, time_limit):
    """Xử lý một ván cờ và trả về dữ liệu."""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board = game.board()
    result = result_to_value(game.headers.get("Result", "1/2-1/2"))
    game_data = []
    
    for move in game.mainline_moves():
        policy = get_stockfish_policy(board, engine, time_limit)
        state = board_to_tensor(board)
        value = result if board.turn == chess.WHITE else -result
        game_data.append((state, policy, value))
        board.push(move)
    
    engine.quit()
    return game_data

def process_pgn_chunk(pgn_chunk_path, stockfish_path, time_limit, chunk_id):
    """Xử lý một chunk PGN và lưu dữ liệu sau mỗi 10 game."""
    chunk_data = []
    game_counter = 0
    part_counter = 1
    
    with open(pgn_chunk_path, 'r', encoding='utf-8') as f:
        while True:
            game = chess.pgn.read_game(f)
            if not game:
                break
            
            try:
                print(f"Đang xử lý game {game_counter}")
                game_data = process_single_game(game, stockfish_path, time_limit)
                print(f"Xu lý xong game {game_counter}")
                chunk_data.extend(game_data)
                game_counter += 1
                
                # Lưu sau mỗi 10 game
                if game_counter % 10 == 0:
                    save_path = os.path.join(
                        "optimized_data", 
                        f"chunk_{chunk_id}_part_{part_counter}.pt"
                    )
                    states, policies, values = zip(*chunk_data)
                    torch.save({
                        'states': torch.stack(states),
                        'policies': torch.tensor(np.array(policies), dtype=torch.float32),
                        'values': torch.tensor(np.array(values), dtype=torch.float32)
                    }, save_path)
                    print(f"Đã lưu chunk {chunk_id} - phần {part_counter}")
                    chunk_data = []
                    part_counter += 1
                    
            except Exception as e:
                print(f"Lỗi ở game {game_counter} trong chunk {chunk_id}: {e}")
    
    # Lưu phần còn lại
    if chunk_data:
        save_path = os.path.join(
            "optimized_data", 
            f"chunk_{chunk_id}_part_{part_counter}.pt"
        )
        states, policies, values = zip(*chunk_data)
        torch.save({
            'states': torch.stack(states),
            'policies': torch.tensor(np.array(policies), dtype=torch.float32),
            'values': torch.tensor(np.array(values), dtype=torch.float32)
        }, save_path)
        print(f"Đã lưu chunk {chunk_id} - phần cuối ({len(chunk_data)} game)")

def main():
    stockfish_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"
    time_limit = 0.5
    save_dir = "optimized_data"
    os.makedirs(save_dir, exist_ok=True)
    
    pgn_chunks = [
        ("lichess_db_standard_rated_2013-01.pgn", 1),
        ("lichess_db_standard_rated_2013-02.pgn", 2),
        ("lichess_db_standard_rated_2013-03.pgn", 3),
        ("lichess_db_standard_rated_2013-04.pgn", 4)
    ]
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for chunk_path, chunk_id in pgn_chunks:
            future = executor.submit(
                process_pgn_chunk,
                chunk_path,
                stockfish_path,
                time_limit,
                chunk_id
            )
            futures.append(future)
        
        for future in futures:
            future.result()

    print("Hoàn thành xử lý!")

if __name__ == "__main__":
    print("start")
    main()