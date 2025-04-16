import pygame
import chess
import numpy as np
from mcts import MCTS
from model import AlphaZeroNet
from utils import board_to_tensor, get_policy_vector

class SelfPlay:
    def __init__(self, model, time_limit, board, board_size=600):
        self.model = model
        self.time_limit = time_limit
        self.board = board
        self.mcts = MCTS(model, time_limit=self.time_limit)
        self.board_size = board_size
        
        # Khởi tạo Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.board_size, self.board_size))
        pygame.display.set_caption("AlphaZero Chess")
        
        # Màu sắc
        self.WHITE_COLOR = (255, 230, 185)
        self.BLACK_COLOR = (185, 130, 85)
        self.GRID_COLOR = (200, 200, 200)
        
        # Tải hình ảnh quân cờ
        self.piece_images = self.load_piece_images()

    def load_piece_images(self):
        piece_images = {}
        pieces = ['p', 'r', 'n', 'b', 'q', 'k']
        colors = ['w', 'b']
        for color in colors:
            for piece in pieces:
                piece_images[f'{piece}{color}'] = pygame.image.load(f"images/{piece}{color}.png")
        return piece_images

    def draw_board(self):
        square_size = self.board_size // 8
        # Vẽ ô cờ
        for rank in range(8):
            for file in range(8):
                color = self.WHITE_COLOR if (rank + file) % 2 == 0 else self.BLACK_COLOR
                pygame.draw.rect(self.screen, color, (file*square_size, rank*square_size, square_size, square_size))
        
        # Vẽ quân cờ
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                row = 7 - row  # Sửa tọa độ
                piece_name = f"{piece.symbol().lower()}{'w' if piece.color == chess.WHITE else 'b'}"
                image = self.piece_images.get(piece_name)
                if image:
                    image = pygame.transform.scale(image, (square_size, square_size))
                    self.screen.blit(image, (col * square_size, row * square_size))

    def play_game(self):
        game_data = []
        game_result = None
        turn = chess.WHITE
        move_count = 0

        while not self.board.is_game_over():
            # Xử lý sự kiện
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Vẽ bàn cờ
            self.screen.fill((200, 200, 200))
            self.draw_board()
            pygame.display.flip()

            if self.board.turn == turn:
                pygame.event.pump()  # Giữ GUI phản hồi

                # Điều chỉnh temperature theo số lượng nước đi
                temperature = 1.0 if move_count < 15 else 0.5
                move_count += 1
                
                # Đặt temperature cho MCTS
                self.mcts.temperature = temperature
                
                # Tìm nước đi bằng MCTS
                move = self.mcts.search(self.board)
                if move is None:
                    print("No move found by MCTS. Ending game early.")
                    break

                # Lưu trạng thái trước khi đẩy nước đi
                state = board_to_tensor(self.board.copy())
                policy = get_policy_vector(self.board, self.mcts.root)
                game_data.append((state, policy, 0))  # tạm thời gán value = 0

                self.board.push(move)
            else:
                pygame.time.wait(500)

            turn = not turn

        # Kết quả ván cờ
        result = self.board.result()
        if result == "1-0":
            game_result = 1
        elif result == "0-1":
            game_result = -1
        else:
            game_result = 0

        # Gán lại value cho từng trạng thái
        game_data = [(s, p, game_result if i % 2 == 0 else -game_result) for i, (s, p, _) in enumerate(game_data)]

        return game_data, game_result
