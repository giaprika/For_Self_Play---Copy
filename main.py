import os

import pygame
import chess
import engine as ce
import threading
import time

from pygame.locals import *

# Constants
WIDTH, HEIGHT = 480, 480
SQ_SIZE = WIDTH // 8
FPS = 60
WHITE = (237, 199, 126)
BLACK = (180, 104, 23)

# Initialize Pygame
pygame.init()
# Map chess symbols to custom image filenames
PIECE_IMAGE_MAP = {
    'P': 'pl',  # White Pawn (Light)
    'N': 'nl',  # White Knight (Light)
    'B': 'bl',  # White Bishop (Light)
    'R': 'rl',  # White Rook (Light)
    'Q': 'ql',  # White Queen (Light)
    'K': 'kl',  # White King (Light)
    'p': 'pd',  # Black Pawn (Dark)
    'n': 'nd',  # Black Knight (Dark)
    'b': 'bd',  # Black Bishop (Dark)
    'r': 'rd',  # Black Rook (Dark)
    'q': 'qd',  # Black Queen (Dark)
    'k': 'kd',  # Black King (Dark)
}

# Load piece images with custom filenames
PIECE_IMAGES = {}
for piece, image_name in PIECE_IMAGE_MAP.items():
    try:
        image_path = f'images/{image_name}.png'
        PIECE_IMAGES[piece] = pygame.image.load(image_path)
        PIECE_IMAGES[piece] = pygame.transform.scale(PIECE_IMAGES[piece], (SQ_SIZE, SQ_SIZE))
    except FileNotFoundError:
        print(f"Error: Image for piece '{piece}' not found at {image_path}.")
        exit()

# Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess GUI")

# Timer
font = pygame.font.Font(None, 36)


def display_game_over():
    result_text = "Game Over"
    game_over_text = font.render(result_text, True, WHITE)
    screen.fill(BLACK)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2))
    pygame.display.flip()
    pygame.time.wait(5000)  # Wait for 5 seconds before closing


class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.selected_square = None
        self.dragging_piece = None
        self.start_pos = None
        self.timer_white = 600  # 10 minutes
        self.timer_black = 600
        self.last_update = pygame.time.get_ticks()
        self.current_turn = chess.WHITE
        self.player_color = None
        self.model = ""
        self.max_time = None

    def draw_board(self):
        colors = [WHITE, BLACK]
        for row in range(8):
            for col in range(8):
                # Flip rows and columns for black's perspective
                draw_row = 7 - row if self.player_color == chess.BLACK else row
                draw_col = 7 - col if self.player_color == chess.BLACK else col
                color = colors[(row + col) % 2]
                pygame.draw.rect(screen, color, pygame.Rect(draw_col * SQ_SIZE, draw_row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_pieces(self):
        for row in range(8):
            for col in range(8):
                # Flip rows and columns for black's perspective
                board_row = 7 - row if self.player_color == chess.BLACK else row
                board_col = 7 - col if self.player_color == chess.BLACK else col
                piece = chess_game.board.piece_at(chess.square(col, 7 - row))
                if piece:
                    piece_symbol = piece.symbol()
                    if chess_game.selected_square == chess.square(col, 7 - row) and chess_game.dragging_piece:
                        # Skip drawing the piece being dragged
                        continue
                    draw_x = board_col * SQ_SIZE
                    draw_y = board_row * SQ_SIZE
                    screen.blit(PIECE_IMAGES[piece_symbol], (draw_x, draw_y))

    def draw_dragging_piece(self):
        if self.dragging_piece:
            pos = pygame.mouse.get_pos()
            screen.blit(self.dragging_piece, (pos[0] - SQ_SIZE // 2, pos[1] - SQ_SIZE // 2))

    def handle_events(self):
        # Player
        if chess_game.current_turn == self.player_color:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()
                elif event.type == MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event)
                elif event.type == MOUSEBUTTONUP:
                    self.handle_mouse_up(event)

        # Engine
        else:
            if not hasattr(self, "ai_thread") or not self.ai_thread.is_alive():
                self.ai_thread = threading.Thread(target=self.run_engine_move)
                self.ai_thread.start()

    def run_engine_move(self):
        try:
            print("[AI] Thinking...")
            best_move = self.playEngineMove(self.model, self.max_time, self.current_turn)
            print(f"[AI] Best move: {best_move}")
            
            # ✅ Delay 1s trước khi push nước đi => đảm bảo vẽ ra sau khi delay
            time.sleep(1)
            
            self.board.push(best_move)
            self.current_turn = not self.current_turn
            print("[AI] Move done.")
        except Exception as e:
            print(f"[ERROR] AI move failed: {e}")


    def playEngineMove(self, model, max_time, color):
        try:
            engine = ce.Engine(model, self.board, max_time, color)
            engine.max_depth = 4
            return engine.get_best_move()  # ✅ Chỉ trả về move
        except Exception as e:
            print(f"[ERROR] in playEngineMove: {e}")
            return None

    def handle_mouse_down(self, event):
        row, col = event.pos[1] // SQ_SIZE, event.pos[0] // SQ_SIZE
        # Mirror the square for black's perspective
        if self.player_color == chess.BLACK:
            row, col = 7 - row, 7 - col
        square = chess.square(col, 7 - row)
        piece = self.board.piece_at(square)
        if piece and piece.color == self.current_turn:
            self.selected_square = square
            self.start_pos = (row, col)
            self.dragging_piece = PIECE_IMAGES[piece.symbol()]

    def handle_mouse_up(self, event):
        if not self.dragging_piece:
            return

        end_row, end_col = event.pos[1] // SQ_SIZE, event.pos[0] // SQ_SIZE
        if self.player_color == chess.BLACK:
            end_row, end_col = 7 - end_row, 7 - end_col

        end_square = chess.square(end_col, 7 - end_row)

        move = None
        for legal_move in self.board.legal_moves:
            if legal_move.from_square == self.selected_square and legal_move.to_square == end_square:
                move = legal_move
                break

        if move:
            self.board.push(move)
            self.current_turn = not self.current_turn

        self.selected_square = None
        self.dragging_piece = None


    def update_timers(self):
        now = pygame.time.get_ticks()
        elapsed_time = (now - self.last_update) // 1000
        if self.current_turn == chess.WHITE:
            self.timer_white -= elapsed_time
        else:
            self.timer_black -= elapsed_time
        self.last_update = now

    def init(self):

        """Starts a new game and handles player inputs and engine moves."""
        # Get human player's color
        color = ""
        while color not in ["b", "w"]:
            color = input('Play as (type "b" or "w"): ')
            if color.lower() == "b":
                self.player_color = chess.BLACK
            else:
                self.player_color = chess.WHITE

        # Get the model
        try:
            while not os.path.exists(self.model):
                self.model = input('Type in the model path: ').lower()
                if not os.path.exists(self.model):
                    print(f"Model '{self.model}' not found. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return

        # Get thinking time
        while True:
            try:
                self.max_time = int(input("Choose thinking time (in seconds): "))
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

    def run(self):
        #self.init()
        self.player_color = chess.WHITE
        self.model = "data/saved_models/chess_test_model16.365107344799355.pth"
        self.max_time = 2

        clock = pygame.time.Clock()
        while not self.board.is_checkmate():
            self.handle_events()
            self.draw_board()
            self.draw_pieces()
            self.draw_dragging_piece()
            pygame.display.flip()
            clock.tick(FPS)
        display_game_over()


# Main Entry
if __name__ == '__main__':
    chess_game = ChessGame()
    chess_game.run()