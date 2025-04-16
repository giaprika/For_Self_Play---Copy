import numpy as np
import chess
import time
from collections import defaultdict
from utils import move_to_index

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior

    def is_expanded(self):
        return len(self.children) > 0

    def select_child(self):
        log_total = np.log(self.visit_count + 1)
        best_score = -np.inf
        best_move = None
        best_child = None
        
        # Hệ số điều chỉnh exploration-exploitation
        c_puct = 2.0  # Tăng giá trị này sẽ khuyến khích khám phá hơn

        for move, child in self.children.items():
            q_value = child.total_value / (child.visit_count + 1e-8)
            u_value = c_puct * child.prior * np.sqrt(log_total) / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def expand(self, policy):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return
        # Đảm bảo policy cho các nước đi hợp lệ có tổng xác suất > 0
        move_probs = [policy[move_to_index(move)] for move in legal_moves]
        if sum(move_probs) <= 1e-8:
            # Nếu tổng xác suất quá nhỏ, gán xác suất đều cho tất cả các nước đi
            policy_sum = 1.0 / len(legal_moves)
            for move in legal_moves:
                idx = move_to_index(move)
                policy[idx] = policy_sum
        else:
            # Chuẩn hóa xác suất
            total = sum(policy[move_to_index(move)] for move in legal_moves)
            for move in legal_moves:
                policy[move_to_index(move)] /= total
        
        # Tạo các node con
        for move in legal_moves:
            prob = policy[move_to_index(move)]
            next_board = self.board.copy()
            next_board.push(move)
            self.children[move] = MCTSNode(next_board, parent=self, prior=prob)

    def backpropagate(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(-value)

class MCTS:
    def __init__(self, model, time_limit):
        self.model = model
        self.time_limit = time_limit
        self.root = None

    def search(self, board):
        self.root = MCTSNode(board)
        policy, _ = self.model.predict(board)

        # Thêm Dirichlet noise vào root node để tăng khám phá
        legal_moves = list(board.legal_moves)
        noise = np.random.dirichlet([0.3] * len(legal_moves))
        
        # Điều chỉnh policy với noise
        for i, move in enumerate(legal_moves):
            idx = move_to_index(move)
            policy[idx] = 0.75 * policy[idx] + 0.25 * noise[i]

        self.root.expand(policy)

        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = self.root
            path = [node]

            # Selection
            while node.is_expanded() and not node.board.is_game_over():
                move, node = node.select_child()
                path.append(node)

            # Evaluation
            if not node.board.is_game_over():
                policy, value = self.model.predict(node.board)
                node.expand(policy)
            else:
                result = node.board.result()
                if result == "1-0":
                    value = 1
                elif result == "0-1":
                    value = -1
                else:
                    value = 0

            # Backpropagation
            for node in reversed(path):
                node.backpropagate(value)
                value = -value

        # Thêm temperature parameter để điều chỉnh khả năng khám phá
        temperature = 1.0  # Cao hơn = khám phá nhiều hơn, thấp hơn = khai thác nhiều hơn

        if temperature == 0:
            # Chọn nước đi có visit_count cao nhất
            best_move = max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]
        else:
            # Chọn nước đi dựa trên phân phối xác suất với temperature
            counts = np.array([child.visit_count for child in self.root.children.values()])
            counts = counts ** (1.0 / temperature)
            probs = counts / np.sum(counts)
            moves = list(self.root.children.keys())
            best_move = np.random.choice(moves, p=probs)
        
        return best_move
