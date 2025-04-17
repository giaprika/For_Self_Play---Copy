import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from self_play import SelfPlay
from model import AlphaZeroNet
import chess
from replay_buffer import load_buffer, save_buffer, add_games_to_buffer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AlphaZeroTrainer:
    def __init__(self, model, epochs=10, batch_size=64, learning_rate=1e-4):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def train(self, game_data):
        # Prepare data for training
        states, policies, values = zip(*game_data)
        states = torch.stack(states).to(device)
        policies = torch.tensor(np.array(policies), dtype=torch.float32).to(device)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(device)
        
        dataset = TensorDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                state_batch, policy_batch, value_batch = batch
                state_batch = state_batch.to(device)
                policy_batch = policy_batch.to(device)
                value_batch = value_batch.to(device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                policy_pred, value_pred = self.model(state_batch)
                
                # Loss calculation
                # policy_loss = self.loss_fn(policy_pred, policy_batch)
                value_loss = self.loss_fn(value_pred.view(-1), value_batch.view(-1))
                # Policy loss (cross-entropy with target policy)
                policy_log_probs = torch.log_softmax(policy_pred, dim=1)  # log(π̂_θ)
                policy_loss = -torch.mean(torch.sum(policy_batch * policy_log_probs, dim=1))  # -π · log(π̂)
                
                loss = policy_loss + value_loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(dataloader)}")
    
    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
    
    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))

def train():
    # Khởi tạo mô hình
    model = AlphaZeroNet()
    model.load_state_dict(torch.load('model_self_play_1.pt', map_location=device))
    
    # Khởi tạo Trainer
    trainer = AlphaZeroTrainer(model, epochs=10, batch_size=64)
    
    # Số lượng ván cờ để tạo dữ liệu huấn luyện
    num_games = 60
    buffer = load_buffer("replay_buffer.pt")
    buffer_1 = load_buffer("replay_buffer_1.pt")
    buffer_2 = load_buffer("replay_buffer_2.pt")
    time_limit = 1.0
    
    for game_num in range(num_games):
        print(f"Game {game_num+1}/{num_games} started.")
        # Khởi tạo SelfPlay
        board = chess.Board()
        self_play = SelfPlay(model, time_limit=time_limit, board=board)
        
        game_data, game_result = self_play.play_game()
        
        if game_data:
            print("Đang add game_data vào buffer")
            buffer_2 = add_games_to_buffer(buffer_2, game_data)
            print("Đang lưu vào file")
            save_buffer(buffer_2, "replay_buffer_2.pt")

            if (game_num + 1) % 20 == 0:
                buffer = add_games_to_buffer(buffer, buffer_1)
                buffer = add_games_to_buffer(buffer, buffer_2)
                trainer.train(buffer)
                trainer.save_model("model_self_play_1.pt")
            print(f"Game {game_num+1}/{num_games} finished. Result: {game_result}")
        else:
            print(f"Game {game_num+1}/{num_games} skipped. No data collected.")
    
    # Lưu mô hình cuối cùng
    trainer.save_model("model_self_play_1.pt")
    print("Đã lưu model tại model_self_play_1.pt")

if __name__ == "__main__":
    print('Start')
    train()
    # model = AlphaZeroNet()
    # model.load_state_dict(torch.load('model_self_play_1.pt', map_location=device))
    
    # # Khởi tạo Trainer
    # trainer = AlphaZeroTrainer(model, epochs=10, batch_size=64)
    # buffer = load_buffer("replay_buffer.pt")
    # trainer.train(buffer)
