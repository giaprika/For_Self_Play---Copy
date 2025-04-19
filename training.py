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
import multiprocessing
import time
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AlphaZeroTrainer:
    def __init__(self, model, epochs=20, batch_size=64, learning_rate=1e-4):
        self.model = model.to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def train(self, game_data):
        if not game_data:
            print("⚠️ Không có dữ liệu để train.")
            return
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
                self.optimizer.zero_grad()
                
                policy_pred, value_pred = self.model(state_batch)
                
                value_loss = self.loss_fn(value_pred.view(-1), value_batch.view(-1))
                policy_log_probs = torch.log_softmax(policy_pred, dim=1)
                policy_loss = -torch.mean(torch.sum(policy_batch * policy_log_probs, dim=1))
                
                loss = policy_loss + value_loss
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(dataloader)}")
    
    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

def run_self_play_worker(worker_id, model_path, games_per_worker):
    model = AlphaZeroNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    buffer_file = f"replay_buffer_workers_{worker_id}.pt"
    buffer = load_buffer(buffer_file)

    for i in range(games_per_worker):
        print(f"[Worker {worker_id}] Game {i+1}/{games_per_worker}")
        board = chess.Board()
        sp = SelfPlay(model, time_limit=1.0, board=board)
        game_data, result = sp.play_game()

        if game_data:
            buffer = add_games_to_buffer(buffer, game_data)
            save_buffer(buffer, buffer_file)
            print(f"[Worker {worker_id}] Game {i+1} saved. Result: {result}")
        time.sleep(0.05)

def main():
    multiprocessing.set_start_method('spawn', force=True)  # Windows/macOS compatibility

    model_path = "model_self_play_1.pt"
    model = AlphaZeroNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    trainer = AlphaZeroTrainer(model, epochs=20, batch_size=64)

    num_workers = 4
    games_per_worker = 5
    num_iterations = 20

    for iteration in range(num_iterations):
        print(f"\n==============================")
        print(f"🔁 Iteration {iteration+1}/{num_iterations}")
        print(f"==============================")

        # ✅ Backup model trước khi train
        if os.path.exists(model_path):
            backup_path = f"model_self_play_backup_iter_{iteration+1}.pt"
            shutil.copyfile(model_path, backup_path)
            print(f"🗂️ Backup model to {backup_path}")

        # 🧠 Load mô hình mới nhất cho self-play
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Self-play phase
        processes = []
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=run_self_play_worker,
                args=(i, model_path, games_per_worker)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Gom buffer
        combined_buffer = []
        for i in range(num_workers):
            buf = load_buffer(f"replay_buffer_workers_{i}.pt")
            combined_buffer = add_games_to_buffer(combined_buffer, buf[-5000:]) #Đổi thành -10000 với gpu

        # Train
        print(f"🧠 Training on {len(combined_buffer)} samples...")
        trainer.train(combined_buffer)

        # Save model
        trainer.save_model(model_path)
        print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    main()
    

    
    