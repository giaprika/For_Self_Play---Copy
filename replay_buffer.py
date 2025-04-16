import os
import torch

def load_buffer(filepath):
    try:
        buffer = torch.load(filepath, weights_only=False)
        return buffer
    except (FileNotFoundError, EOFError) as e:
        return []

def save_buffer(buffer, filepath):
    tmp_path = filepath + ".tmp"
    backup_path = filepath + ".bak"

    try:
        # Ghi file tạm
        torch.save(buffer, tmp_path)

        # Nếu đã có file chính, backup lại
        if os.path.exists(filepath):
            try:
                os.replace(filepath, backup_path)
            except Exception as e:
                print(f"[Warning] Không thể tạo backup: {e}")

        # Ghi đè file chính bằng file tạm
        os.replace(tmp_path, filepath)
        print(f"[Info] Đã lưu buffer an toàn vào '{filepath}'")

    except Exception as e:
        print(f"[Error] Lỗi khi lưu buffer: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def add_games_to_buffer(buffer, new_games, max_size=200000):
    buffer.extend(new_games)
    if len(buffer) > max_size:
        buffer = buffer[-max_size:]
    return buffer
