"""
monitor_training.py - Xem tiến độ training SNN Vision
"""
import json
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

def monitor_training(log_file='models/vision/vision_training_history.json'):
    """Live plot training metrics"""
    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    print(f"📊 Monitoring {log_file}...")
    
    while True:
        try:
            if not os.path.exists(log_file):
                print("⏳ Waiting for training log file...")
                time.sleep(5)
                continue

            with open(log_file, 'r') as f:
                logs = json.load(f)
            
            if not logs:
                time.sleep(2)
                continue

            epochs = [l['epoch'] for l in logs]
            train_losses = [l['train_loss'] for l in logs]
            val_losses = [l['val_loss'] for l in logs]
            
            ax1.clear()
            ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
            ax1.plot(epochs, val_losses, 'r-o', label='Val Loss', markersize=4)
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('SNN Vision Training Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(5)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training()
