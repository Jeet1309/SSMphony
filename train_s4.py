import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# Import your modules
try:
    from tts_model import S4TTS
    from dataset import HindiDataset, tts_collate_fn
except ImportError:
    print("❌ Error: Missing files. Make sure 'tts_model.py' and 'dataset.py' are in this folder.")
    sys.exit(1)

def train():
    # Add this at the top of train()
    
    # ==========================================
    # 1. CONFIGURATION (EDIT THESE PATHS!)
    # ==========================================
    # Use forward slashes '/' even on Windows to avoid errors
    METADATA_PATH = "label.txt"   # <--- POINT TO YOUR REAL LABEL FILE
    WAV_DIR = "wav"        # <--- POINT TO YOUR REAL WAV FOLDER
    
    # Hyperparameters
    BATCH_SIZE = 1 # Start small (8 or 16) to avoid "Out of Memory"
    EPOCHS = 20      # How many times to go through the dataset
    LR = 2e-4            # Learning Rate (Standard for S4)
    SAVE_DIR = "checkpoints"
    
    # Auto-detect GPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on: {DEVICE}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==========================================
    # 2. DATASET & DATALOADER
    # ==========================================
    print("⏳ Loading Dataset (this might take a moment)...")
    try:
        train_dataset = HindiDataset(METADATA_PATH, WAV_DIR, train=True)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   -> Please edit METADATA_PATH in train.py to point to your real data.")
        return

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=tts_collate_fn,
        num_workers=0,      # Windows users: Keep this 0 if you get "BrokenPipe" errors
        pin_memory=True
    )
    
    vocab_size = len(train_dataset.tokenizer.char2idx)
    print(f"✅ Data Loaded. Vocab Size: {vocab_size}")

    # ==========================================
    # 3. INITIALIZE MODEL
    # ==========================================
    # d_model=384, n_layers=6 matches the config we designed earlier
    model = S4TTS(vocab_size=vocab_size, d_model=128, n_layers=6).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.L1Loss() # Standard Loss for Spectrograms (Mean Absolute Error)

    # ==========================================
    # 4. TRAINING LOOP
    # ==========================================
    print("\n🔥 STARTING TRAINING 🔥")

    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for text, mel_target in progress_bar:
            text, mel_target = text.to(DEVICE), mel_target.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass
            mel_pred = model(text)
            
            # --- SHAPE FIX (CRITICAL) ---
            # S4 upsampling might be 1-2 frames longer/shorter than audio due to rounding.
            # We assume the target is correct and crop/pad.
            
            # 1. Get minimum length
            min_len = min(mel_pred.shape[1], mel_target.shape[1])
            
            # 2. Crop both to match
            mel_pred = mel_pred[:, :min_len, :]
            mel_target = mel_target[:, :min_len, :]
            
            # Loss calculation
            loss = criterion(mel_pred, mel_target)
            
            # Backward Pass
            loss.backward()
            
            # Clip Gradients (Prevents S4 from exploding)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"   -> Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.5f}")
        
        # Save Checkpoint
        # We save the Vocab too, so inference is easy later!
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'vocab': train_dataset.tokenizer.char2idx 
        }, f"{SAVE_DIR}/epoch_{epoch+1}.pth")
        
        # Keep a "latest" file for easy resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'vocab': train_dataset.tokenizer.char2idx 
        }, f"{SAVE_DIR}/latest_checkpoint.pth")

if __name__ == "__main__":
    train()
