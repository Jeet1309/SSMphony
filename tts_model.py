import torch
import torch.nn as nn
from s4 import S4 # Imports your existing S4 layer

class S4TTS(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_mels=80, dropout=0.1):
        """
        S4-TTS: A robust End-to-End Text-to-Speech Model.
        
        Args:
            vocab_size: Total characters in your dataset (Hindi + English + Punctuation).
            d_model: Hidden dimension (384 is a good balance for 5k samples).
            n_layers: Number of S4 layers in Encoder and Decoder.
            n_mels: Output spectrogram channels (standard is 80).
        """
        super().__init__()
        
        # 1. TEXT ENCODER (The "Brain")
        # Projects char indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Pre-Net (Simple MLP to smooth embeddings before S4)
        self.encoder_prenet = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # S4 Encoder Stack (captures grammar, context, and bilingual switching)
        self.encoder_layers = nn.ModuleList([
            S4(d_model=d_model, d_state=64, l_max=2000, dropout=dropout, transposed=False)
            for _ in range(n_layers)
        ])
        
        # 2. UPSAMPLER (The "Bridge")
        # Stretches Text Length -> Audio Length.
        # We use 2 layers of ConvTranspose1d with stride 4 each (Total 16x upsampling).
        # This means 1 character ~= 16 audio frames (flexible via the conv weights).
        # 2. UPSAMPLER COMPONENTS
        # We define layers individually to handle reshaping manually in forward()
        
        # Upsample Block 1
        self.up_conv1 = nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=4, padding=0)
        self.up_norm1 = nn.LayerNorm(d_model) # Expects (Batch, Length, Channels)
        self.up_act1  = nn.GELU()
        self.up_drop1 = nn.Dropout(dropout)
        
        # Upsample Block 2
        self.up_conv2 = nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=4, padding=0)
        self.up_norm2 = nn.LayerNorm(d_model) # Expects (Batch, Length, Channels)
        self.up_act2  = nn.GELU()
        self.up_drop2 = nn.Dropout(dropout)

        # 3. MEL DECODER (The "Voice")
        # Refines the stretched vectors into smooth acoustic details (Mel Spectrogram).
        self.decoder_layers = nn.ModuleList([
            S4(d_model=d_model, d_state=64, l_max=16000, dropout=dropout, transposed=False)
            for _ in range(n_layers)
        ])
        
        # Post-Net (Refines the output to be sharper)
        self.output_head = nn.Linear(d_model, n_mels)

    def forward(self, x):
        # x: (Batch, Text_Length) - Character Indices
        
        # --- Encoder ---
        x = self.embedding(x) # (B, L_text, D)
        x = self.encoder_prenet(x)
        
        for layer in self.encoder_layers:
            x = layer(x) # (B, L_text, D)
            
        # --- Upsampling ---
        # ConvTranspose expects (Batch, Channels, Length)
        # 1. Prepare for Conv1 (Needs Channel First: B, C, L)
        x = x.transpose(1, 2) 
        x = self.up_conv1(x)      # -> (B, C, L*4)
        
        # 2. Prepare for Norm1 (Needs Channel Last: B, L, C)
        x = x.transpose(1, 2) 
        x = self.up_norm1(x)
        x = self.up_act1(x)
        x = self.up_drop1(x)
        
        # 3. Prepare for Conv2 (Needs Channel First: B, C, L)
        x = x.transpose(1, 2)
        x = self.up_conv2(x)      # -> (B, C, L*16)
        
        # 4. Prepare for Norm2 (Needs Channel Last: B, L, C)
        x = x.transpose(1, 2)
        x = self.up_norm2(x)
        x = self.up_act2(x)
        x = self.up_drop2(x)
        
        # --- Decoder ---
        for layer in self.decoder_layers:
            x = layer(x)
            
        # --- Output ---
        mels = self.output_head(x) # (B, L_audio, n_mels)
        return mels