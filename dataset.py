import torch
import torchaudio
import os
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ==========================================
# 1. CONFIGURATION
# ==========================================
# These constants control how we process audio.
SAMPLE_RATE = 22050     # Standard for TTS (balance between quality and speed)
N_MELS = 80             # Number of frequency bands in the spectrogram
N_FFT = 1024            # Size of the FFT window
HOP_LENGTH = 256        # How much we slide the window (Controls time resolution)
MAX_WAV_VALUE = 32768.0 # Max value for 16-bit audio

# ==========================================
# 2. TOKENIZER CLASS
# ==========================================
class TTSTokenizer:
    """
    Converts Text -> Numbers (Indices) and back.
    It builds a vocabulary from your dataset automatically.
    """
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        
        # Reserved Tokens:
        # <PAD> (0): Used to fill empty space in short sentences.
        # <UNK> (1): Used if we see a character we don't know during inference.
        self.pad_token = "<PAD>" 
        self.unk_token = "<UNK>" 
        self.insert_token(self.pad_token)
        self.insert_token(self.unk_token)

    def insert_token(self, char):
        """Adds a new character to the vocabulary."""
        if char not in self.char2idx:
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def build_vocab(self, all_text_list):
        """
        Scans your entire dataset (all_text_list) to find every unique character
        (Hindi, English, Punctuation) and assigns it an ID.
        """
        unique_chars = set()
        for text in all_text_list:
            unique_chars.update(list(text))
        
        # Sort characters so the ID assignment is always the same
        for char in sorted(list(unique_chars)):
            self.insert_token(char)
        
        print(f"✅ Vocabulary Built! Size: {len(self.char2idx)} tokens.")

    def text_to_sequence(self, text):
        """Converts string 'Hello' -> [12, 5, 12, 12, 15]"""
        return [self.char2idx.get(c, self.char2idx[self.unk_token]) for c in text]

    def sequence_to_text(self, seq):
        """Converts [12, 5, ...] -> 'Hello'"""
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        # We skip 0 because that is the <PAD> token
        return "".join([self.idx2char.get(idx, "") for idx in seq if idx != 0])
    
    def save(self, path="vocab.pth"):
        """Saves the vocabulary to disk so we can use it for inference later."""
        torch.save(self.char2idx, path)
        
    def load(self, path="vocab.pth"):
        """Loads a saved vocabulary."""
        self.char2idx = torch.load(path)
        self.idx2char = {v: k for k, v in self.char2idx.items()}

# ==========================================
# 3. DATASET CLASS
# ==========================================
class HindiDataset(Dataset):
    """
    Handles loading your specific data format.
    Reads 'label.txt', loads .wav files, and converts them to Mel-Spectrograms.
    """
    def __init__(self, metadata_path, wav_dir, tokenizer=None, train=True):
        self.wav_dir = wav_dir
        self.items = []
        MAX_SECONDS = 7.5
        
        
        # Check if file exists
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # --- PARSING LOGIC ---
        # Your format is: ( filename " text " )
        # We use Regex to pull out the filename and the text.
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r'\(\s*(\S+)\s+"(.*?)"\s*\)', line)
                if match:
                    file_id = match.group(1)
                    text = match.group(2).strip()
                    wav_path = os.path.join(wav_dir, f"{file_id}.wav")
                    
                    
                    if os.path.exists(wav_path):
                        # --- NEW: Check Length before adding ---
                        # We use os.path.getsize as a rough proxy first to be fast, 
                        # or just load metadata. Let's load metadata properly.
                        info = torchaudio.info(wav_path)
                        duration = info.num_frames / info.sample_rate
                        
                        if duration <= MAX_SECONDS:
                            self.items.append((wav_path, text))
                        # else:
                            # print(f"⚠️ Skipping {file_id} (Too long: {duration:.2f}s)")
        
        print(f"🔹 Found {len(self.items)} valid samples in {metadata_path}")

        # --- TOKENIZER SETUP ---
        if tokenizer is None:
            self.tokenizer = TTSTokenizer()
            # If we are training, we build the vocab from scratch using our text
            all_texts = [x[1] for x in self.items]
            self.tokenizer.build_vocab(all_texts)
            if len(self.items) > 0:
                self.tokenizer.save("vocab.pth") 
        else:
            self.tokenizer = tokenizer

        # --- AUDIO TRANSFORM SETUP ---
        # We use Torchaudio to turn sound waves into an image (Spectrogram)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            center=False
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, text = self.items[idx]
        
        # A. PROCESS TEXT
        # Convert "Hello" -> tensor([12, 5, 12, 12, 15])
        text_seq = torch.tensor(self.tokenizer.text_to_sequence(text), dtype=torch.long)
        
        # B. PROCESS AUDIO
        waveform, sr = torchaudio.load(wav_path)
        
        # 1. Resample: Ensure audio is 22050Hz (crucial for model consistency)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # 2. Mono: Remove stereo channels if present
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :]
            
        # 3. Spectrogram: Convert Wave -> Mel Spectrogram
        mel = self.mel_transform(waveform)
        
        # 4. Log Compression: Audio energy is exponential; models learn better in Log space.
        # We use clamp(min=1e-5) to prevent log(0) errors (Negative Infinity).
        mel = torch.log(torch.clamp(mel, min=1e-5))
        
        # 5. Transpose: Model expects (Time, Channels), Torchaudio gives (Channels, Time)
        mel = mel.squeeze(0).transpose(0, 1)
        
        return text_seq, mel

# ==========================================
# 4. COLLATION FUNCTION (BATCHING)
# ==========================================
def tts_collate_fn(batch):
    """
    Takes a list of samples [(text1, mel1), (text2, mel2)...]
    and pads them so they can be stacked into a single Batch Tensor.
    """
    # 1. Sort by audio length (descending)
    # This reduces padding waste (packing similar lengths together)
    batch.sort(key=lambda x: x[1].shape[0], reverse=True)
    
    text_seqs = [item[0] for item in batch]
    mel_seqs = [item[1] for item in batch]
    
    # 2. Pad Text
    # Short sentences get filled with 0 (<PAD>) at the end
    text_padded = pad_sequence(text_seqs, batch_first=True, padding_value=0)
    
    # 3. Pad Audio
    # Short audio gets filled with silence (-11.5) at the end
    # Why -11.5? Because log(1e-5) ≈ -11.51. This represents "Zero Energy".
    mel_padded = pad_sequence(mel_seqs, batch_first=True, padding_value=-11.5129)
    
    return text_padded, mel_padded

# ==========================================
# 5. SELF-TEST BLOCK
# ==========================================
if __name__ == "__main__":
    import shutil
    import numpy as np
    from scipy.io import wavfile

    print("🚀 RUNNING DATASET SELF-TEST 🚀\n")

    # Setup dummy directories
    TEST_DIR = "temp_test_data"
    WAV_DIR = os.path.join(TEST_DIR, "wavs")
    LABEL_FILE = os.path.join(TEST_DIR, "label.txt")
    
    if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)
    os.makedirs(WAV_DIR)

    print("🔹 Generating dummy wav files and label.txt...")
    
    # Create fake audio (Sine waves)
    sr = 22050
    wavfile.write(os.path.join(WAV_DIR, "test_01.wav"), sr, np.zeros((sr*2,), dtype=np.float32))
    wavfile.write(os.path.join(WAV_DIR, "test_02.wav"), sr, np.zeros((sr*4,), dtype=np.float32))

    # Create fake label file
    with open(LABEL_FILE, "w", encoding="utf-8") as f:
        f.write('( test_01 "Hello World" )\n')
        f.write('( test_02 "नमस्ते भारत" )\n')

    # Test Loading
    try:
        print("🔹 Initializing Dataset...")
        ds = HindiDataset(LABEL_FILE, WAV_DIR)
        
        print(f"   Vocab Size: {len(ds.tokenizer.char2idx)}")
        
        print("🔹 Testing Batching...")
        loader = DataLoader(ds, batch_size=2, collate_fn=tts_collate_fn)
        for t, m in loader:
            print(f"   Batch Text Shape: {t.shape}")
            print(f"   Batch Mel Shape:  {m.shape}")
            if m.shape[1] > 300: 
                print("✅ PASS: Pipeline works!")
            else:
                print("❌ FAIL: Padding issue.")
            break
            
    except Exception as e:
        print(f"❌ CRASH: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)