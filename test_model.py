import torch
import sys
import re

# 1. Import Model
try:
    from tts_model import S4TTS
except ImportError:
    print("❌ Error: Could not import S4TTS. Make sure 'tts_model.py' and 's4.py' are here.")
    sys.exit(1)

def test_parsing_logic():
    print("\n🔹 [1/2] Testing Data Parsing Logic...")
    
    # Example line from your dataset
    line = '( train_hindifullmale_00001 " प्रसिद्द कबीर अध्येता, पुरुषोत्तम अग्रवाल का यह शोध आलेख, उस रामानंद की खोज करता है " )'
    
    # Regex to extract: ID and Text
    # Look for: ( ID " TEXT " )
    match = re.search(r'\(\s*(\S+)\s+"(.*?)"\s*\)', line)
    
    if match:
        file_id = match.group(1)
        text_content = match.group(2).strip()
        print(f"✅ Parsed ID:   {file_id}")
        print(f"✅ Parsed Text: {text_content[:30]}...") # Print first 30 chars
        
        if file_id == "train_hindifullmale_00001" and "कबीर" in text_content:
            print("🎉 Parsing Logic Verified!")
            return True
    else:
        print("❌ Parsing Failed. The Regex didn't match your format.")
        return False

def test_model_forward():
    print("\n🔹 [2/2] Testing Model Forward Pass...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Running on: {device}")
    
    # Hyperparameters
    vocab_size = 100 # Approx size for Hindi+English chars
    batch_size = 2
    text_len = 50     # Length of a sentence
    
    # Initialize Model
    model = S4TTS(vocab_size=vocab_size, d_model=256, n_layers=4).to(device)
    print("   Model initialized successfully.")
    
    # Create Dummy Inputs
    text_indices = torch.randint(0, vocab_size, (batch_size, text_len)).to(device)
    
    try:
        # Forward Pass
        mels = model(text_indices)
        
        # Check Output
        # Input 50 chars * 16x upsample = 800 frames expected
        expected_len = text_len * 16 
        
        print(f"   Input Shape: {text_indices.shape}")
        print(f"   Output Shape: {mels.shape}")
        
        if mels.shape[1] == expected_len and mels.shape[2] == 80:
            print("🎉 Success! Model produced correct Mel-Spectrogram shape.")
            return True
        else:
            print(f"❌ Shape Mismatch. Expected (B, {expected_len}, 80).")
            return False
            
    except Exception as e:
        print(f"❌ Model Crash: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING TTS MODEL & DATA CHECK")
    if test_parsing_logic() and test_model_forward():
        print("\n✅ SYSTEM READY. You can proceed to training.")
    else:
        print("\n⚠️ SYSTEM NOT READY. Fix errors above.")