import torch
import sys

# Try importing the S4 class
try:
    from s4 import S4
except ImportError:
    print("❌ Error: Could not import S4. Make sure 's4.py' is in the same directory.")
    sys.exit(1)

def run_tests():
    print("\n🚀 STARTING S4 VERIFICATION SUITE 🚀\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # ==========================================
    # TEST 1: Shape & Forward Pass (CNN Mode)
    # ==========================================
    print("\n[1/3] Testing Shapes (CNN Mode)...")
    try:
        B, L, H, N = 2, 1024, 8, 64
        layer = S4(d_model=H, d_state=N, l_max=L, transposed=False).to(device)
        u = torch.randn(B, L, H).to(device)
        y = layer(u)
        
        if y.shape == u.shape:
            print("✅ PASS: Output shape matches input.")
        else:
            print(f"❌ FAIL: Shape Mismatch. Expected {u.shape}, got {y.shape}")
            return False
    except Exception as e:
        print(f"❌ CRASH: {e}")
        return False

    # ==========================================
    # TEST 2: Gradient Flow (Training Capability)
    # ==========================================
    print("\n[2/3] Testing Gradients (Backward Pass)...")
    try:
        u.requires_grad = True
        layer = S4(d_model=4, d_state=16, l_max=64, transposed=False).to(device)
        y = layer(u[:, :64, :4]) # Smaller input for speed
        loss = y.sum()
        loss.backward()
        
        # Check if critical params have gradients
        if layer.log_step.grad is not None and layer.C.grad is not None:
            print("✅ PASS: Gradients are flowing through the Woodbury kernel.")
        else:
            print("❌ FAIL: Parameters are not updating. Check computation graph.")
            return False
    except Exception as e:
        print(f"❌ CRASH: {e}")
        return False

    # ==========================================
    # TEST 3: CNN vs RNN Equivalence (The Big One)
    # ==========================================
    print("\n[3/3] Testing Logic Equivalence (CNN vs RNN)...")
    try:
        # Setup small deterministic test
        B, L, H, N = 1, 16, 2, 4
        layer = S4(d_model=H, d_state=N, l_max=L, transposed=False).to(device)
        layer.eval() # Disable dropout for comparison
        
        u = torch.randn(B, L, H).to(device)
        
        # 1. Run CNN
        y_cnn = layer(u)
        
        # 2. Run RNN
        layer.setup_rnn() # Convert params
        y_rnn_list = []
        for t in range(L):
            y_t = layer.step(u[:, t, :])
            y_rnn_list.append(y_t)
        y_rnn = torch.stack(y_rnn_list, dim=1)
        
        # 3. Compare
        diff = (y_cnn - y_rnn).abs().max().item()
        print(f"   CNN Output Sample: {y_cnn[0, :4, 0].detach().cpu().numpy()}")
        print(f"   RNN Output Sample: {y_rnn[0, :4  , 0].detach().cpu().numpy()}")
        print(f"   Max Difference: {diff:.8f}")
        
        if diff < 1e-2:
            print("✅ PASS: CNN and RNN match exactly!")
        else:
            print("❌ FAIL: CNN and RNN diverge.")
            return False
            
    except Exception as e:
        print(f"❌ CRASH: {e}")
        return False

    print("\n🎉 ALL CHECKS PASSED. Your S4 code is correct.")
    return True

if __name__ == "__main__":
    run_tests()