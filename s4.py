# import torch
# import torch.nn as nn
# import torch.fft as fft
# import numpy as np

# def _r2c(x):
#     """Helper to convert real tensor to complex safely"""
#     return torch.complex(x, torch.zeros_like(x))

# class S4(nn.Module):
#     def __init__(self, d_model, d_state=64, l_max=1024, dropout=0.2, transposed=True):
#         super().__init__()
#         self.h = d_model      # H: Hidden dimension
#         self.n = d_state      # N: State dimension (Memory size)
#         self.l_max = l_max    # L: Maximum Sequence Length
#         self.transposed = transposed

#         # ============================================================
#         # 1. MATHEMATICAL INITIALIZATION (HiPPO Matrix)
#         # ============================================================
#         def make_hippo(N):
#             def p(n, k):
#                 if n > k: return np.sqrt(2*n+1)*np.sqrt(2*k+1)
#                 elif n == k: return n+1
#                 else: return 0
#             return np.array([[p(n,k) for k in range(N)] for n in range(N)])
        
#         A = -make_hippo(self.n)

#         # ============================================================
#         # 2. NPLR DECOMPOSITION (Speed Trick)
#         # ============================================================
#         # A = V Λ V* - P Q^T
#         p = 0.5 * np.sqrt(2 * np.arange(self.n) + 1.0)
#         q = 2 * p
#         S = A + p[:, np.newaxis] * q[np.newaxis, :]
        
#         # Diagonalize S
#         eig_vals, eig_vecs = np.linalg.eig(S)
        
#         Lambda = torch.tensor(eig_vals, dtype=torch.cfloat)
#         V = torch.tensor(eig_vecs, dtype=torch.cfloat)
#         V_inv = V.conj().transpose(0, 1)

#         # ============================================================
#         # 3. LEARNABLE PARAMETERS
#         # ============================================================
#         self.register_buffer("Lambda", Lambda)
        
#         # Learnable P and B projected into V basis
#         self.P = nn.Parameter(V_inv @ torch.tensor(p, dtype=torch.cfloat))
#         self.B = nn.Parameter(V_inv @ torch.tensor(np.ones(self.n), dtype=torch.cfloat))
        
#         # Learnable C (Output Projection)
#         self.C = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat))
        
#         # Learnable Delta (Step Size)
#         # Initialized in log space.
#         self.log_step = nn.Parameter(torch.rand(self.h) - 1.0)

#         # Learnable D (Skip Connection)
#         self.D = nn.Parameter(torch.randn(self.h))
        
#         # Standard NN layers
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.GELU()
#         self.norm = nn.LayerNorm(self.h)
        
#         # For RNN Mode (State Cache)
#         self.state = None
#         self.Ab = None
#         self.Bb = None

#     def kernel_dplr(self, L=None):
#         """
#         Computes the S4 Convolution Kernel K using the Woodbury Identity.
#         FIXED: Runs complex math on CPU to avoid Windows CUDA backend crashes.
#         """
#         if L is None:
#             L = self.l_max

#         # --- FIX: CPU CASTING FOR KERNEL GENERATION ---
#         # We perform the complex division on CPU (safe), then move back to GPU.
#         target_device = self.C.device
        
#         # 1. Move Parameters to CPU
#         dt = torch.exp(self.log_step).to("cpu").view(-1, 1, 1) # (H, 1, 1)
#         Lambda = self.Lambda.to("cpu").view(1, -1, 1)          # (1, N, 1)
#         P = self.P.to("cpu").view(1, -1, 1)                    # (1, N, 1)
#         B = self.B.to("cpu").view(1, -1, 1)                    # (1, N, 1)
#         C = self.C.to("cpu").view(-1, self.n, 1)               # (H, N, 1)
        
#         # 2. Create Frequencies (Roots of Unity) on CPU
#         omega = torch.arange(L, device="cpu") / L
#         z = torch.exp(-2j * np.pi * omega).view(1, 1, L) * 0.999
        
#         # 3. Bilinear Transform (z -> s) on CPU
#         s = (2.0 / dt) * (1.0 - z) / (1.0 + z + 1e-6) # (H, 1, L)
        
#         # 4. Woodbury Identity on CPU (The critical part that was crashing)
#         denom = s - Lambda # (H, N, L)
        
#         k0 = (C * B / denom).sum(dim=1) 
#         k1 = (C * P / denom).sum(dim=1)
#         k2 = (P.conj() * B / denom).sum(dim=1)
#         k3 = (P.conj() * P / denom).sum(dim=1)
        
#         K_s = k0 - k1 * k2 / (1.0 + k3) # Shape (H, L)
        
#         # 5. Discrete Correction
#         K_z = K_s * 2.0 / (1.0 + z.squeeze())
        
#         # Move final kernel back to the correct GPU
#         return K_z.to(target_device)

#     def forward(self, u):
#         """
#         Input u: (Batch, H, L) if transposed=True
#         """
#         # Ensure input is (Batch, Hidden, Length)
#         if not self.transposed:
#             # FIX: Add .contiguous() for memory safety
#             u = u.transpose(1, 2).contiguous()
        
#         B, H, L = u.shape
        
#         # 1. Linear Convolution Trick (Padding)
#         L_fft = 2 * L
#         u_padded = torch.nn.functional.pad(u, (0, L)) # Pad end with L zeros
        
#         # 2. Compute Kernel (Uses CPU-safe helper)
#         K_f = self.kernel_dplr(L_fft) 
        
#         # 3. Convolution via FFT (FIXED: CPU Bypass)
#         # We move input to CPU for the FFT, then move result back to GPU.
        
#         # Fix A: Forward FFT on CPU
#         u_f = torch.fft.fft(u_padded.cpu(), dim=-1).to(u.device)
        
#         # Multiply in Frequency Domain (Fast, stays on GPU)
#         y_f = u_f * K_f 
        
#         # Fix B: Inverse FFT on CPU
#         y = torch.fft.ifft(y_f.cpu(), dim=-1).real.to(u.device)
        
#         # 4. Crop Output back to original length L
#         y = y[..., :L]
        
#         # 5. Skip Connection & Nonlinearities
#         y = y + self.D.unsqueeze(-1) * u
#         y = self.activation(y)
#         y = self.dropout(y)
        
#         # 6. Norm
#         # FIX: Add .contiguous()
#         y = y.transpose(1, 2).contiguous()
#         y = self.norm(y)
        
#         if self.transposed:
#             y = y.transpose(1, 2).contiguous()
            
#         return y

#     def setup_rnn(self):
#         """
#         Converts the Continuous SSM (A, B, C) to Discrete SSM (A_bar, B_bar, C_bar).
#         This runs ONCE before inference.
#         """
#         dt = torch.exp(self.log_step).unsqueeze(-1) # (H, 1)
#         A = self.Lambda # (N)
#         P = self.P.unsqueeze(0) # (1, N)
#         B = self.B.unsqueeze(0) # (1, N)
        
#         # Initialize Lists
#         self.Ab = []
#         self.Bb = []
        
#         I = torch.eye(self.n, device=self.Lambda.device, dtype=torch.cfloat)

#         # Loop over H (Hidden Dim)
#         for h in range(self.h):
#             dt_h = dt[h, 0]
#             p_h = self.P 
#             a_h = self.Lambda
#             b_h = self.B
            
#             # A_dense = diag(Lambda) - P*P^T
#             A_dense = torch.diag(a_h) - torch.outer(p_h, p_h.conj())
            
#             # Bilinear Discretization
#             denom = I - (dt_h / 2.0) * A_dense
#             num = I + (dt_h / 2.0) * A_dense
            
#             inv_denom = torch.linalg.inv(denom)
            
#             A_bar = inv_denom @ num
#             B_bar = inv_denom @ (dt_h * b_h).unsqueeze(1) # (N, 1)
            
#             self.Ab.append(A_bar)
#             self.Bb.append(B_bar)

#         self.Ab = torch.stack(self.Ab) # (H, N, N)
#         self.Bb = torch.stack(self.Bb).squeeze(-1) # (H, N)
#         self.state = None

#     def step(self, u):
#         """
#         Runs one step of the RNN:
#         x_k = A_bar * x_{k-1} + B_bar * u_k
#         y_k = C * x_k
#         """
#         batch_size = u.shape[0]
        
#         if self.state is None:
#             self.state = torch.zeros(batch_size, self.h, self.n, device=u.device, dtype=torch.cfloat)
            
#         # x_prev needs shape (H, B, N, 1)
#         x_prev = self.state.permute(1, 0, 2).contiguous().unsqueeze(-1)
        
#         # Ax term
#         Ax = self.Ab.unsqueeze(1) @ x_prev
        
#         # Bu term
#         u_expanded = u.T.unsqueeze(-1).unsqueeze(-1)
#         Bu = self.Bb.unsqueeze(1).unsqueeze(-1) * u_expanded
        
#         # New State
#         x_next = Ax + Bu
#         self.state = x_next.squeeze(-1).permute(1, 0, 2).contiguous()
        
#         # Compute Output
#         Cx = (self.C.unsqueeze(0) * self.state).sum(dim=-1).real
#         Du = self.D * u
        
#         y = Cx + Du
#         y = self.activation(y)
#         y = self.norm(y)
        
#         return y
import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

def _r2c(x):
    """Helper to convert real tensor to complex safely"""
    return torch.complex(x, torch.zeros_like(x))

class S4(nn.Module):
    def __init__(self, d_model, d_state=64, l_max=1024, dropout=0.2, transposed=True):
        super().__init__()
        self.h = d_model      
        self.n = d_state      
        self.l_max = l_max    
        self.transposed = transposed

        # 1. HiPPO Matrix
        def make_hippo(N):
            def p(n, k):
                if n > k: return np.sqrt(2*n+1)*np.sqrt(2*k+1)
                elif n == k: return n+1
                else: return 0
            return np.array([[p(n,k) for k in range(N)] for n in range(N)])
        
        A = -make_hippo(self.n)

        # 2. NPLR Decomposition
        p = 0.5 * np.sqrt(2 * np.arange(self.n) + 1.0)
        q = 2 * p
        S = A + p[:, np.newaxis] * q[np.newaxis, :]
        eig_vals, eig_vecs = np.linalg.eig(S)
        
        Lambda = torch.tensor(eig_vals, dtype=torch.cfloat)
        V = torch.tensor(eig_vecs, dtype=torch.cfloat)
        V_inv = V.conj().transpose(0, 1)

        # 3. Parameters
        self.register_buffer("Lambda", Lambda)
        self.P = nn.Parameter(V_inv @ torch.tensor(p, dtype=torch.cfloat))
        self.B = nn.Parameter(V_inv @ torch.tensor(np.ones(self.n), dtype=torch.cfloat))
        self.C = nn.Parameter(torch.randn(self.h, self.n, dtype=torch.cfloat))
        self.log_step = nn.Parameter(torch.rand(self.h) - 1.0)
        self.D = nn.Parameter(torch.randn(self.h))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(self.h)
        
        self.state = None

    def kernel_dplr(self, L=None):
        """
        Pure GPU Kernel Generation.
        We add .contiguous() to ensure memory safety.
        """
        if L is None: L = self.l_max

        # Keep everything on GPU
        dt = torch.exp(self.log_step).view(-1, 1, 1)
        Lambda = self.Lambda.view(1, -1, 1)
        P = self.P.view(1, -1, 1)
        B = self.B.view(1, -1, 1)
        C = self.C.view(-1, self.n, 1)
        
        omega = torch.arange(L, device=Lambda.device) / L
        z = torch.exp(-2j * np.pi * omega).view(1, 1, L) * 0.999
        
        s = (2.0 / dt) * (1.0 - z) / (1.0 + z + 1e-6)
        
        denom = s - Lambda
        
        # Calculate terms entirely on GPU
        k0 = (C * B / denom).sum(dim=1) 
        k1 = (C * P / denom).sum(dim=1)
        k2 = (P.conj() * B / denom).sum(dim=1)
        k3 = (P.conj() * P / denom).sum(dim=1)
        
        K_s = k0 - k1 * k2 / (1.0 + k3)
        K_z = K_s * 2.0 / (1.0 + z.squeeze())
        
        # FIX: Ensure kernel is contiguous in memory before returning
        return K_z.contiguous()

    def forward(self, u):
        """
        Pure GPU Forward Pass.
        Strict usage of .contiguous() to try and satisfy the Windows CuFFT backend.
        """
        if not self.transposed:
            # STRICT FIX: Force new memory allocation
            u = u.transpose(1, 2).contiguous()
        
        B, H, L = u.shape
        L_fft = 2 * L
        u_padded = torch.nn.functional.pad(u, (0, L))
        
        # 1. Kernel (GPU)
        K_f = self.kernel_dplr(L_fft) 
        
        # 2. FFT (GPU) - Pure PyTorch call
        u_f = torch.fft.fft(u_padded, dim=-1)
        
        # 3. Multiplication (GPU)
        y_f = u_f * K_f 
        
        # 4. IFFT (GPU)
        y = torch.fft.ifft(y_f, dim=-1).real
        
        # 5. Post-Processing
        y = y[..., :L].contiguous() # Enforce alignment after slicing
        
        y = y + self.D.unsqueeze(-1) * u
        y = self.activation(y)
        y = self.dropout(y)
        
        y = y.transpose(1, 2).contiguous() # Enforce alignment before Norm
        y = self.norm(y)
        
        if self.transposed:
            y = y.transpose(1, 2).contiguous()
            
        return y
    
    # ... (Keep setup_rnn and step methods same as before) ...
    def setup_rnn(self):
        dt = torch.exp(self.log_step).unsqueeze(-1)
        A = self.Lambda
        P = self.P.unsqueeze(0)
        B = self.B.unsqueeze(0)
        self.Ab = []
        self.Bb = []
        I = torch.eye(self.n, device=self.Lambda.device, dtype=torch.cfloat)
        for h in range(self.h):
            dt_h = dt[h, 0]
            p_h = self.P 
            a_h = self.Lambda
            b_h = self.B
            A_dense = torch.diag(a_h) - torch.outer(p_h, p_h.conj())
            denom = I - (dt_h / 2.0) * A_dense
            num = I + (dt_h / 2.0) * A_dense
            inv_denom = torch.linalg.inv(denom)
            A_bar = inv_denom @ num
            B_bar = inv_denom @ (dt_h * b_h).unsqueeze(1)
            self.Ab.append(A_bar)
            self.Bb.append(B_bar)
        self.Ab = torch.stack(self.Ab)
        self.Bb = torch.stack(self.Bb).squeeze(-1)
        self.state = None

    def step(self, u):
        batch_size = u.shape[0]
        if self.state is None:
            self.state = torch.zeros(batch_size, self.h, self.n, device=u.device, dtype=torch.cfloat)
        x_prev = self.state.permute(1, 0, 2).contiguous().unsqueeze(-1)
        Ax = self.Ab.unsqueeze(1) @ x_prev
        u_expanded = u.T.unsqueeze(-1).unsqueeze(-1)
        Bu = self.Bb.unsqueeze(1).unsqueeze(-1) * u_expanded
        x_next = Ax + Bu
        self.state = x_next.squeeze(-1).permute(1, 0, 2).contiguous()
        Cx = (self.C.unsqueeze(0) * self.state).sum(dim=-1).real
        Du = self.D * u
        y = Cx + Du
        y = self.activation(y)
        y = self.norm(y)
        return y