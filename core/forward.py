import torch
import numpy as np
class ForwardModel:
    def __init__(self, gamma, Zc, L=1000.0, device=None):
        self.gamma = gamma
        self.Zc = Zc
        self.L = L
        self.device = device

    def compute_H_from_Y(self, L1, YF, YL):
        # shapes: L1 [N] float32, YF [N] cfloat, YL [N] cfloat
        # Convert 0-D to 1-D
        if L1.dim() == 0: L1 = L1.unsqueeze(0)
        if YF.dim() == 0: YF = YF.unsqueeze(0)
        if YL.dim() == 0: YL = YL.unsqueeze(0)

        dev = self.gamma.device
        N = L1.shape[0]; 
        F = self.gamma.numel()
        L1_nf = L1.unsqueeze(1).to(torch.cfloat)      # [N,1]
        YF_nf = YF.unsqueeze(1)                       # [N,1]
        YL_nf = YL.unsqueeze(1)                       # [N,1]
        gamma = self.gamma.unsqueeze(0)               # [1,F]
        Zc    = self.Zc.unsqueeze(0)                  # [1,F]
        L     = self.L

        tmp1 = gamma * L                              # [1,F]
        tmp2 = gamma * L1_nf                          # [N,F]
        tmp3 = gamma * (L - L1).unsqueeze(1).to(torch.cfloat)

        A1 = torch.cosh(tmp1) + (Zc * YF_nf) * torch.sinh(tmp2) * torch.cosh(tmp3)
        B1 = Zc * torch.sinh(tmp1) + (Zc * Zc) * YF_nf * torch.sinh(tmp2) * torch.sinh(tmp3)

        H = 1.0 / (A1 + B1 * YL_nf)                   # [N,F]
        return H

    def compute_H_complex(self, L1, ZF, ZL):
        """
        Compute forward model for possibly batched inputs. 
        Accepts:
        L1: [..., N] float32
        ZF: [..., N] cfloat
        ZL: [..., N] cfloat
        Returns:
        H:  [..., N, F] cfloat
        """
        dev = self.gamma.device
        F = self.gamma.numel()

        # Convert tensor scalars to 1-D
        if L1.dim() == 0: L1 = L1.unsqueeze(0)
        if ZF.dim() == 0: ZF = ZF.unsqueeze(0)
        if ZL.dim() == 0: ZL = ZL.unsqueeze(0)
        assert L1.shape == ZF.shape == ZL.shape, "L1, ZF, ZL must share the same shape [..., N]"
        *batch, N = L1.shape
        has_batch = len(batch) > 0
        if has_batch:
            B = int(np.prod(batch)) if has_batch else 1
            # Flatten batch dimensions to [B*N]
            L1 = L1.reshape(B*N).to(device=dev, dtype=torch.float32)
            ZF  = ZF.reshape(B*N).to(device=dev, dtype=torch.cfloat)
            ZL  = ZL.reshape(B*N).to(device=dev, dtype=torch.cfloat)
        else:
            L1 = L1.to(device=dev, dtype=torch.float32) #[N]
            ZF  = ZF.to(device=dev, dtype=torch.cfloat) #[N]
            ZL  = ZL.to(device=dev, dtype=torch.cfloat) #[N]

        gamma = self.gamma.unsqueeze(0)  # [1, F]
        Zc    = self.Zc.unsqueeze(0)     # [1, F]
        L     = self.L

        # broadcast to [N, F] via [N,1] x [1,F]
        L1_nf = L1.unsqueeze(1).to(torch.cfloat)          # [N,1]
        ZF_nf = ZF.unsqueeze(1)                           # [N,1]
        ZL_nf = ZL.unsqueeze(1)                           # [N,1]

        # core terms
        tmp1 = gamma * L                                  # [1,F]
        tmp2 = gamma * L1_nf                              # [N,F]
        tmp3 = gamma * (L - L1).unsqueeze(1).to(torch.cfloat)  # [N,F]
        tmp4 = Zc / ZF_nf                                 # [N,F]
        tmp5 = (Zc * Zc) / ZF_nf                          # [N,F]

        A1 = torch.cosh(tmp1) + tmp4 * torch.sinh(tmp2) * torch.cosh(tmp3)      # [N,F]
        B1 = Zc * torch.sinh(tmp1) + tmp5 * torch.sinh(tmp2) * torch.sinh(tmp3) # [N,F]
        H = ZL_nf / (A1 * ZL_nf + B1)                                # [N,F] cfloat
        
        if(has_batch):
            return H.reshape(*batch, N, F)
        else:
            return H


    # def compute_H_from_L1(self, L1, ZF=1000+50j, ZL=100-5j):
    #     if L1.ndim == 0: #Make a scalar a batch of size 1
    #         L1 = L1.unsqueeze(0)
    #     gamma, Zc, L = self.gamma, self.Zc, self.L
    #     ZF = torch.as_tensor(ZF, device=gamma.device)
    #     ZL = torch.as_tensor(ZL, device=gamma.device)

    #     tmp1 = gamma * L  
    #     tmp2 = gamma * L1.unsqueeze(-1) #If L1 is shape [B] (batch of scalar values) then it becomes [B, 1]. Then you can do tmp2 = [B, 1] * [F] -> [B, F]
    #     tmp3 = gamma * (L - L1).unsqueeze(-1)
    #     tmp4 = Zc / ZF
    #     tmp5 = Zc * Zc / ZF

    #     A1 = torch.cosh(tmp1) + tmp4 * torch.sinh(tmp2) * torch.cosh(tmp3)
    #     B1 = Zc * torch.sinh(tmp1) + tmp5 * torch.sinh(tmp2) * torch.sinh(tmp3)
    #     H  = ZL / (A1 * ZL + B1)               # [B,F]
    #     return H.squeeze(0) if H.shape[0] == 1 else H

    # def compute_H_from_ZF(self, ZF, L1=500.0, ZL=100-5j):
    #     """
    #     Varies ZF (scalar or [B]) with fixed L1, ZL.
    #     Returns: [B,F] if ZF is a vector, else [F] if ZF is scalar.
    #     """
    #     # Make scalar a batch of size 1
    #     if torch.is_tensor(ZF):
    #         if ZF.ndim == 0:
    #             ZF = ZF.unsqueeze(0)            # [1]
    #     else:
    #         ZF = torch.tensor([ZF])

    #     gamma, Zc, L = self.gamma, self.Zc, self.L
    #     device = gamma.device

    #     # Cast inputs to tensors on the correct device/dtype
    #     ZF = torch.as_tensor(ZF, dtype=Zc.dtype, device=device)            # [B]
    #     ZL = torch.as_tensor(ZL, dtype=Zc.dtype, device=device)            # scalar complex
    #     L1 = torch.as_tensor(L1, dtype=torch.float32, device=device)       # scalar real

    #     tmp1 = gamma * L                    # [F]
    #     tmp2 = gamma * L1                   # [F]
    #     tmp3 = gamma * (L - L1)             # [F]

    #     # Terms that depend on ZF must be [B,F]
    #     tmp4 = Zc.unsqueeze(0) / ZF.unsqueeze(1)            # [1, F] / [B, 1] = [B,F]
    #     tmp5 = (Zc * Zc).unsqueeze(0) / ZF.unsqueeze(1)     

    #     # Broadcast scalars/frequency-only terms to [B,F]
    #     A1 = torch.cosh(tmp1).unsqueeze(0) + tmp4 * torch.sinh(tmp2).unsqueeze(0) * torch.cosh(tmp3).unsqueeze(0)  # [1,F]
    #     B1 = (Zc * torch.sinh(tmp1)).unsqueeze(0) + tmp5 * torch.sinh(tmp2).unsqueeze(0) * torch.sinh(tmp3).unsqueeze(0)  # [B,F]

    #     H  = ZL / (A1 * ZL + B1)            # [B,F] (ZL broadcasts)

    #     # Return [F] if ZF was scalar, else [B,F]
    #     return H.squeeze(0) if H.shape[0] == 1 else H
    
    # def compute_H_from_ZL(self, ZL, L1=500.0, ZF=1000-5j):
    #     if torch.is_tensor(ZL) and ZL.ndim == 0:
    #         ZL = ZL.unsqueeze(0)
    #     elif not torch.is_tensor(ZL):
    #         ZL = torch.tensor([ZL])

    #     gamma, Zc, L = self.gamma, self.Zc, self.L
    #     device = gamma.device
    #     ZL = torch.as_tensor(ZL, dtype=Zc.dtype, device=device)     # [B]
    #     ZF = torch.as_tensor(ZF, dtype=Zc.dtype, device=device)
    #     L1 = torch.as_tensor(L1, dtype=torch.float32, device=device)

    #     tmp1 = gamma * L
    #     tmp2 = gamma * L1
    #     tmp3 = gamma * (L - L1)

    #     A1 = torch.cosh(tmp1) + (Zc/ZF) * torch.sinh(tmp2) * torch.cosh(tmp3)    # [F]
    #     B1 = Zc*torch.sinh(tmp1) + (Zc*Zc/ZF) * torch.sinh(tmp2) * torch.sinh(tmp3)  # [F]
    #     A1 = A1.unsqueeze(0); B1 = B1.unsqueeze(0)                                 # [1,F]
    #     ZL = ZL.unsqueeze(1)                                                       # [B,1]
    #     H  = ZL / (A1 * ZL + B1)                                                  # [B,F]
    #     return H.squeeze(0) if H.shape[0] == 1 else H

        