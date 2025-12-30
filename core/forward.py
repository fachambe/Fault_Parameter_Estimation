import torch
import numpy as np
class ForwardModel:
    def __init__(self, gamma, Zc, L=1000.0, device=None):
        self.gamma = gamma
        self.Zc = Zc
        self.L = L
        self.device = device


    def compute_H_complex(self, L1, ZF, ZL):
        """
        Compute transfer function of the simple forward model for batched inputs. 
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

