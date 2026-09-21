# data/manager.py
import torch


class DatasetManager:
    def __init__(self, device=None):
        self.device = device

    def generate_observations(self, snr_db, N, fm, L1_true, ZF_true, ZL_true):
        """
        Generate N noisy CTF observations given true parameters.

        Args:
            snr_db: Signal-to-noise ratio in dB
            N: Number of observations (M Monte Carlo trials)
            fm: ForwardModel instance (already created with gamma, Zc, L, Zs)
            L1_true: True L1 values [N] (numpy array or scalar)
            ZF_true: True ZF values [N] (complex, numpy array or scalar)
            ZL_true: True ZL values [N] (complex, numpy array or scalar)

        Returns:
            h_obs: Complex observations [N, F]
            var: Noise variance per observation [N, 1]
        """
        # Convert to torch tensors
        L1_t = torch.tensor(L1_true, dtype=torch.float32, device=self.device)
        ZF_t = torch.tensor(ZF_true, dtype=torch.cfloat, device=self.device)
        ZL_t = torch.tensor(ZL_true, dtype=torch.cfloat, device=self.device)

        # Ensure shape is [N] for broadcasting
        if L1_t.dim() == 0:
            L1_t = L1_t.unsqueeze(0).expand(N)
        if ZF_t.dim() == 0:
            ZF_t = ZF_t.unsqueeze(0).expand(N)
        if ZL_t.dim() == 0:
            ZL_t = ZL_t.unsqueeze(0).expand(N)

        # Forward model
        H_clean = fm.compute_H_complex(L1=L1_t, ZF=ZF_t, ZL=ZL_t)  # [N, F]

        # Add noise based on SNR
        snr_lin = 10.0 ** (snr_db / 10.0)
        sigpow = torch.mean(torch.abs(H_clean) ** 2, dim=1, keepdim=True)  # [N, 1]
        var = sigpow / snr_lin  # [N, 1]
        std_f = torch.sqrt(var / 2)  # [N, 1]

        h_obs = H_clean + std_f * torch.randn_like(H_clean.real) + 1j * std_f * torch.randn_like(H_clean.imag)

        return h_obs, var
