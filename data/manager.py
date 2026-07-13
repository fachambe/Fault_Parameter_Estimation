# data/manager.py
import numpy as np
import torch


class DatasetManager:
    def __init__(self, device=None):
        self.device = device

    def generate_observations(
        self, snr_db, N, fm, seed=5678,
        target=None, fixed=None, gen_cfg=None
    ):
        """
        Generate N CTF observations.

        Args:
            snr_db: Signal-to-noise ratio in dB
            N: Number of observations to generate
            fm: ForwardModel instance (already created with gamma, Zc, L, Zs)
            seed: Random seed for reproducibility
            target: "ALL3SP" (frequentist) or "ALL3DP" (Bayesian)
            fixed: Dict with fixed parameter values
            gen_cfg: Dict with parameter ranges for Bayesian case

        Returns:
            dict with observation data arrays
        """
        fx = {k: (complex(fixed[k]["re"], fixed[k]["im"]) if isinstance(fixed[k], dict) else fixed[k])
              for k in fixed}
        tgt = str(target).upper()

        rng = np.random.default_rng(seed)
        if tgt == "ALL3DP":  # Bayesian: random params per observation
            g1 = gen_cfg["L1"]
            g2 = gen_cfg["ZF"]
            g3 = gen_cfg["ZL"]
            L1_true = rng.uniform(g1["min"], g1["max"], size=N).astype(np.float32)
            ZF_re_true = rng.uniform(g2["re"]["min"], g2["re"]["max"], size=N).astype(np.float32)
            ZF_im_true = rng.uniform(g2["im"]["min"], g2["im"]["max"], size=N).astype(np.float32)
            ZF_true = (ZF_re_true + 1j * ZF_im_true).astype(np.complex64)
            ZL_re_true = rng.uniform(g3["re"]["min"], g3["re"]["max"], size=N).astype(np.float32)
            ZL_im_true = rng.uniform(g3["im"]["min"], g3["im"]["max"], size=N).astype(np.float32)
            ZL_true = (ZL_re_true + 1j * ZL_im_true).astype(np.complex64)
        else:  # ALL3SP: same fixed params for all observations
            L1_true = np.full(N, float(fx["L1"]), dtype=np.float32)
            ZF_true = np.full(N, fx["ZF"], dtype=np.complex64)
            ZL_true = np.full(N, fx["ZL"], dtype=np.complex64)

        # Forward model + noise
        L1_t = torch.tensor(L1_true, dtype=torch.float32, device=self.device)
        ZF_t = torch.tensor(ZF_true, dtype=torch.cfloat, device=self.device)
        ZL_t = torch.tensor(ZL_true, dtype=torch.cfloat, device=self.device)

        H_true = fm.compute_H_complex(L1=L1_t, ZF=ZF_t, ZL=ZL_t)  # [N,F]
        snr_lin = 10.0 ** (snr_db / 10.0)
        sigpow = torch.mean(torch.abs(H_true) ** 2, dim=1, keepdim=True)  # [N,1]
        var_f = sigpow / snr_lin  # [N,1]
        std_f = torch.sqrt(var_f / 2)  # [N,1]
        obs = H_true + std_f * torch.randn_like(H_true.real) + 1j * std_f * torch.randn_like(H_true.imag)

        return dict(
            h_obs_real=obs.real.cpu().numpy(),
            h_obs_imag=obs.imag.cpu().numpy(),
            h_true_real=H_true.real.cpu().numpy(),
            h_true_imag=H_true.imag.cpu().numpy(),
            noise_var=var_f.cpu().numpy(),
            L1_true=L1_true,
            ZF_true_re=np.asarray(np.real(ZF_true), dtype=np.float32),
            ZF_true_im=np.asarray(np.imag(ZF_true), dtype=np.float32),
            ZL_true_re=np.asarray(np.real(ZL_true), dtype=np.float32),
            ZL_true_im=np.asarray(np.imag(ZL_true), dtype=np.float32),
        )
