# data/manager.py
import numpy as np, json, pathlib, torch, hashlib
from core.forward import ForwardModel

class DatasetManager:
    def __init__(self, root="data/datasets", device=None):
        self.root = pathlib.Path(root); 
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = device

    def _tag_from_cfg(self, seed, target, gen_cfg, fixed, freq_cfg, estimate):
        # Taf from cfg params using hashlib
        key = {
            "seed": int(seed),
            "target": str(target).upper(),
            "param_range": gen_cfg, 
            "fixed": fixed,
            "freq": {
                "start_hz": float(freq_cfg["start_hz"]),
                "stop_hz":  float(freq_cfg["stop_hz"]),
                "num_points": int(freq_cfg["num_points"]),
            },
            "estimate": estimate
        }
        s = json.dumps(key, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(s.encode("utf-8"), digest_size=6).hexdigest()

    def _path(self, dataset_id, split, snr_db, target, tag):
        tgt = str(target).lower()
        return self.root / dataset_id / tgt / tag / f"{split}_snr{snr_db}.npz"

    def _save_npz(self, path, **arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    def _load_npz(self, path):
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}

    def build_or_load_dataset(
        self, dataset_id, snr_db, N, gamma, Zc, L=1000.0, seed=5678,
        target=None, fixed=None, gen_cfg=None, freq_cfg=None, force=False, desired_freq=None, estimate=None, split=None
    ):
        tag = self._tag_from_cfg(seed=seed, target=target, gen_cfg=gen_cfg,
                                 fixed=fixed, freq_cfg=freq_cfg, estimate=estimate)
        path = self._path(dataset_id, split, snr_db, target, tag)

        if path.exists() and not force:
            print("Not generating just loading")
            return self._load_npz(path)

        print("Generating data")
        fx = {k: (complex(fixed[k]["re"], fixed[k]["im"]) if isinstance(fixed[k], dict) else fixed[k])
              for k in fixed}
        tgt = str(target).upper()

        rng = np.random.default_rng(seed ^ (snr_db + 12345)) #Seed and SNR control the randomness used to select true parameter values
        if tgt == "L1":
            lo, hi = float(gen_cfg["L1"]["min"]), float(gen_cfg["L1"]["max"])
            L1_true = rng.uniform(lo, hi, size=N).astype(np.float32)
            ZF_true = np.full(N, fx["ZF"], dtype=np.complex64)
            ZL_true = np.full(N, fx["ZL"], dtype=np.complex64)
        elif tgt == "ZF":
            g = gen_cfg["ZF"]
            if estimate == "real":
                re = rng.uniform(g["re"]["min"], g["re"]["max"], size=N).astype(np.float32)
                im = np.full(N, np.imag(fx["ZF"]), dtype=np.float32)
            elif estimate == "imag":
                re = np.full(N, np.real(fx["ZF"]), dtype=np.float32)
                im = rng.uniform(g["im"]["min"], g["im"]["max"], size=N).astype(np.float32)
            else:  # full complex
                re = rng.uniform(g["re"]["min"], g["re"]["max"], size=N).astype(np.float32)
                im = rng.uniform(g["im"]["min"], g["im"]["max"], size=N).astype(np.float32)
            ZF_true = (re + 1j * im).astype(np.complex64)
            L1_true = np.full(N, float(fx["L1"]), dtype=np.float32)
            ZL_true = np.full(N, fx["ZL"], dtype=np.complex64)
        elif tgt == "ZL":  # "ZL"
            g = gen_cfg["ZL"]
            if estimate == "real":
                re = rng.uniform(g["re"]["min"], g["re"]["max"], size=N).astype(np.float32)
                im = np.full(N, np.imag(fx["ZL"]), dtype=np.float32)
            elif estimate == "imag":
                re = np.full(N, np.real(fx["ZL"]), dtype=np.float32)
                im = rng.uniform(g["im"]["min"], g["im"]["max"], size=N).astype(np.float32)
            else:  # full complex
                re = rng.uniform(g["re"]["min"], g["re"]["max"], size=N).astype(np.float32)
                im = rng.uniform(g["im"]["min"], g["im"]["max"], size=N).astype(np.float32)
            ZL_true = (re + 1j * im).astype(np.complex64)
            L1_true = np.full(N, float(fx["L1"]), dtype=np.float32)
            ZF_true = np.full(N, fx["ZF"], dtype=np.complex64)
        elif tgt == "ALL3DP": #ALL 3 diff params
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
        else: #ALL3SP
            L1_true = np.full(N, float(fx["L1"]), dtype=np.float32)
            ZF_true = np.full(N, fx["ZF"], dtype=np.complex64)
            ZL_true = np.full(N, fx["ZL"], dtype=np.complex64)




        # ---- forward + noise ----
        fm = ForwardModel(gamma, Zc, L=L, device=self.device)

        L1_t = torch.tensor(L1_true, dtype=torch.float32, device=self.device)   # [N]
        ZF_t = torch.tensor(ZF_true, dtype=torch.cfloat,  device=self.device)   # [N]
        ZL_t = torch.tensor(ZL_true, dtype=torch.cfloat,  device=self.device)   # [N]
        H_true = fm.compute_H_complex(L1=L1_t, ZF=ZF_t, ZL=ZL_t)                 # [N,F] cfloat

        # import matplotlib.pyplot as plt
        # plt.title(f'ZF = {ZF_t[0]} at SNR = {snr_db}')
        # plt.plot((desired_freq/1e6).cpu().numpy(),20*torch.log10(torch.abs(H_true[0])).cpu().numpy())
        # plt.show()
        # plt.title(f'ZF = {ZF_t[20]} at SNR = {snr_db}')
        # plt.plot((desired_freq/1e6).cpu().numpy(),20*torch.log10(torch.abs(H_true[20])).cpu().numpy())
        # plt.show()
        # plt.title(f'ZF = {ZF_t[40]} at SNR = {snr_db}')
        # plt.plot((desired_freq/1e6).cpu().numpy(),20*torch.log10(torch.abs(H_true[40])).cpu().numpy())
        # plt.show()
        # H_true2 = fm.compute_H_from_L1(L1=torch.tensor(200.0, device=self.device))

        snr_lin = 10.0 ** (snr_db / 10.0)

        
        #Option A: calibrate to a fixed "reference" TF at nominal params (fx)
        # ref_L1 = torch.tensor(float(fx["L1"]), dtype=torch.float32, device=self.device)
        # ref_ZF = torch.tensor(fx["ZF"],        dtype=torch.cfloat,  device=self.device)
        # ref_ZL = torch.tensor(fx["ZL"],       dtype=torch.cfloat,  device=self.device)

        # H_ref  = fm.compute_H_complex(L1=ref_L1.unsqueeze(0),
        #                             ZF=ref_ZF.unsqueeze(0),
        #                             ZL=ref_ZL.unsqueeze(0)).squeeze(0)  # [F]
        # P_ref  = torch.mean(torch.abs(H_ref)**2).real.to(torch.float32)   # scalar power
        # sigma2 = (P_ref / snr_lin)                                        # scalar σ²
        # var_f  = sigma2 * torch.ones_like(H_true.real)                     # [N, F]
        # std_f  = torch.sqrt(sigma2 / 2.0)                                  # scalar


        sigpow  = torch.mean(torch.abs(H_true)**2, dim=1, keepdim=True)          # [N,1]
        var_f   = (sigpow / snr_lin).real.expand_as(H_true)                      # [N,F]
        std_f   = torch.sqrt(var_f / 2)
        
        obs     = H_true + std_f*torch.randn_like(H_true.real) + 1j*std_f*torch.randn_like(H_true.imag)
        data = dict(
            h_obs_real = obs.real.cpu().numpy(),
            h_obs_imag = obs.imag.cpu().numpy(),
            h_true_real = H_true.real.cpu().numpy(),
            h_true_imag = H_true.imag.cpu().numpy(),
            noise_var = var_f.cpu().numpy(),
            L1_true = L1_true,
            ZF_true_re = np.asarray(np.real(ZF_true), dtype=np.float32),
            ZF_true_im = np.asarray(np.imag(ZF_true), dtype=np.float32),
            ZL_true_re = np.asarray(np.real(ZL_true), dtype=np.float32),
            ZL_true_im = np.asarray(np.imag(ZL_true), dtype=np.float32),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **data)
        return data
   

    def build_or_load_pooled_dataset(
        self, dataset_id, snr_list, N_per_snr,
        gamma, Zc, L=1000.0, seed=5678,
        target=None, fixed=None, gen_cfg=None, freq_cfg=None,
        force=False, desired_freq=None, estimate=None, split="train", include_snr_feature=True
    ):
        """Returns X_train = [N_train x len(snr_list), 2F+1]
                   y_train = [N_train x len(snr_list), D]
                   where D is number of parameters we are estimating."""
        X_list, y_list = [], []

        #gen_cfg is grid range not true parameter range so that training data does not have a prior
        for snr_db in snr_list:
            data = self.build_or_load_dataset(
                dataset_id=dataset_id, split=split,
                snr_db=snr_db, N=N_per_snr,
                gamma=gamma, Zc=Zc, L=L, seed=seed + (777 if split=="train" else 0),
                target=target, fixed=fixed, gen_cfg=gen_cfg, freq_cfg=freq_cfg,
                force=force, desired_freq=desired_freq, estimate=estimate
            )
            ri = np.stack([data["h_obs_real"], data["h_obs_imag"]], axis=2)  # [N,F,2] numpy array
            X = ri.reshape(ri.shape[0], -1).astype(np.float32) #[N, 2F]

            if include_snr_feature:
                # Per-sample SNR estimate (in dB) from observed power and provided noise_var
                snr_col = np.full((X.shape[0], 1), float(snr_db), dtype=np.float32) #[N, 1]
                X = np.concatenate([X, snr_col], axis=1) #[N, 2F]
            # labels (1D target)
            t = str(target).upper()
            if t == "L1":
                y = data["L1_true"]
            elif t in ("ZF", "ZL"):
                if estimate == "real":
                    y = data[f"{t}_true_re"]
                elif estimate == "imag":
                    y = data[f"{t}_true_im"]
                else:
                    raise NotImplementedError("Full complex target not returned here; use multi-output later.")
            else:
                raise ValueError(f"Unknown target: {target}")

            X_list.append(X)
            y_list.append(y)
        X_train = np.concatenate(X_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)
        return X_train, y_train
