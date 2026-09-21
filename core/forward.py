import torch
import numpy as np
"""
2-conductor Transmission Line (TL) Model
"""

def calculate_gamma_zc(frequencies, device=None):
    """
    Calculate propagation constant (gamma) and characteristic impedance (Zc)
    for a 2-conductor transmission line from physical cable parameters.

    Parameters:
    -----------
    frequencies : torch.Tensor or np.ndarray
        Frequencies in Hz, shape [F]
    device : torch.device, optional
        Device for torch tensor output

    Returns:
    --------
    gamma : torch.Tensor [F], cfloat
        Propagation constant
    Zc : torch.Tensor [F], cfloat
        Characteristic impedance
    """
    # Convert to numpy if torch tensor
    if isinstance(frequencies, torch.Tensor):
        frequencies = frequencies.cpu().numpy()

    frequencies = np.asarray(frequencies)
    omega = 2 * np.pi * frequencies

    # Physical constants
    MU0 = 4 * np.pi * 1e-7
    EPS0 = 8.854187817620e-12

    # XLPE cable constants 
    COND_RAD = 0.0039894 #3.9mm
    D = 0.015 #15mm 
    SIGMA = 5.69e7

    STRAND_RAD = 0.000915
    NUM_OUTER_STRANDS = 12  # Number of strands on outer ring of conductor

    EPS_R_XLPE = 2.3 - 1j * 0.001


    delta = 1.0 / np.sqrt(np.pi * frequencies * MU0 * SIGMA)

    r_solid = np.where(
        delta > 2 * COND_RAD,
        1.0 / (SIGMA * np.pi * COND_RAD**2),
        (1.0 / (2 * COND_RAD))
        * np.sqrt(MU0 * frequencies / (np.pi * SIGMA)),
    )

    x_c = (
        NUM_OUTER_STRANDS
        * (
            np.arccos((STRAND_RAD - delta) / STRAND_RAD)
            * STRAND_RAD**2
            - (STRAND_RAD - delta)
            * np.sqrt(
                STRAND_RAD**2
                - (STRAND_RAD - delta)**2
            )
        )
        / (2 * COND_RAD * delta * np.pi)
    )

    r_stranded = r_solid / x_c
    R = 2 * r_stranded

    L = (
        MU0
        / np.pi
        * np.log(D / COND_RAD)
    )
    C_complex = (
        MU0
        * EPS0
        * EPS_R_XLPE
        / L
    )

    C = np.real(C_complex)

    G = -np.imag(C_complex) * omega

    Z = R + 1j * omega * L
    Y = G + 1j * omega * C

    Z_c = np.sqrt(Z / Y)
    gamma = np.sqrt(Z * Y)

    # Convert to torch tensors
    if device is None:
        device = torch.device('cpu')

    gamma_torch = torch.tensor(gamma, dtype=torch.cfloat, device=device)
    Zc_torch = torch.tensor(Z_c, dtype=torch.cfloat, device=device)

    return gamma_torch, Zc_torch


class ForwardModel:
    def __init__(self, frequencies, L, Zs, device=None):
        self.device = device
        gamma, Zc = calculate_gamma_zc(frequencies, device)
        self.gamma = gamma
        self.Zc = Zc
        self.Zs = Zs
        self.L = L

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

        # Convert scalars to 1-D
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
        C1 = torch.sinh(tmp1)/Zc + torch.cosh(tmp2)*torch.cosh(tmp3)/ZF_nf
        D1 = torch.cosh(tmp1) + tmp4*torch.sinh(tmp3)*torch.cosh(tmp2)

        H = ZL_nf / (A1 * ZL_nf + B1)
        #H = ZL_nf / (A1 * ZL_nf + B1 + C1*ZL_nf*self.Zs + D1*self.Zs)   # [N,F] cfloat


        if(has_batch):
            return H.reshape(*batch, N, F)
        else:
            return H
