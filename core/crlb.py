# crlb.py
import torch
from torch.func import vmap, jacfwd

def u_ri_single(fm, zfr, zfi, zlr, zli, l1):
    """
    ONE-sample forward that returns REAL output [F,2] = [Re u, Im u]
    so jacfwd can differentiate.

    Inputs are scalars (0-D) in float32; we build [1]-length tensors
    for compute_H_complex, then squeeze back to [F].
    """
    # ensure float32 everywhere
    zfr = zfr.to(torch.float32); zfi = zfi.to(torch.float32)
    zlr = zlr.to(torch.float32); zli = zli.to(torch.float32)
    l1  = l1.to(torch.float32)

    ZF = torch.complex(zfr, zfi).unsqueeze(0).to(torch.cfloat)  # [1]
    ZL = torch.complex(zlr, zli).unsqueeze(0).to(torch.cfloat)  # [1]
    L1 = l1.unsqueeze(0)                                        # [1] float32

    # compute_H_complex expects [N] tensors → returns [1,F]
    u = fm.compute_H_complex(L1=L1, ZF=ZF, ZL=ZL)[0]            # [F] cfloat

    # Real-valued for jacfwd
    return torch.stack((u.real, u.imag), dim=-1)                # [F,2] float32


# Forward-mode Jacobian wrt real parameters (ZF_re, ZF_im, ZL_re, ZL_im, L1)
jac_fwd_single = jacfwd(u_ri_single, argnums=(1, 2, 3, 4, 5))

@torch.no_grad()
def complex_partials_fullbatch(fm, test, device):
    """
    Build complex partials for ALL samples at once.

    Returns:
      du_aug: [N, F, 5] complex,  columns = [∂u/∂ZF, ∂u/∂ZL, ∂u/∂ZF*, ∂u/∂ZL*, ∂u/∂L1]
    """
    # use float32 to match compute_H_complex dtypes and keep JVPs alive
    ZF_re = torch.tensor(test["ZF_true_re"], device=device, dtype=torch.float32)
    ZF_im = torch.tensor(test["ZF_true_im"], device=device, dtype=torch.float32)
    ZL_re = torch.tensor(test["ZL_true_re"], device=device, dtype=torch.float32)
    ZL_im = torch.tensor(test["ZL_true_im"], device=device, dtype=torch.float32)
    L1    = torch.tensor(test["L1_true"],    device=device, dtype=torch.float32)

    # Vectorize jacobian over Nters
    # Each of the 5 outputs is [N, F, 2] where last dim = (∂Re u, ∂Im u) with respect to that parameter
    dri_ZF_re, dri_ZF_im, dri_ZL_re, dri_ZL_im, dri_L1 = vmap(
    jac_fwd_single, in_dims=(None, 0, 0, 0, 0, 0)
    )(fm, ZF_re, ZF_im, ZL_re, ZL_im, L1)

    # Convert to COMPLEX partials: ∂u = dRe(u) + i dIm(u), ∂u* = dRe(u) - i dIm(u)
    def ri_to_cplx(d_ri):  # [N,F,2] -> [N,F] cfloat
        du     = d_ri[..., 0] + 1j*d_ri[..., 1]   # ∂u/∂θ
        #du_star= d_ri[..., 0] - 1j*d_ri[..., 1]   # ∂u*/∂θ
        return du
    

    du_ZF_re = ri_to_cplx(dri_ZF_re)
    du_ZF_im = ri_to_cplx(dri_ZF_im)
    du_ZL_re = ri_to_cplx(dri_ZL_re)
    du_ZL_im = ri_to_cplx(dri_ZL_im)
    du_L1 = ri_to_cplx(dri_L1)

    # Wirtinger definitions
    du_dZF   = 0.5 * (du_ZF_re - 1j * du_ZF_im)   # ∂u/∂ZF = 0.5(∂u/∂Re[ZF] - j ∂u/∂Im[ZF])
    du_dZF_c = 0.5 * (du_ZF_re + 1j * du_ZF_im)   # ∂u/∂ZF*
    du_dZL   = 0.5 * (du_ZL_re - 1j * du_ZL_im)   # ∂u/∂ZL
    du_dZL_c = 0.5 * (du_ZL_re + 1j * du_ZL_im)   # ∂u/∂ZL*
    du_dL1   = du_L1                              # ∂u/∂L1

    # du_star_dZF   = 0.5 * (du_star_ZF_re - 1j * du_star_ZF_im)   # ∂u*/∂ZF
    # du_star_dZF_c = 0.5 * (du_star_ZF_re + 1j * du_star_ZF_im)   # ∂u*/∂ZF*
    # du_star_dZL   = 0.5 * (du_star_ZL_re - 1j * du_star_ZL_im)   # ∂u*/∂ZL
    # du_star_dZL_c = 0.5 * (du_star_ZL_re + 1j * du_star_ZL_im)   # ∂u*/∂ZL*
    # du_star_dL1   = du_star_L1                              # ∂u*/∂L1

    # Stack in augmented parameter order
    du_aug = torch.stack([du_dZF, du_dZL, du_dZF_c, du_dZL_c, du_dL1], dim=-1)  # [N,F,5]
    # du_star_aug = torch.stack([du_star_dZF, du_star_dZL, du_star_dZF_c, du_star_dZL_c, du_star_dL1], dim=-1)  # [N,F,5]
    return du_aug 


@torch.no_grad()
def debug_jacobian_mags(fm, test, device):
    """Print summary stats of |∂u/∂ZF_re|, |∂u/∂ZF_im|, |∂u/∂L1|."""
    ZF_re = torch.tensor(test["ZF_true_re"], device=device, dtype=torch.float32)
    ZF_im = torch.tensor(test["ZF_true_im"], device=device, dtype=torch.float32)
    ZL_re = torch.tensor(test["ZL_true_re"], device=device, dtype=torch.float32)
    ZL_im = torch.tensor(test["ZL_true_im"], device=device, dtype=torch.float32)
    L1    = torch.tensor(test["L1_true"],    device=device, dtype=torch.float32)

    dri_ZF_re, dri_ZF_im, _, _, d_ri_L1 = vmap(
        jac_fwd_single, in_dims=(None, 0, 0, 0, 0, 0)
    )(fm, ZF_re, ZF_im, ZL_re, ZL_im, L1)  # each [N,F,2]
    #print("dri_ZF_re", dri_ZF_re)
    #print("dri_ZF_im", dri_ZF_im)
    #print("d_ri_L1", d_ri_L1)
    def to_c(dri): return dri[...,0] + 1j*dri[...,1]  # [N,F] complex
    g_zfr = to_c(dri_ZF_re)
    g_zfi = to_c(dri_ZF_im)
    g_L1  = to_c(d_ri_L1)

    # norms per sample (sum over F)
    n_zfr = (g_zfr.abs()**2).sum(dim=1).sqrt()
    n_zfi = (g_zfi.abs()**2).sum(dim=1).sqrt()
    n_L1  = (g_L1.abs()**2).sum(dim=1).sqrt()

    print("[∥∂u/∂ZF_re∥_2]_mean =", n_zfr.mean().item(),
          " median =", n_zfr.median().item())
    print("[∥∂u/∂ZF_im∥_2]_mean =", n_zfi.mean().item(),
          " median =", n_zfi.median().item())
    print("[∥∂u/∂L1∥_2]_mean   =", n_L1.mean().item(),
          " median =", n_L1.median().item(), flush=True)

    # optional: relative sensitivity (scale by parameter magnitude)
    ZF = torch.complex(ZF_re, ZF_im)
    rel_zf = ( (g_zfr.abs()**2 + g_zfi.abs()**2).sum(dim=1).sqrt() /
               (fm.compute_H_complex(L1=L1, ZF=ZF, ZL=torch.complex(ZL_re, ZL_im)).abs()**2).sum(dim=1).sqrt().clamp_min(1e-12) )
    print("[relative sensitivity ZF]_mean =", rel_zf.mean().item())


@torch.no_grad()
def fim_from_complex_jac(du_aug, var_NF):
    """
    Build the per-sample augmented complex FIM from Wirtinger Jacobians.

    Inputs
    ------
    du_aug : [N, F, 5] complex
        Columns must be in GROUPED order:
            [ ∂u/∂ZF , ∂u/∂ZL , ∂u/∂ZF* , ∂u/∂ZL* , ∂u/∂L1 ].
        (ZF, ZL are complex parameters; L1 is a real parameter.)

    var_NF : [N, F] or [F] real
        Noise variance σ² per (sample, frequency). Will broadcast to [N, F].

    Returns
    -------
    I_n : [N, 5, 5] complex
        Augmented complex Fisher information matrix per sample.

    ----------------------------------------------------------------------
    Key Wirtinger identities for a conjugated output u* (z complex, r real):
        ∂u*/∂z   = ( ∂u/∂z* )*,
        ∂u*/∂z*  = ( ∂u/∂z  )*,
        ∂u*/∂r   = ( ∂u/∂r  )*.

    With the **GROUPED** column order [z, z*, r] realized here as
        [ZF, ZL, ZF*, ZL*, L1],
    the Jacobian of u* in the **same** column order is obtained by
    swapping each (z, z*) pair and then conjugating:

        perm = [2, 3, 0, 1, 4]     # (ZF ↔ ZF*), (ZL ↔ ZL*), keep L1
    """
    perm_grouped = torch.tensor([2, 3, 0, 1, 4], device=du_aug.device)    
    du_star_from_du = du_aug.index_select(-1, perm_grouped).conj()
    # ok = torch.allclose(du_star_from_du, du_star_aug, rtol=1e-6, atol=1e-8)
    # print("du_star_aug matches swap+conj(du_aug):", ok)
    # if not ok:
    #     diff = (du_star_from_du - du_star_aug).abs()
    #     print("max abs diff:", diff.max().item())
    f = du_aug.unsqueeze(-1)                     # [N,F,5,1]  du/dθ_aug
    i = du_star_from_du.unsqueeze(-1)                  # [N,F,5,1]  du*/dθ_aug
    I_nf = (f.conj() @ f.transpose(-1, -2)) + (i.conj() @ i.transpose(-1, -2))  # [N,F,5,5] tranpose(-1, -2) swaps last and second last dimensions 5x1 -> 1x5
    w = (1.0 / var_NF).unsqueeze(-1).unsqueeze(-1)                                # [N,F,1,1]
    I_n = (I_nf * w).sum(dim=1)                                                    # [N,5,5]   and sum over F dimension 
    #print("VAR NF", var_NF)
    # import numpy as np
    # print("1/ZF I_n[:, 0,0]", np.sqrt((1/I_n[:, 0, 0]).mean().item()))
    # print("ZF I_n[0,0,0]", I_n[0, 0, 0])
    # print("ZL I_n[0,1,1]", I_n[0, 1, 1])
    # print("ZF* I_n[0,2,2]", I_n[0, 2, 2])
    # print("ZL* I_n[0,3,3]", I_n[0, 3, 3])
    # print("L1 I_n[0,4,4]", I_n[0, 4, 4])
    return I_n

@torch.no_grad()
def get_CRLB(FIM_total):
    """
    Compute CRLBs from the augmented complex FIM for parameters ordered as
        theta = [theta1, theta1*, theta2]
    where theta1 = [ZF, ZL] (complex, length 2) and theta2 = [L1] (real, length 1).

    Args:
        FIM_total: [N, 5, 5] complex tensor 

    Returns:
        crlb_L1:  [...,]       real tensor (variance lower bound for L1)
        crlb_ZF:  [...,]       real tensor (variance lower bound for ZF)
        crlb_ZL:  [...,]       real tensor (variance lower bound for ZL)
    """

    # Block slicing (assumes ordering [theta1, theta1*, theta2] with sizes 2,2,1)
    A      = FIM_total[..., 0:2, 0:2]   # [N,2,2]
    A_conj = FIM_total[..., 2:4, 2:4]   # [N,2,2] == A.conj()
    B     = FIM_total[..., 2:4, 0:2]    # [N,2,2]
    B_conj= FIM_total[..., 0:2, 2:4]    # [N,2,2] == B.conj()
    P     = FIM_total[..., 4:5, 0:2]    # [N,1,2]
    P_conj= FIM_total[..., 4:5, 2:4]    # [N,1,2] == P.conj()
    P_H   = FIM_total[..., 0:2, 4:5]    # [N,2,1] == P.conj().transpose(-1,-2)
    P_T   = FIM_total[..., 2:4, 4:5]    # [N,2,1] == P.transpose(-1,-2)
    Q     = FIM_total[..., 4:5, 4:5]    # [N,1,1] (real)

    I2 = torch.eye(2, dtype=A.dtype, device=A.device).expand(A.shape[:-2] + (2, 2))

    # Helper: solve instead of inv for stability
    # X = A_conj^{-1} B
    X = torch.linalg.solve(A_conj, B)                 # [N,2,2]
    # S = A - B^* A_conj^{-1} B
    S = A - B_conj @ X                                # [N,2,2]
    # C = S^{-1}
    C = torch.linalg.solve(S, I2)                     # [N,2,2]
    # D = - A_conj^{-1} B C = - X C
    D = -(X @ C)                                      # [N,2,2]

    # CRLB for real block (L1): CRLB_r = (Q - 2 Re[ P C P^H + P^* D P^H ])^{-1}
    term1 = P @ C @ P_H                               # [N, 1, 2] x [N, 2, 2,] x [N, 2, 1] = [N,1,1]
    term2 = P_conj @ D @ P_H                          # [N, 1, 2] x [N, 2, 2] x [N,2, 1] = [N,1,1]
    denom = (Q - 2.0 * (term1 + term2).real)          # [N,1,1]
    crlb_L1 = 1.0 / denom.squeeze(-1).squeeze(-1)     # [N,]

    # CRLB for complex block (ZF,ZL):
    # CRLB_c = C + (C P^H + D^* P^T) * CRLB_r * (P C^H + P^* D^T)
    left  = C @ P_H + D.conj() @ P_T                  # [N, 2, 2] x [N, 2, 1] + [N, 2, 2] x [N,2,1] = [N, 2, 1]
    right = P @ C.conj().transpose(-1, -2) + P_conj @ D.transpose(-1, -2)  # [N,1,2]
    crlb_c = C + left @ crlb_L1.unsqueeze(-1).unsqueeze(-1) @ right      # [N,2,2]
    #print("crlb_c", crlb_c)
    # Diagonal variances (ZF, ZL). 
    diag_c = crlb_c.diagonal(offset=0, dim1=-2, dim2=-1).real  # [N,2]
    crlb_ZF = diag_c[..., 0]                                   # [N,]
    crlb_ZL = diag_c[..., 1]                                   # [N,]


    return crlb_L1, crlb_ZF, crlb_ZL


@torch.no_grad()
def crlb_L1_only_batch(fm, test, var_NF, eps=1e-12):
    """
    Batched CRLB for L1 only, over all samples in `test`.

    Inputs
    ------
    fm: ForwardModel
    test: dict with keys
        "ZF_true_re", "ZF_true_im", "ZL_true_re", "ZL_true_im", "L1_true"
        Each is [N] float32 (as produced elsewhere in this file).
    var_NF: [N, F] or [F] real tensor (noise variance per frequency, broadcastable)

    Returns
    -------
    FI_L1:   [N] real tensor (Fisher Information per sample)
    CRLB_L1: [N] real tensor
    """
    device = fm.gamma.device
    ZF_re = torch.as_tensor(test["ZF_true_re"], device=device, dtype=torch.float32)
    ZF_im = torch.as_tensor(test["ZF_true_im"], device=device, dtype=torch.float32)
    ZL_re = torch.as_tensor(test["ZL_true_re"], device=device, dtype=torch.float32)
    ZL_im = torch.as_tensor(test["ZL_true_im"], device=device, dtype=torch.float32)
    L1    = torch.as_tensor(test["L1_true"],    device=device, dtype=torch.float32)

    # Build ZF, ZL complex [N]
    ZF = torch.complex(ZF_re, ZF_im).to(torch.cfloat)
    ZL = torch.complex(ZL_re, ZL_im).to(torch.cfloat)

    # We just need the derivative wrt L1: reuse jac_fwd_single via vmap
    # Outputs: each arg Jacobian is [N, F, 2]; we only keep the L1 piece.
    _, _, _, _, d_ri_L1 = vmap(jac_fwd_single, in_dims=(None, 0, 0, 0, 0, 0))(
        fm, ZF_re, ZF_im, ZL_re, ZL_im, L1
    )  # [N, F, 2] with last dim (dRe, dIm)

    dRe = d_ri_L1[..., 0]   # [N, F]
    dIm = d_ri_L1[..., 1]   # [N, F]

    # Broadcast var_NF to [N, F]
    if var_NF.ndim == 1:
        var_NF = var_NF.unsqueeze(0).expand_as(dRe)
    # FI per-sample
    FI_L1 = torch.sum(2.0 * (dRe**2 + dIm**2) / var_NF, dim=-1)  # [N]
    CRLB_L1 = 1.0 / FI_L1.clamp_min(eps)                         # [N]
    return FI_L1, CRLB_L1




@torch.no_grad()
def crlb_for_target_estimate(du_aug: torch.Tensor,
                             var_NF: torch.Tensor,
                             target: str,
                             estimate=None) -> torch.Tensor:
    """
    Returns per-sample CRLB [N] for the requested parameterization.

    du_aug : [N,F,5] complex with columns
             [∂u/∂ZF, ∂u/∂ZL, ∂u/∂ZF*, ∂u/∂ZL*, ∂u/∂L1]
    var_NF : [N,F] or [F] real
    target : "L1" | "ZF" | "ZL"
    estimate : None | "real" | "imag" |
        - L1: ignored (always scalar real)
        - ZF/ZL + "real":  CRLB(Re)  = 1 / I11
        - ZF/ZL + "imag":  CRLB(Im)  = 1 / I22
        - ZF/ZL + None:    complex MSE CRLB = trace(inv(I2)) = (I11+I22)/(I11*I22 - I12^2)
    """
    t = str(target).upper()
    e = None if estimate is None else str(estimate).lower()
    # Broadcast var to [N,F]
    if var_NF.ndim == 1:
        var = var_NF.unsqueeze(0).expand(du_aug.shape[0], -1)
    else:
        var = var_NF
    
    # L1: FI = Σ_f 2|∂u/∂L1|^2 / σ^2
    if t == "L1":
        du_L1 = du_aug[..., 4]
        FI = (2.0 * (du_L1.conj() * du_L1).real / var).sum(dim=1)
        return 1.0 / FI  # [N]
    
    # Choose the right Wirtinger columns
    if t == "ZF":
        du, duc = du_aug[..., 0], du_aug[..., 2]  # ∂u/∂ZF, ∂u/∂ZF*
    elif t == "ZL":
        du, duc = du_aug[..., 1], du_aug[..., 3]  # ∂u/∂ZL, ∂u/∂ZL*
    else:
        raise ValueError(f"Unknown target: {target}")
    
    # Real-parameter gradients for (z_r, z_i)
    dzr = du + duc            # ∂u/∂z_r 
    dzi = -1j*du + 1j*duc     # ∂u/∂z_i 

    w   = 2.0 / var
    I11 = (w * (dzr.conj()*dzr).real).sum(dim=1)  # [N]
    I22 = (w * (dzi.conj()*dzi).real).sum(dim=1)  # [N]
    I12 = (w * (dzr.conj()*dzi).real).sum(dim=1)                 # [N]

    if e == "real":
        return 1.0 / I11
    if e == "imag":
        return 1.0 / I22

    # Full complex param: CRLB for complex MSE = trace(inv(I2))
    det = (I11*I22 - I12**2)
    return (I11 + I22) / det    # [N]
