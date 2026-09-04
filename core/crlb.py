# crlb.py
import torch
import numpy as np
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
def fim_from_complex_jac(du_aug, var_NF):
    """
    Build the per-sample augmented complex FIM from Wirtinger Jacobians.

    Inputs
    ------
    du_aug : [N, F, 5] complex
        Columns must be in GROUPED order:
            [ ∂u/∂ZF , ∂u/∂ZL , ∂u/∂ZF* , ∂u/∂ZL* , ∂u/∂L1 ].
        (ZF, ZL are complex parameters; L1 is a real parameter.)

    var_NF : [N, F] or [N, 1] real
        Noise variance

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


    return crlb_L1.real, crlb_ZF, crlb_ZL


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


def crlb_for_1_real_param(fm, target, fixed, var_f, device):
    """
    Compute CRLB for a single real parameter using wrapper + jacfwd approach.

    Uses the formula: I(θ) = (2/σ²) Σ_f |∂H(f;θ)/∂θ|²

    Args:
        fm: Forward model with compute_H_complex method
        target: "L1" | "ZF_re" | "ZF_im" | "ZL_re" | "ZL_im"
        fixed: Dict with true parameter values to evaluate at, e.g.:
               {"L1": 250.0, "ZF": {"re": 100.0, "im": -50.0}, "ZL": {"re": 100.0, "im": -5.0}}
        var_f: Noise variance - scalar or [F] tensor
        device: torch device

    Returns:
        FI: scalar tensor - Fisher Information
        CRLB: scalar tensor - Cramer-Rao Lower Bound
    """
    # Get fixed values as tensors
    L1_fixed = torch.tensor(fixed["L1"], device=device, dtype=torch.float32)
    ZF_re_fixed = torch.tensor(fixed["ZF"]["re"], device=device, dtype=torch.float32)
    ZF_im_fixed = torch.tensor(fixed["ZF"]["im"], device=device, dtype=torch.float32)
    ZL_re_fixed = torch.tensor(fixed["ZL"]["re"], device=device, dtype=torch.float32)
    ZL_im_fixed = torch.tensor(fixed["ZL"]["im"], device=device, dtype=torch.float32)

    # Create wrapper that takes single real param and returns H as [F, 2] real
    if target == "L1":
        true_val = L1_fixed
        def wrapper(theta):
            ZF = torch.complex(ZF_re_fixed, ZF_im_fixed)
            ZL = torch.complex(ZL_re_fixed, ZL_im_fixed)
            H = fm.compute_H_complex(theta.unsqueeze(0), ZF.unsqueeze(0), ZL.unsqueeze(0))[0]  # [F]
            return torch.stack([H.real, H.imag], dim=-1)  # [F, 2]
    elif target == "ZF_re":
        true_val = ZF_re_fixed
        def wrapper(theta):
            ZF = torch.complex(theta, ZF_im_fixed)
            ZL = torch.complex(ZL_re_fixed, ZL_im_fixed)
            H = fm.compute_H_complex(L1_fixed.unsqueeze(0), ZF.unsqueeze(0), ZL.unsqueeze(0))[0]
            return torch.stack([H.real, H.imag], dim=-1)
    elif target == "ZF_im":
        true_val = ZF_im_fixed
        def wrapper(theta):
            ZF = torch.complex(ZF_re_fixed, theta)
            ZL = torch.complex(ZL_re_fixed, ZL_im_fixed)
            H = fm.compute_H_complex(L1_fixed.unsqueeze(0), ZF.unsqueeze(0), ZL.unsqueeze(0))[0]
            return torch.stack([H.real, H.imag], dim=-1)
    elif target == "ZL_re":
        true_val = ZL_re_fixed
        def wrapper(theta):
            ZF = torch.complex(ZF_re_fixed, ZF_im_fixed)
            ZL = torch.complex(theta, ZL_im_fixed)
            H = fm.compute_H_complex(L1_fixed.unsqueeze(0), ZF.unsqueeze(0), ZL.unsqueeze(0))[0]
            return torch.stack([H.real, H.imag], dim=-1)
    elif target == "ZL_im":
        true_val = ZL_im_fixed
        def wrapper(theta):
            ZF = torch.complex(ZF_re_fixed, ZF_im_fixed)
            ZL = torch.complex(ZL_re_fixed, theta)
            H = fm.compute_H_complex(L1_fixed.unsqueeze(0), ZF.unsqueeze(0), ZL.unsqueeze(0))[0]
            return torch.stack([H.real, H.imag], dim=-1)
    else:
        raise ValueError(f"Unknown target: {target}. Use L1, ZF_re, ZF_im, ZL_re, or ZL_im")

    # Compute Jacobian at true value: wrapper: [] -> [F, 2], so J: [F, 2]
    J = jacfwd(wrapper)(true_val)  # [F, 2]

    # Reconstruct complex Jacobian: ∂H/∂θ = ∂H_re/∂θ + j * ∂H_im/∂θ
    dH = J[:, 0] + 1j * J[:, 1]  # [F] complex

    # FI = (2/σ²) Σ_f |∂H/∂θ|²
    if var_f.ndim == 0:
        var = var_f
    else:
        var = var_f  # [F]

    FI = (2.0 * (dH.conj() * dH).real / var).sum()
    CRLB = 1.0 / FI

    return FI, CRLB


@torch.no_grad()
def crlb_for_target_estimate(du_aug: torch.Tensor,
                             var_NF: torch.Tensor,
                             target: str) -> torch.Tensor:
    """
    Returns per-sample CRLB [N] for the requested parameterization.

    du_aug : [N,F,5] complex with columns
             [∂u/∂ZF, ∂u/∂ZL, ∂u/∂ZF*, ∂u/∂ZL*, ∂u/∂L1]
    var_NF : [N,F] or [F] real
    target : "L1" | "ZF_re" | "ZF_im" | "ZL_re" | "ZL_im"
    """
    # Broadcast var to [N,F]
    if var_NF.ndim == 1:
        var = var_NF.unsqueeze(0).expand(du_aug.shape[0], -1)
    else:
        var = var_NF

    # L1: FI = Σ_f 2|∂u/∂L1|^2 / σ^2
    if target == "L1":
        du_L1 = du_aug[..., 4]
        FI = (2.0 * (du_L1.conj() * du_L1).real / var).sum(dim=1)
        return 1.0 / FI  # [N]

    # Choose the right Wirtinger columns based on target
    if target.startswith("ZF"):
        du, duc = du_aug[..., 0], du_aug[..., 2]  # ∂u/∂ZF, ∂u/∂ZF*
    elif target.startswith("ZL"):
        du, duc = du_aug[..., 1], du_aug[..., 3]  # ∂u/∂ZL, ∂u/∂ZL*
    else:
        raise ValueError(f"Unknown target: {target}")

    # Real-parameter gradients for (z_r, z_i)
    dzr = du + duc            # ∂u/∂z_r
    dzi = -1j*du + 1j*duc     # ∂u/∂z_i

    w   = 2.0 / var
    I11 = (w * (dzr.conj()*dzr).real).sum(dim=1)  # [N]
    I22 = (w * (dzi.conj()*dzi).real).sum(dim=1)  # [N]

    if target.endswith("_re"):
        return 1.0 / I11
    elif target.endswith("_im"):
        return 1.0 / I22
    else:
        raise ValueError(f"Unknown target: {target}")


# ============================================================================
# MTL Network CRLB Functions (for 100+ parameter network)
# ============================================================================

def compute_real_FIM_mtl(var_f, scenario, wrapper_fn, get_true_param_flat_fn):
    """
    Compute Real Fisher Information Matrix for normalized theta in [0, 1].

    g = H(θ): ℝᴾ → ℂᶠ  (or ℝ²ᶠ treating real/imag separately)
    p = number of parameters (inputs)
    f = number of frequencies (outputs)

    Args:
        var_f: Noise variance (determined by SNR) [] if white noise (constant)
            or [F] if frequency dependent
        scenario: "no_fault" or "with_fault" (for logging/reference only)
        wrapper_fn: Forward model wrapper function (H_nofault_wrapper or H_fault_wrapper)
        get_true_param_flat_fn: Function that returns flat tensor of true params

    Returns:
        I: FIM [p, p] in param_order_list order
    """
    params_flat = get_true_param_flat_fn()

    # Compute Jacobian dH/dtheta
    J = jacfwd(wrapper_fn)(params_flat)  # [F, 2, P]
    Delta = J[:, 0, :] + 1j * J[:, 1, :]  # ∂g/∂θ [F, P] complex
    Delta_tilde = J[:, 0, :] - 1j * J[:, 1, :]  # ∂g*/∂θ = (∂g/∂θ)^* for real θ [F, P] complex

    Delta = Delta.unsqueeze(-1)         # [F, P, 1]
    Delta_tilde = Delta_tilde.unsqueeze(-1)  # [F, P, 1]
    # Δ_f ⊗ Δ̃_f^T + Δ̃_f ⊗ Δ_f^T = 2*Re(Δ⊗Δᴴ) which is real
    I_f = (Delta @ Delta_tilde.transpose(-1, -2)) + (Delta_tilde @ Delta.transpose(-1, -2))  # [F, P, P]
    # FIM should be real - take real part (imag should be ~0 due to numerics)
    I = ((1 / var_f) * I_f.sum(dim=0)).real  # [P, P] real and should be symmetric + PSD
    return I


def compute_real_CRLB(var_f, sorted_keys, sensitivities, scenario,
                      wrapper_fn, get_true_param_flat_fn, get_inferred_param_order_fn):
    """
    Compute real FIM and CRLB for P inferred real parameters. We use pseudoinverse so works when FIM is singular. 

    g = H(θ): ℝᴾ → ℂᶠ  (or ℝ²ᶠ treating real/imag separately)
    P = number of parameters (inputs)
    F = number of frequencies (outputs)

    Note FIM and CRLB are normalized here.
    Uses float64 precision for numerical stability.

    Args:
        var_f: Noise variance
        sorted_keys: Sorted parameter keys
        sensitivities: Sensitivity values corresponding to sorted_keys
        scenario: "no_fault" or "with_fault"
        wrapper_fn: Forward model wrapper function (H_nofault_wrapper or H_fault_wrapper)
        get_true_param_flat_fn: Function that returns flat tensor of true params
        get_inferred_param_order_fn: Function that returns (param_order_list, num_params)

    Returns:
        crlb_dict: Dict mapping parameter keys to Cramér-Rao Lower Bounds
    """
    import math

    param_order_list, _ = get_inferred_param_order_fn()
    I = compute_real_FIM_mtl(var_f, scenario, wrapper_fn, get_true_param_flat_fn)  # [p, p] Normalized FIM in [0, 1] space
    eigvals, _ = torch.linalg.eigh(I)
    print("Eigvals of FIM (descending)", torch.sort(eigvals, descending=True).values)
    eigvals = torch.sort(eigvals, descending=True).values
    lambda_max = eigvals[0]
    lambda_min = eigvals[-1]
    condition_number = lambda_max / lambda_min
    print("condition number", condition_number)

    J_pinv_torch = torch.linalg.pinv(I)
    CRLB_U1U1T = torch.diag(J_pinv_torch)

    # Build mapping from param_order_list index to sorted_keys key
    def param_order_to_key(entry):
        """Convert param_order_list entry to sorted_keys format."""
        param_type, name1, name2 = entry
        if param_type == "cable":
            return name1  # e.g., "l_w_4"
        elif param_type == "load":
            return f"{name1}.{name2}"  # e.g., "load_1.C_s"
        elif param_type == "fault_param":
            return name1

    # Create mapping: key -> index in param_order_list.
    key_to_idx = {param_order_to_key(param_order_list[i]): i for i in range(len(param_order_list))}

    print("=" * 220)
    print(f"{'Idx':<5} {'Parameter':<22} {'Sens':<10} {'CRLB U1U1T':<14} {'Unc U1U1T':<10}")
    print("-" * 220)

    # Build dicts sorted by sorted_keys order
    crlb_u1u1t_dict = {}  # key -> CRLB value
    # Print in sorted_keys order
    for index, key in enumerate(sorted_keys):
        if key not in key_to_idx:
            continue
        i = key_to_idx[key]
        sens = sensitivities[index]
        crlb_u1u1t = CRLB_U1U1T[i].item()
        crlb_u1u1t_dict[key] = crlb_u1u1t
        uncert_u1u1t_pct = math.sqrt(crlb_u1u1t) * 100

        print(f"{i:<5} {key:<22} {sens:<10} {crlb_u1u1t:<14.2e} {uncert_u1u1t_pct:>5.2f}%")

    print("=" * 220)
    return crlb_u1u1t_dict


# ============================================================================
# Bayesian CRLB Functions (for MTL network)
# ============================================================================

def beta_prior_fim_closed_form(alpha, p):
    """
    Closed form prior FIM for Beta(α,α) priors.

    For Beta(α,α) prior, the Fisher information for each parameter is:
    J_π = 4 * (2α - 1) * (α - 1) / (α - 2)

    Args:
        alpha: Beta distribution hyperparameter (must be > 2)
        p: Number of parameters

    Returns:
        J_pi: [p, p] diagonal prior FIM
    """
    if alpha <= 2:
        raise ValueError("alpha must be > 2 for finite FIM")

    j_pi_scalar = 4 * (2 * alpha - 1) * (alpha - 1) / (alpha - 2)
    J_pi = j_pi_scalar * torch.eye(p)  # Diagonal
    return J_pi


def key_to_tuple(key, network_params):
    """
    Convert parameter key string to tuple format used in param_order_list.

    Args:
        key: Parameter key string, e.g.:
            - 'l_w_4' (cable parameter)
            - 'load_1.C_m_leak' (load parameter)
            - 'fault_position' (fault parameter)
        network_params: Network parameters dict (to check fault_parameters)

    Returns:
        tuple: Parameter tuple in format:
            - ('cable', 'l_w_4', None)
            - ('load', 'load_1', 'C_m_leak')
            - ('fault_param', 'fault_position', None)
    """
    if '.' in key:
        # Load parameter: 'load_1.C_m_leak' → ('load', 'load_1', 'C_m_leak')
        parts = key.split('.')
        return ('load', parts[0], parts[1])
    elif key in network_params.get("fault_parameters", {}):
        # Fault parameter: 'fault_position' → ('fault_param', 'fault_position', None)
        return ('fault_param', key, None)
    else:
        # Cable parameter: 'l_w_4' → ('cable', 'l_w_4', None)
        return ('cable', key, None)


def compute_expected_data_fim(snr_db, all_thetas, scenario, forward_model,
                               wrapper_fn, get_true_param_flat_fn,
                               get_inferred_param_order_fn,
                               set_network_params_from_normalized_fn,
                               build_params_from_flat_fn):
    """
    Compute E_π[I(θ)] via Monte Carlo.

    Args:
        snr_db: SNR in dB
        all_thetas: Pre-generated theta samples [M, p]
        scenario: "no_fault" or "with_fault"
        forward_model: MTLForwardModel instance
        wrapper_fn: Forward model wrapper function
        get_true_param_flat_fn: Function that returns flat tensor of true params
        get_inferred_param_order_fn: Function that returns (param_order_list, num_params)
        set_network_params_from_normalized_fn: Function to set network params from normalized values
        build_params_from_flat_fn: Function to build params from flat tensor

    Returns:
        E_I: Expected data FIM [p, p]
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    num_samples = all_thetas.shape[0]
    num_params = all_thetas.shape[1]

    # Accumulate FIMs
    E_I = torch.zeros(num_params, num_params)
    param_order_list, _ = get_inferred_param_order_fn()

    for m in range(num_samples):
        #print(f"{m+1} out of Monte Carlo {num_samples} for E_π[I(θ)] at {snr_db} dB")
        set_network_params_from_normalized_fn(all_thetas[m], param_order_list)

        # Compute H_clean for this theta
        cable_lengths, load_params, fault_params = build_params_from_flat_fn(
            get_true_param_flat_fn(), param_order_list
        )
        if scenario == "with_fault":
            H_clean = forward_model.calculate_Hnw(cable_lengths, load_params, fault_params)
        else:
            H_clean = forward_model.calculate_Hnw_nofault(cable_lengths, load_params)

        # Compute var_f for this theta
        sigpow = torch.mean(torch.abs(H_clean)**2)
        var_f = sigpow / snr_lin

        # Compute FIM at this theta
        I_phi = compute_real_FIM_mtl(var_f, scenario, wrapper_fn, get_true_param_flat_fn)

        E_I += I_phi

    E_I /= num_samples
    return E_I


def compute_real_BCRLB(snr_db, selected_keys, all_thetas, scenario, alpha,
                        forward_model, network_params,
                        wrapper_fn, get_true_param_flat_fn,
                        get_inferred_param_order_fn,
                        set_network_params_from_normalized_fn,
                        build_params_from_flat_fn):
    """
    Compute Bayesian CRLB assuming beta prior with hyperparameter alpha.

    Args:
        snr_db: SNR in dB
        selected_keys: List of parameter keys to extract (in desired order)
        all_thetas: Pre-generated theta samples [M, p] (same as used for RMSE)
        scenario: "no_fault" or "with_fault"
        alpha: Beta distribution hyperparameter
        forward_model: MTLForwardModel instance
        network_params: Network parameters dict
        wrapper_fn: Forward model wrapper function
        get_true_param_flat_fn: Function that returns flat tensor of true params
        get_inferred_param_order_fn: Function that returns (param_order_list, num_params)
        set_network_params_from_normalized_fn: Function to set network params from normalized values
        build_params_from_flat_fn: Function to build params from flat tensor

    Returns:
        bcrlb_dict: {param_name: BCRLB_value} in selected_keys order
    """
    _, p = get_inferred_param_order_fn()
    print(f"Computing BCRLB with p = {p} parameters")

    # Compute E[I(θ)] using same theta samples as RMSE
    E_I = compute_expected_data_fim(
        snr_db, all_thetas, scenario, forward_model,
        wrapper_fn, get_true_param_flat_fn,
        get_inferred_param_order_fn,
        set_network_params_from_normalized_fn,
        build_params_from_flat_fn
    )

    # Compute J_π (prior FIM)
    J_pi = beta_prior_fim_closed_form(alpha, p)

    # Bayesian FIM
    J_B = E_I + J_pi

    # Bayesian CRLB
    BCRLB = torch.linalg.inv(J_B)
    bcrlb_diag_full = torch.diag(BCRLB)  # [p] in param_order_list order
    bcrlb_dict = {}
    param_order_list, _ = get_inferred_param_order_fn()

    for key in selected_keys:
        key_tuple = key_to_tuple(key, network_params)
        if key_tuple in param_order_list:
            idx = param_order_list.index(key_tuple)
            bcrlb_dict[key] = bcrlb_diag_full[idx].item()
        else:
            print(f"Warning: {key} ({key_tuple}) not found in param_order_list")

    return bcrlb_dict


def beta_prior_score(theta, alpha):
    return (alpha - 1.0) * (
        1.0 / theta - 1.0 / (1.0 - theta)
    )

def beta_prior_Lp(theta, alpha):
    g = beta_prior_score(theta, alpha)   # [p]
    return torch.outer(g, g)             # [p,p]

def compute_real_FIM_mtl_at_theta(var_f, wrapper_fn, params_flat):
    # Jacobian dH/dtheta at this specific parameter vector
    J = jacfwd(wrapper_fn)(params_flat)   # [F, 2, p]

    Delta = J[:, 0, :] + 1j * J[:, 1, :]
    Delta_tilde = J[:, 0, :] - 1j * J[:, 1, :]

    Delta = Delta.unsqueeze(-1)
    Delta_tilde = Delta_tilde.unsqueeze(-1)

    I_f = (
        Delta @ Delta_tilde.transpose(-1, -2)
        + Delta_tilde @ Delta.transpose(-1, -2)
    )

    I = ((1.0 / var_f) * I_f.sum(dim=0)).real
    return I

def compute_ATBCRB(snr_db, selected_keys, all_thetas, alpha, network_params, wrapper_fn, get_inferred_param_order_fn):
    param_order_list, p = get_inferred_param_order_fn()
    snr_lin = 10.0 ** (snr_db / 10.0)
    print(f"Computing AT-BCRB with p = {p} parameters")

    def JDP_of_theta(params_flat):
        L_P = beta_prior_Lp(params_flat, alpha)

        H_ri = wrapper_fn(params_flat)
        H_clean = H_ri[:, 0] + 1j * H_ri[:, 1]
        sigpow = torch.mean(torch.abs(H_clean)**2)
        var_f = (sigpow / snr_lin).detach()

        J_D = compute_real_FIM_mtl_at_theta(
            var_f,
            wrapper_fn,
            params_flat
        )

        return J_D + L_P

    def W_of_theta(params_flat):
        """
        params_flat: [p], normalized parameter vector
        returns W(theta) = J_DP(theta)^(-1): [p,p]
        """

        J_DP = JDP_of_theta(params_flat)
        W = torch.linalg.inv(J_DP)
        return W
    
    def div_W(params_flat):
        JW = jacfwd(W_of_theta)(params_flat) #[p, p, p] full derivative tensor of W matrix 
        #Jw[i, j, k] = dW_{ij} / d theta_k 
        #Divergence wants terms where j = k -> torch.diag with last 2 dimensions does this 
        # Select JW[i,j,j] and sum over j
        d = torch.diagonal(
            JW,
            dim1=1,
            dim2=2
        ).sum(dim=-1)                              # [p]

        return d
    num_samples = all_thetas.shape[0]
    dtype = all_thetas.dtype
    device = all_thetas.device

    sum_W = torch.zeros((p, p), dtype=dtype, device=device)
    sum_ddT = torch.zeros_like(sum_W)
    sum_WDdT = torch.zeros_like(sum_W)
    sum_DdW = torch.zeros_like(sum_W)

    all_cond_JDP = []
    all_min_eig_JDP = []

    for m, theta_sample in enumerate(all_thetas):
        if m % 25 == 0:
            print(f"AT-BCRB theta {m+1}/{num_samples}")

        params_flat = (
            theta_sample
            .detach()
            .clone()
            .requires_grad_(True)
        )
        # Diagnostic only — once per MC sample
        with torch.no_grad():
            J_DP_diag = JDP_of_theta(params_flat)

            eigvals = torch.linalg.eigvalsh(J_DP_diag)
            cond = torch.linalg.cond(J_DP_diag)

            all_min_eig_JDP.append(eigvals.min().item())
            all_cond_JDP.append(cond.item())

        W = W_of_theta(params_flat)
        d = div_W(params_flat) 
        # Jacobian of d
        Dd = jacfwd(div_W)(params_flat)            # [p,p]
        
        with torch.no_grad():
            sum_W += W
            sum_ddT += torch.outer(d, d)
            sum_WDdT += W @ Dd.T
            sum_DdW += Dd @ W

        del W, d, Dd, params_flat


    E_W = sum_W / num_samples
    E_ddT = sum_ddT / num_samples
    E_WDdT = sum_WDdT / num_samples
    E_DdW = sum_DdW / num_samples

    
     # ------------------------------------------------------------
    # Paper's F matrix
    #
    # F = E[W]
    #     - E[d d^T]
    #     - E[W (∂d/∂theta)^T]
    #     - E[(∂d/∂theta) W]
    # ------------------------------------------------------------
    F_AT = (
        E_W
        - E_ddT
        - E_WDdT
        - E_DdW
    )
    AT_BCRB = E_W @ torch.linalg.solve(F_AT, E_W)

    cond_arr = np.asarray(all_cond_JDP)
    mineig_arr = np.asarray(all_min_eig_JDP)

    print("J_DP condition number stats:")
    print("min   =", cond_arr.min())
    print("median=", np.median(cond_arr))
    print("mean  =", cond_arr.mean())
    print("95%   =", np.percentile(cond_arr, 95))
    print("99%   =", np.percentile(cond_arr, 99))
    print("max   =", cond_arr.max())

    print("J_DP minimum eigenvalue stats:")
    print("min   =", mineig_arr.min())
    print("median=", np.median(mineig_arr))
    print("mean  =", mineig_arr.mean())

    worst = np.argsort(cond_arr)[-10:][::-1]

    print("\nWorst J_DP condition numbers:")
    for idx in worst:
        print(
            f"sample {idx}: "
            f"cond={cond_arr[idx]:.3e}, "
            f"min_eig={mineig_arr[idx]:.3e}, "
            f"theta={all_thetas[idx].tolist()}"
        )

    print("E_W =\n", E_W)
    print("F_AT =\n", F_AT)

    print("E_W:")
    print(E_W)

    print("E_ddT:")
    print(E_ddT)

    print("E_WDdT:")
    print(E_WDdT)

    print("E_DdW:")
    print(E_DdW)

    print("F_AT:")
    print(F_AT)

    print("eig(E_W):", torch.linalg.eigvalsh(E_W))
    print("eig(F_AT):", torch.linalg.eigvalsh(F_AT))
    print("eig(AT):", torch.linalg.eigvalsh(AT_BCRB))
    print("||E_W|| =", torch.linalg.norm(E_W).item())
    print("||E_ddT|| =", torch.linalg.norm(E_ddT).item())
    print("||E_WDdT|| =", torch.linalg.norm(E_WDdT).item())
    print("||E_DdW|| =", torch.linalg.norm(E_DdW).item())
    print("sqrt diag E[W]:", torch.sqrt(torch.diag(E_W)))


    print("AT-BCRB =")
    print(AT_BCRB)
    atbcrb_diag_full = torch.diag(AT_BCRB)

    atbcrb_dict = {}

    for key in selected_keys:
        key_tuple = key_to_tuple(key, network_params)

        if key_tuple in param_order_list:
            idx = param_order_list.index(key_tuple)
            atbcrb_dict[key] = atbcrb_diag_full[idx].item()
        else:
            print(
                f"Warning: {key} ({key_tuple}) "
                "not found in param_order_list"
            )

    return atbcrb_dict

def compute_ATBCRB2(snr_db, selected_keys, all_thetas, alpha, network_params, wrapper_fn, get_inferred_param_order_fn):
    param_order_list, p = get_inferred_param_order_fn()
    snr_lin = 10.0 ** (snr_db / 10.0)
    print(f"Computing AT-BCRB with p = {p} parameters")

    def JD_JDP_of_theta(params_flat):
        L_P = beta_prior_Lp(params_flat, alpha)

        H_ri = wrapper_fn(params_flat)
        H_clean = H_ri[:, 0] + 1j * H_ri[:, 1]
        sigpow = torch.mean(torch.abs(H_clean)**2)
        var_f = (sigpow / snr_lin).detach()

        J_D = compute_real_FIM_mtl_at_theta(
            var_f,
            wrapper_fn,
            params_flat
        )
        J_DP = J_D + L_P
        return J_D, J_DP 

    def W_of_theta(params_flat):
        """
        params_flat: [p], normalized parameter vector
        returns W(theta) = J_DP(theta)^(-1): [p,p]
        """

        _, J_DP = JD_JDP_of_theta(params_flat)
        W = torch.linalg.inv(J_DP)
        return W
    
    def div_W(params_flat):
        JW = jacfwd(W_of_theta)(params_flat) #[p, p, p] full derivative tensor of W matrix 
        #Jw[i, j, k] = dW_{ij} / d theta_k 
        #Divergence wants terms where j = k -> torch.diag with last 2 dimensions does this 
        # Select JW[i,j,j] and sum over j
        d = torch.diagonal(
            JW,
            dim1=1,
            dim2=2
        ).sum(dim=-1)                              # [p]

        return d
    num_samples = all_thetas.shape[0]
    dtype = all_thetas.dtype
    device = all_thetas.device

    sum_W = torch.zeros(
        (p, p),
        dtype=dtype,
        device=device
    )

    sum_F = torch.zeros_like(sum_W)

    all_cond_JDP = []
    all_min_eig_JDP = []
    F_contrib_norms = []

    for m, theta_sample in enumerate(all_thetas):
        #if m % 25 == 0:
        print(f"AT-BCRB2 theta {m+1}/{num_samples}")

        params_flat = (
            theta_sample
            .detach()
            .clone()
            .requires_grad_(True)
        )

        J_D, J_DP = JD_JDP_of_theta(params_flat)

        L_P = beta_prior_Lp(params_flat, alpha)
        with torch.no_grad():
            JD_sym = 0.5 * (J_D + J_D.T)
            LP_sym = 0.5 * (L_P + L_P.T)
            JDP_sym = 0.5 * (J_DP + J_DP.T)

            eig_JD = torch.linalg.eigvalsh(JD_sym)
            eig_LP = torch.linalg.eigvalsh(LP_sym)
            eig_JDP = torch.linalg.eigvalsh(JDP_sym)

            cond_JD = torch.linalg.cond(JD_sym)
            cond_JDP = torch.linalg.cond(JDP_sym)

            print(f"\n--- sample {m} ---")
            print("theta =", params_flat.detach().cpu().numpy())

            print("eig(J_D)  =", eig_JD.cpu().numpy())
            print("cond(J_D) =", cond_JD.item())

            print("eig(L_P)  =", eig_LP.cpu().numpy())
            print("||s||^2   =", eig_LP[-1].item())

            print("eig(J_DP) =", eig_JDP.cpu().numpy())
            print("cond(J_DP)=", cond_JDP.item())

        W = W_of_theta(params_flat)
        d = div_W(params_flat) 
        g = beta_prior_score(params_flat, alpha)   # [p]
        q = W @ g + d          # [p]
        F_sample = (
            W @ J_D @ W
            + torch.outer(q, q)
        )                               # [p,p]

        with torch.no_grad():
            print("||W||     =", torch.linalg.norm(W).item())
            print("||div W|| =", torch.linalg.norm(d).item())
            print("||F_m||   =", torch.linalg.norm(F_sample).item())
        
        with torch.no_grad():
            # Accumulate WITHOUT retaining autograd graphs
            sum_W += W.detach()
            sum_F += F_sample.detach()
        del J_D, J_DP, W, d, g, q, F_sample, params_flat

    
    E_W = sum_W / num_samples
    F_AT = sum_F / num_samples


    AT_BCRB = (
        E_W @ torch.linalg.solve(F_AT, E_W)
    )

    atbcrb_diag_full = torch.diag(AT_BCRB)
    atbcrb_dict = {}

    for key in selected_keys:

        key_tuple = key_to_tuple(
            key,
            network_params
        )

        if key_tuple in param_order_list:

            idx = param_order_list.index(
                key_tuple
            )

            atbcrb_dict[key] = (
                atbcrb_diag_full[idx].item()
            )

        else:

            print(
                f"Warning: {key} ({key_tuple}) "
                "not found in param_order_list"
            )

    return atbcrb_dict


def compute_ECRB(
    snr_db,
    selected_keys,
    all_thetas,
    network_params,
    wrapper_fn,
    get_inferred_param_order_fn
):
    """
    Compute the Expected Cramer-Rao Bound (ECRB):

        ECRB = E_theta[ J_D(theta)^(-1) ]

    where J_D(theta) is the pointwise data Fisher information matrix.

    The expectation over theta is approximated using the samples in
    all_thetas.
    """

    param_order_list, p = get_inferred_param_order_fn()

    snr_lin = 10.0 ** (snr_db / 10.0)

    num_samples = all_thetas.shape[0]
    dtype = all_thetas.dtype
    device = all_thetas.device

    print(f"Computing ECRB with p = {p} parameters")

    # ------------------------------------------------------------
    # Monte Carlo accumulator
    #
    # ECRB = E_theta[J_D(theta)^(-1)]
    # ------------------------------------------------------------
    sum_JD_inv = torch.zeros(
        (p, p),
        dtype=dtype,
        device=device
    )

    # Diagnostics
    all_cond_JD = []
    all_min_eig_JD = []

    # ------------------------------------------------------------
    # Monte Carlo expectation over theta
    # ------------------------------------------------------------
    for m, theta_sample in enumerate(all_thetas):

        if m % 25 == 0:
            print(f"ECRB theta {m+1}/{num_samples}")

        params_flat = (
            theta_sample
            .detach()
            .clone()
        )

        # --------------------------------------------------------
        # Signal power at this theta
        #
        # Noise variance is selected to give the requested SNR
        # at this particular theta.
        # --------------------------------------------------------
        H_ri = wrapper_fn(params_flat)

        H_clean = (
            H_ri[:, 0]
            + 1j * H_ri[:, 1]
        )

        sigpow = torch.mean(
            torch.abs(H_clean) ** 2
        )

        var_f = sigpow / snr_lin

        # --------------------------------------------------------
        # Pointwise data Fisher information
        #
        # J_D(theta)
        # --------------------------------------------------------
        J_D = compute_real_FIM_mtl_at_theta(
            var_f,
            wrapper_fn,
            params_flat
        )

        # Numerical symmetry cleanup for diagnostics/inversion
        J_D = 0.5 * (
            J_D + J_D.T
        )

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------
        with torch.no_grad():

            eigvals = torch.linalg.eigvalsh(J_D)
            cond = torch.linalg.cond(J_D)

            all_min_eig_JD.append(
                eigvals.min().item()
            )

            all_cond_JD.append(
                cond.item()
            )

        # --------------------------------------------------------
        # Pointwise CRLB
        #
        # CRLB(theta) = J_D(theta)^(-1)
        # --------------------------------------------------------
        J_D_inv = torch.linalg.inv(J_D)

        # --------------------------------------------------------
        # Accumulate expectation
        # --------------------------------------------------------
        with torch.no_grad():
            sum_JD_inv += J_D_inv

        del J_D
        del J_D_inv
        del params_flat

    # ------------------------------------------------------------
    # Expected CRLB
    #
    # ECRB = E_theta[J_D(theta)^(-1)]
    # ------------------------------------------------------------
    ECRB = sum_JD_inv / num_samples

    # Numerical symmetry cleanup
    ECRB = 0.5 * (
        ECRB + ECRB.T
    )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    cond_arr = np.asarray(all_cond_JD)
    mineig_arr = np.asarray(all_min_eig_JD)

    print("\nJ_D condition number stats:")
    print("min    =", cond_arr.min())
    print("median =", np.median(cond_arr))
    print("mean   =", cond_arr.mean())
    print("95%    =", np.percentile(cond_arr, 95))
    print("99%    =", np.percentile(cond_arr, 99))
    print("max    =", cond_arr.max())

    print("\nJ_D minimum eigenvalue stats:")
    print("min    =", mineig_arr.min())
    print("median =", np.median(mineig_arr))
    print("mean   =", mineig_arr.mean())

    worst = np.argsort(cond_arr)[-10:][::-1]

    print("\nWorst J_D condition numbers:")

    for idx in worst:
        print(
            f"sample {idx}: "
            f"cond={cond_arr[idx]:.3e}, "
            f"min_eig={mineig_arr[idx]:.3e}, "
            f"theta={all_thetas[idx].tolist()}"
        )

    print("\nECRB =")
    print(ECRB)

    print("\neig(ECRB) =")
    print(torch.linalg.eigvalsh(ECRB))

    print("\nsqrt diag ECRB =")
    print(torch.sqrt(torch.diag(ECRB)))

    # ------------------------------------------------------------
    # Return requested diagonal elements
    # ------------------------------------------------------------
    ecrb_diag_full = torch.diag(ECRB)

    ecrb_dict = {}

    for key in selected_keys:

        key_tuple = key_to_tuple(
            key,
            network_params
        )

        if key_tuple in param_order_list:

            idx = param_order_list.index(
                key_tuple
            )

            ecrb_dict[key] = (
                ecrb_diag_full[idx].item()
            )

        else:

            print(
                f"Warning: {key} ({key_tuple}) "
                "not found in param_order_list"
            )

    return ecrb_dict