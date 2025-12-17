# lfbgs.py
import math
import torch
from typing import Callable, Tuple

Tensor = torch.Tensor

# ---------------------------------------------------------------------
# Loss per (n, r): negative log-likelihood (no sum over restarts)
# ---------------------------------------------------------------------
@torch.no_grad()
def nll_per_block(
    U: Tensor,          # [N, R, 5] unconstrained params
    y: Tensor,          # [N, F] complex
    nv: Tensor,         # [N, F] real (noise variance per freq)
    fm,                 # object with .compute_H_complex(L1, ZF, ZL) -> [..., F] complex
    u_to_theta: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]],  # maps u[...,5] -> (L1, ZF, ZL)
) -> Tensor:
    """
    Returns per-(n,r) NLL as [N, R].
    """
    assert U.dim() == 3 and U.size(-1) == 5, f"U must be [N,R,5], got {U.shape}"
    N, R, _ = U.shape
    _, F = y.shape

    # theta for each (n,r)
    L1, ZF, ZL = u_to_theta(U)     # each [N, R]
    H = fm.compute_H_complex(L1, ZF, ZL)  # [N, R, F] complex (or will be reshaped inside fm)

    # residuals and variance-weighted squared mag
    diff = y[:, None, :] - H                         # [N,R,F]
    ll   = -((diff.abs() ** 2) / nv[:, None, :]).sum(dim=-1)  # [N,R]
    nll  = -ll                                       # [N,R]
    return nll


# ---------------------------------------------------------------------
# Per-block L-BFGS state (histories)
# ---------------------------------------------------------------------
class LBFGSState:
    def __init__(self, B: int, d: int, M: int, device: torch.device):
        self.S    = torch.zeros(B, M, d, device=device)  # s history
        self.Y    = torch.zeros(B, M, d, device=device)  # y history
        self.rho  = torch.zeros(B, M,    device=device)  # 1/(y^T s)
        self.size = torch.zeros(B, dtype=torch.long, device=device)
        self.head = torch.zeros(B, dtype=torch.long, device=device)
        self.M = M
        self.d = d

# ---------------------------------------------------------------------
# Two-loop recursion (batched over blocks)
# ---------------------------------------------------------------------
def two_loop(S: Tensor, Y: Tensor, rho: Tensor, size: Tensor, g: Tensor) -> Tensor:
    """
    S, Y: [B, M, d], rho: [B, M], size: [B], g: [B, d]
    returns: p = -H^{-1} g  with L-BFGS approximation, shape [B, d]
    """
    B, M, d = S.shape
    q = g.clone()

    # backward pass
    alpha = torch.zeros(B, M, device=g.device)
    for j in range(M - 1, -1, -1):
        mask = (size > j)                             # [B]
        Sj   = S[:, j, :]                             # [B,d]
        Yj   = Y[:, j, :]                             # [B,d]
        rhoj = rho[:, j]                              # [B]
        dot_Sq = (Sj * q).sum(dim=-1)                 # [B]
        alpha_j = rhoj * dot_Sq                       # [B]
        alpha[:, j] = torch.where(mask, alpha_j, torch.zeros_like(alpha_j))
        q = q - (alpha[:, j].unsqueeze(-1) * Yj) * mask.unsqueeze(-1)

    # initial H0 scaling per block
    last_idx = torch.clamp(size - 1, min=0)          # [B]
    S_last = S[torch.arange(B, device=g.device), last_idx, :]  # [B,d]
    Y_last = Y[torch.arange(B, device=g.device), last_idx, :]  # [B,d]
    sy = (S_last * Y_last).sum(dim=-1)               # [B]
    yy = (Y_last * Y_last).sum(dim=-1)               # [B]
    H0 = torch.where(size > 0, sy / (yy + 1e-12), torch.ones_like(sy))
    z = H0.unsqueeze(-1) * q                         # [B,d]

    # forward pass
    for j in range(M):
        mask = (size > j)
        Sj   = S[:, j, :]
        Yj   = Y[:, j, :]
        rhoj = rho[:, j]
        beta = rhoj * (Yj * z).sum(dim=-1)           # [B]
        z = z + ((alpha[:, j] - beta).unsqueeze(-1) * Sj) * mask.unsqueeze(-1)

    p = -z
    return p

# ---------------------------------------------------------------------
# Per-block Armijo backtracking (batched)
# ---------------------------------------------------------------------
def armijo_backtracking(
    x: Tensor, f: Tensor, g: Tensor, p: Tensor,
    eval_fg, c1: float = 1e-4, tau: float = 0.5, max_ls: int = 20,
    alpha0: float = 1.0
):
    """
    x: [B,d], f: [B], g: [B,d], p: [B,d]
    eval_fg(theta) -> (f_new[B], g_new[B,d])  (no side effects)
    returns: alpha[B], f_new[B], g_new[B,d], x_new[B,d]
    """
    B, d = x.shape
    alpha = torch.full((B,), float(alpha0), device=x.device)
    gTp = (g * p).sum(dim=-1)              # [B]
    descent = gTp < 0
    # start
    x_new = x + alpha.unsqueeze(-1) * p
    f_new, g_new = eval_fg(x_new)

    for _ in range(max_ls):
        armijo_ok = f_new <= f + c1 * alpha * gTp
        done = armijo_ok & descent
        if done.all():
            break
        # shrink where not done
        alpha = torch.where(done, alpha, alpha * tau)
        x_new = x + alpha.unsqueeze(-1) * p
        f_new, g_new = eval_fg(x_new)

    return alpha, f_new, g_new, x_new

# ---------------------------------------------------------------------
# Batched, decoupled L-BFGS
# ---------------------------------------------------------------------
def batched_lbfgs(
    U0: Tensor,           # [N,R,5] initial unconstrained parameters
    y: Tensor,            # [N,F] complex
    nv: Tensor,           # [N,F] real
    fm,
    u_to_theta: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]],
    history_size: int = 12,
    iters: int = 40,
    alpha0: float = 1.0,
) -> Tensor:
    """
    Returns refined U_final: [N,R,5]
    """
    assert U0.dim() == 3 and U0.size(-1) == 5, f"U0 must be [N,R,5], got {U0.shape}"
    device = U0.device
    N, R, d = U0.shape
    B = N * R

    # pack to [B, d]
    x = U0.view(B, d).detach().clone().to(device).requires_grad_(True)

    # helper to eval per-block f and g (no coupling, no sum)
    def eval_fg(x_curr):
        x_curr = x_curr.detach().clone().requires_grad_(True)  # [B,d]
        U = x_curr.view(N, R, d) #[N,R,5]
        # build per-block nll [N,R]
        L1, ZF, ZL = u_to_theta(U)                      # each [N,R]
        H = fm.compute_H_complex(L1, ZF, ZL)            # [N,R,F]
        diff = y[:, None, :] - H                        # [N,R,F]
        nll = ((diff.abs()**2) / nv[:, None, :]).sum(dim=-1)  # [N,R]
        loss_vec = nll.view(-1)                         # [B]
        # gradient per block
        grad = torch.autograd.grad(
            outputs=loss_vec, inputs=x_curr,
            grad_outputs=torch.ones_like(loss_vec),
            retain_graph=False, create_graph=False
        )[0]                                           # [B,d]
        return loss_vec.detach(), grad.detach()

    # initial f, g
    f, g = eval_fg(x)

    # L-BFGS state
    state = LBFGSState(B=B, d=d, M=history_size, device=device)

    for _ in range(iters):
        # 1) two-loop direction
        p = two_loop(state.S, state.Y, state.rho, state.size, g)  # [B,d]
        # ensure descent (fallback to -g if needed)
        gTp = (g * p).sum(dim=-1)
        bad = gTp >= 0
        if bad.any():
            p = torch.where(bad.unsqueeze(-1), -g, p)
            # optional: reset history for bad blocks
            # (omitted for brevity)

        # 2) per-block line search
        alpha, f_new, g_new, x_new = armijo_backtracking(
            x, f, g, p, eval_fg, c1=1e-4, tau=0.5, max_ls=20, alpha0=alpha0
        )

        # 3) update histories (only valid pairs where y^T s > 0)
        s = (x_new - x).detach()        # [B,d]
        yvec = (g_new - g).detach()     # [B,d]
        sy = (s * yvec).sum(dim=-1)     # [B]
        valid = sy > 1e-12

        if valid.any():
            b_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            j_idx = state.head[valid] % state.M
            state.S[b_idx, j_idx, :] = s[b_idx]
            state.Y[b_idx, j_idx, :] = yvec[b_idx]
            state.rho[b_idx, j_idx]  = 1.0 / sy[b_idx]
            state.head[valid] = (state.head[valid] + 1) % state.M
            state.size[valid] = torch.clamp(state.size[valid] + 1, max=state.M)

        # 4) move iterate
        x, f, g = x_new.detach().requires_grad_(True), f_new, g_new

    U_final = x.view(N, R, d)
    return U_final
