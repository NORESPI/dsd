from __future__ import annotations

"""
GatedDeltaNet-X — compact production monofile.

This revision keeps the minimal core while restoring essential production
features: causal depthwise conv, QK normalization, GQA, residual D, optional
Mamba-style gate parameterization, and Kahan cache persistence.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False

try:
    from torch.amp import custom_fwd as _custom_fwd, custom_bwd as _custom_bwd

    custom_fwd = lambda f: _custom_fwd(device_type="cuda")(f)
    custom_bwd = lambda f: _custom_bwd(device_type="cuda")(f)
except Exception:  # pragma: no cover
    from torch.cuda.amp import custom_fwd, custom_bwd


def l2_norm_fn(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-token L2 normalization on the last dimension."""
    return x / (x.float().pow(2).sum(dim=-1, keepdim=True).add(eps).sqrt().to(x.dtype))


def _safe_expm1_neg(x: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(-x)


def _kahan_add(acc: torch.Tensor, comp: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = delta - comp
    t = acc + y
    c = (t - acc) - y
    return t, c


class ShortConvolution(nn.Conv1d):
    """Depthwise causal 1D convolution with streaming state."""

    def __init__(self, d: int, kernel_size: int = 4, bias: bool = False, activation: Optional[str] = "silu"):
        super().__init__(d, d, kernel_size, groups=d, padding=kernel_size - 1, bias=bias)
        self.activation = activation

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation is None:
            return x
        if self.activation in ("silu", "swish"):
            return F.silu(x)
        return getattr(F, self.activation)(x)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, l, d = x.shape
        k = self.kernel_size[0]

        if cache is not None:
            buf = cache
            w = self.weight.squeeze(1)
            outs = []
            for t in range(l):
                new_buf = torch.roll(buf, shifts=-1, dims=-1).clone()
                new_buf[:, :, -1] = x[:, t].detach()
                buf = new_buf
                y_t = (buf * w.unsqueeze(0)).sum(-1)
                if self.bias is not None:
                    y_t = y_t + self.bias
                outs.append(y_t)
            y = self._act(torch.stack(outs, dim=1))
            return y, (buf if output_final_state else None)

        xt = x.transpose(1, 2)
        y = self._conv_forward(xt, self.weight, self.bias)[..., :l].transpose(1, 2)
        y = self._act(y)
        final = None
        if output_final_state:
            final = torch.zeros(b, d, k, device=x.device, dtype=x.dtype)
            take = x[:, -k:, :].transpose(1, 2)
            final[..., -take.shape[-1] :] = take
        return y, final


@dataclass
class LayerState:
    conv_q: Optional[torch.Tensor] = None
    conv_k: Optional[torch.Tensor] = None
    conv_v: Optional[torch.Tensor] = None
    recurrent: Optional[torch.Tensor] = None
    recurrent_comp: Optional[torch.Tensor] = None

    def detach_(self) -> "LayerState":
        for n in ("conv_q", "conv_k", "conv_v", "recurrent", "recurrent_comp"):
            t = getattr(self, n)
            if t is not None:
                setattr(self, n, t.detach())
        return self


@dataclass
class ModelCache:
    states: Dict[int, LayerState] = field(default_factory=dict)

    def get(self, idx: int) -> Optional[LayerState]:
        return self.states.get(idx)

    def update(self, idx: int, state: LayerState) -> None:
        self.states[idx] = state

    def clear(self) -> None:
        self.states.clear()

    def bytes(self) -> int:
        total = 0
        for s in self.states.values():
            for t in (s.conv_q, s.conv_k, s.conv_v, s.recurrent, s.recurrent_comp):
                if t is not None:
                    total += t.numel() * t.element_size()
        return total


def _efla_forward_step(
    S: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_alpha: Optional[torch.Tensor] = None,
    comp: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Closed-form EFLA rank-1 recurrent step.

    S_t = exp(-beta k k^T) (alpha * S_{t-1}) + w k v^T
    where w = (1-exp(-eta))/eta and eta = beta ||k||^2 (headwise effective).
    """
    if beta.dim() == k.dim():
        eta = (beta * k * k).sum(-1, keepdim=True)
        k_eff = beta.sqrt() * k
    else:
        k2 = (k * k).sum(-1, keepdim=True)
        eta = beta.unsqueeze(-1) * k2
        k_eff = beta.sqrt().unsqueeze(-1) * k

    if log_alpha is not None:
        a = log_alpha.exp()
        if a.dim() == 2:
            S = S * a.unsqueeze(-1).unsqueeze(-1)
            if comp is not None:
                comp = comp * a.unsqueeze(-1).unsqueeze(-1)
        else:
            S = S * a.unsqueeze(-1)
            if comp is not None:
                comp = comp * a.unsqueeze(-1)

    safe_eta = torch.where(eta > 1e-12, eta, torch.ones_like(eta))
    w = torch.where(eta > 1e-12, _safe_expm1_neg(eta) / safe_eta, 1.0 - 0.5 * eta)
    c = torch.where(eta > 1e-12, torch.expm1(-eta) / safe_eta, -(1.0 - 0.5 * eta))

    kTS = torch.einsum("bhk,bhkv->bhv", k_eff, S)
    erase = c.unsqueeze(-1) * k_eff.unsqueeze(-1) * kTS.unsqueeze(-2)
    write = w.unsqueeze(-1) * k_eff.unsqueeze(-1) * v.unsqueeze(-2)
    delta = erase + write

    if comp is None:
        return S + delta, None
    return _kahan_add(S, comp, delta)


def _efla_inverse_step(
    S_new: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact analytical inverse of `_efla_forward_step` (without Kahan state)."""
    if beta.dim() == k.dim():
        eta = (beta * k * k).sum(-1, keepdim=True)
        k_eff = beta.sqrt() * k
    else:
        k2 = (k * k).sum(-1, keepdim=True)
        eta = beta.unsqueeze(-1) * k2
        k_eff = beta.sqrt().unsqueeze(-1) * k

    safe_eta = torch.where(eta > 1e-12, eta, torch.ones_like(eta))
    w = torch.where(eta > 1e-12, _safe_expm1_neg(eta) / safe_eta, 1.0 - 0.5 * eta)
    S_hat = S_new - w.unsqueeze(-1) * k_eff.unsqueeze(-1) * v.unsqueeze(-2)

    c_inv = torch.where(eta > 1e-12, torch.expm1(eta) / safe_eta, 1.0 + 0.5 * eta)
    kTS = torch.einsum("bhk,bhkv->bhv", k_eff, S_hat)
    S_prev = S_hat + c_inv.unsqueeze(-1) * k_eff.unsqueeze(-1) * kTS.unsqueeze(-2)

    if log_alpha is not None:
        inv = (-log_alpha).exp()
        if inv.dim() == 2:
            S_prev = S_prev * inv.unsqueeze(-1).unsqueeze(-1)
        else:
            S_prev = S_prev * inv.unsqueeze(-1)
    return S_prev


if _HAS_TRITON:
    @triton.jit
    def _fused_recurrent_efla_fwd(
        q, k, v, alpha, o, h0, ht,
        scale,
        T: tl.constexpr,
        H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
        STORE_FINAL_STATE: tl.constexpr,
    ):
        i_v, i_bh = tl.program_id(0), tl.program_id(1)
        i_n = i_bh // H
        i_h = i_bh % H
        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        bos = i_n * T
        p_q = q + (bos * H + i_h) * K + o_k
        p_k = k + (bos * H + i_h) * K + o_k
        p_v = v + (bos * H + i_h) * V + o_v
        p_a = alpha + bos * H + i_h
        p_o = o + (bos * H + i_h) * V + o_v

        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            p_h0 = h0 + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
            b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

        for _ in tl.range(0, T):
            b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32) * scale
            b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
            b_a = tl.load(p_a).to(tl.float32)

            # beta has been absorbed into k/v beforehand.
            eta = tl.sum(b_k * b_k)
            safe_eta = tl.where(eta > 1e-12, eta, 1.0)
            w = tl.where(eta > 1e-12, (1.0 - tl.exp(-eta)) / safe_eta, 1.0 - 0.5 * eta)
            c = tl.where(eta > 1e-12, (tl.exp(-eta) - 1.0) / safe_eta, -(1.0 - 0.5 * eta))

            b_h = b_h * tl.exp(b_a)
            kTS = tl.sum(b_h * b_k[:, None], axis=0)
            b_h = b_h + c * b_k[:, None] * kTS[None, :]
            b_h = b_h + w * b_k[:, None] * b_v[None, :]

            b_o = tl.sum(b_h * b_q[:, None], axis=0)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            p_q += H * K
            p_k += H * K
            p_v += H * V
            p_a += H
            p_o += H * V

        if STORE_FINAL_STATE:
            p_ht = ht + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def _triton_recurrent_efla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    b, t, h, kdim = q.shape
    vdim = v.shape[-1]
    bk = triton.next_power_of_2(kdim)
    bv = min(32, triton.next_power_of_2(vdim))
    nv = triton.cdiv(vdim, bv)
    out = torch.empty_like(v)
    ht = q.new_empty(b, h, kdim, vdim, dtype=torch.float32) if output_final_state else None
    _fused_recurrent_efla_fwd[(nv, b * h)](
        q, k, v, alpha, out, initial_state, ht, kdim ** -0.5, t,
        H=h, K=kdim, V=vdim, BK=bk, BV=bv,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        num_warps=1, num_stages=3,
    )
    return out, ht


class ReversibleRecurrence(torch.autograd.Function):
    """Reversible BPTT for the EFLA recurrence with O(1)-length activation memory."""
    @staticmethod
    @custom_fwd
    def forward(ctx, q, k, v, beta, log_alpha, init_state, scale):
        b, t, h, kdim = q.shape
        vdim = v.shape[-1]
        S = init_state if init_state is not None else torch.zeros(b, h, kdim, vdim, device=q.device, dtype=torch.float32)
        qf, kf, vf, bf, af = q.float(), k.float(), v.float(), beta.float(), log_alpha.float()

        out = torch.empty(b, t, h, vdim, device=q.device, dtype=torch.float32)
        for i in range(t):
            S, _ = _efla_forward_step(S, kf[:, i], vf[:, i], bf[:, i], af[:, i], None)
            out[:, i] = torch.einsum("bhk,bhkv->bhv", qf[:, i] * scale, S)

        ctx.save_for_backward(q, k, v, beta, log_alpha, S, init_state if init_state is not None else torch.zeros(1, device=q.device))
        ctx.scale = scale
        ctx.has_init = init_state is not None
        return out.to(v.dtype), S

    @staticmethod
    @custom_bwd
    def backward(ctx, d_out, dS_final):
        q, k, v, beta, log_alpha, S_final, _init_stub = ctx.saved_tensors
        scale = ctx.scale

        qf, kf, vf, bf, af = q.float(), k.float(), v.float(), beta.float(), log_alpha.float()
        d_out = d_out.float()
        b, t, h, kdim = q.shape
        vdim = v.shape[-1]

        dq = torch.zeros_like(qf)
        dk = torch.zeros_like(kf)
        dv = torch.zeros_like(vf)
        db = torch.zeros_like(bf)
        da = torch.zeros_like(af)

        S = S_final.clone()
        dS = torch.zeros(b, h, kdim, vdim, device=q.device, dtype=torch.float32) if dS_final is None else dS_final.float().clone()

        for i in reversed(range(t)):
            qi = qf[:, i] * scale
            ki = kf[:, i]
            vi = vf[:, i]
            bi = bf[:, i]
            ai = af[:, i]

            S_prev = _efla_inverse_step(S, ki, vi, bi, ai)

            d_out_i = d_out[:, i]
            dq[:, i] = torch.einsum("bhkv,bhv->bhk", S, d_out_i) * scale
            dS = dS + torch.einsum("bhk,bhv->bhkv", qi, d_out_i)

            inputs = (
                S_prev.detach().requires_grad_(True),
                ki.detach().requires_grad_(True),
                vi.detach().requires_grad_(True),
                bi.detach().requires_grad_(True),
                ai.detach().requires_grad_(True),
            )
            with torch.enable_grad():
                S_new, _ = _efla_forward_step(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], None)
                gS, gk, gv, gb, ga = torch.autograd.grad(S_new, inputs, grad_outputs=dS)

            dS = gS
            dk[:, i] = gk
            dv[:, i] = gv
            db[:, i] = gb
            da[:, i] = ga
            S = S_prev

        d_init = dS if ctx.has_init else None
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), da.to(log_alpha.dtype), d_init, None


class GatedDeltaNetX(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        conv_size: int = 4,
        layer_idx: int = 0,
        qk_norm: str = "l2",  # l2 | softmax | none
        channelwise_beta: bool = True,
        use_kahan_state: bool = True,
        reversible_backprop: bool = True,
        use_mamba_gate: bool = True,
        use_residual: bool = True,
        gate_fn: str = "swish",  # swish | engram
        use_decoupled_beta: bool = False,
    ):
        """Single production layer with causal conv + GQA + EFLA recurrence.

        Key features:
        - streaming cache with fixed-size recurrent state;
        - optional reversible backprop;
        - optional Mamba-style gate parameterization;
        - optional channelwise or decoupled beta projections (FG²-GDN+ style).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_kv_groups = num_heads // self.num_kv_heads

        self.layer_idx = layer_idx
        self.qk_norm = qk_norm
        self.channelwise_beta = channelwise_beta
        self.use_kahan_state = use_kahan_state
        self.reversible_backprop = reversible_backprop
        self.use_mamba_gate = use_mamba_gate
        self.use_residual = use_residual
        self.gate_fn = gate_fn
        self.use_decoupled_beta = use_decoupled_beta and channelwise_beta

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        if self.key_dim % num_heads != 0 or self.value_dim % num_heads != 0:
            raise ValueError("expanded dimensions must be divisible by num_heads")
        self.k_head = self.key_dim // num_heads
        self.v_head = self.value_dim // num_heads

        self.key_dim_kv = self.key_dim // self.num_kv_groups
        self.value_dim_kv = self.value_dim // self.num_kv_groups

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_kv, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_kv, bias=False)

        self.q_conv = ShortConvolution(self.key_dim, kernel_size=conv_size, bias=False, activation="silu")
        self.k_conv = ShortConvolution(self.key_dim_kv, kernel_size=conv_size, bias=False, activation="silu")
        self.v_conv = ShortConvolution(self.value_dim_kv, kernel_size=conv_size, bias=False, activation="silu")

        beta_out = self.key_dim if channelwise_beta else num_heads
        self.beta_proj = nn.Linear(hidden_size, beta_out, bias=True)
        self.beta_v_proj = nn.Linear(hidden_size, self.key_dim, bias=True) if self.use_decoupled_beta else None
        self.alpha_proj = nn.Linear(hidden_size, num_heads, bias=not use_mamba_gate)

        if use_mamba_gate:
            A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            dt_min, dt_max = 0.001, 0.1
            dt = torch.exp(torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt = torch.clamp(dt, min=1e-4)
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        else:
            self.register_parameter("A_log", None)
            self.register_parameter("dt_bias", None)

        self.D = nn.Parameter(torch.ones(num_heads)) if use_residual else None

        self.o_norm = nn.RMSNorm(self.v_head)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self._warned_reversible_kahan = False

    def _expand_gqa(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_kv_groups == 1:
            return k, v
        b, t, hkv, d = k.shape
        _, _, _, dv = v.shape
        k = k.unsqueeze(3).expand(b, t, hkv, self.num_kv_groups, d).reshape(b, t, self.num_heads, d)
        v = v.unsqueeze(3).expand(b, t, hkv, self.num_kv_groups, dv).reshape(b, t, self.num_heads, dv)
        return k, v

    def _norm_qk(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.qk_norm == "l2":
            return l2_norm_fn(q).to(q.dtype), l2_norm_fn(k).to(k.dtype)
        if self.qk_norm == "softmax":
            return q.softmax(dim=-1), k.softmax(dim=-1)
        return q, k

    def _apply_output_gate(self, o: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.gate_fn == "engram":
            gate = torch.sigmoid(g.abs().clamp(min=1e-6).sqrt() * g.sign())
            return self.o_norm(o) * gate
        return self.o_norm(o) * F.silu(g)

    def init_state(self, batch_size: int, device=None, dtype=None) -> LayerState:
        p = next(self.parameters())
        device = device or p.device
        dtype = dtype or p.dtype
        rec = torch.zeros(batch_size, self.num_heads, self.k_head, self.v_head, device=device, dtype=torch.float32)
        comp = torch.zeros_like(rec) if self.use_kahan_state else None
        return LayerState(
            conv_q=torch.zeros(batch_size, self.key_dim, self.q_conv.kernel_size[0], device=device, dtype=dtype),
            conv_k=torch.zeros(batch_size, self.key_dim_kv, self.k_conv.kernel_size[0], device=device, dtype=dtype),
            conv_v=torch.zeros(batch_size, self.value_dim_kv, self.v_conv.kernel_size[0], device=device, dtype=dtype),
            recurrent=rec,
            recurrent_comp=comp,
        )

    def estimate_state_bytes(self, batch_size: int = 1, dtype: torch.dtype = torch.bfloat16) -> int:
        elem = torch.tensor([], dtype=dtype).element_size()
        conv = batch_size * (self.key_dim + self.key_dim_kv + self.value_dim_kv) * self.q_conv.kernel_size[0] * elem
        rec = batch_size * self.num_heads * self.k_head * self.v_head * 4
        if self.use_kahan_state:
            rec *= 2
        return conv + rec

    def forward(self, x: torch.Tensor, *, past_key_values: Optional[ModelCache] = None, use_cache: bool = False):
        b, t, _ = x.shape
        state = None if past_key_values is None else past_key_values.get(self.layer_idx)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, conv_q = self.q_conv(q, cache=None if state is None else state.conv_q, output_final_state=use_cache)
        k, conv_k = self.k_conv(k, cache=None if state is None else state.conv_k, output_final_state=use_cache)
        v, conv_v = self.v_conv(v, cache=None if state is None else state.conv_v, output_final_state=use_cache)

        q = q.view(b, t, self.num_heads, self.k_head)
        k = k.view(b, t, self.num_kv_heads, self.k_head)
        v = v.view(b, t, self.num_kv_heads, self.v_head)
        k, v = self._expand_gqa(k, v)
        q, k = self._norm_qk(q, k)

        alpha_raw = self.alpha_proj(x).float()
        if self.use_mamba_gate:
            alpha = -self.A_log.float().exp() * F.softplus(alpha_raw + self.dt_bias)
        else:
            alpha = F.logsigmoid(alpha_raw)

        if self.channelwise_beta:
            beta = self.beta_proj(x).float().sigmoid().view(b, t, self.num_heads, self.k_head)
            beta_v = self.beta_v_proj(x).float().sigmoid().view(b, t, self.num_heads, self.k_head) if self.beta_v_proj is not None else beta
            k = k * beta.sqrt()
            if self.k_head == self.v_head:
                v = v * beta_v.sqrt()
            beta_dispatch = torch.ones(b, t, self.num_heads, device=x.device, dtype=torch.float32)
            beta_absorbed = True
        else:
            beta_dispatch = self.beta_proj(x).float().sigmoid()
            beta_absorbed = False

        init_S = None if state is None else state.recurrent
        comp = None if state is None else state.recurrent_comp
        scale = self.k_head ** -0.5

        if self.training and self.reversible_backprop and q.requires_grad:
            if self.use_kahan_state and not self._warned_reversible_kahan:
                warnings.warn("Kahan compensation is disabled in reversible training for exact graph consistency.")
                self._warned_reversible_kahan = True
            o, S_new = ReversibleRecurrence.apply(q, k, v, beta_dispatch, alpha, init_S, scale)
            comp_new = None
        else:
            S = init_S if init_S is not None else torch.zeros(b, self.num_heads, self.k_head, self.v_head, device=x.device, dtype=torch.float32)
            if comp is None and self.use_kahan_state:
                comp = torch.zeros_like(S)
            can_triton = (
                _HAS_TRITON
                and x.is_cuda
                and (not self.training)
                and (not self.use_kahan_state)
                and beta_absorbed
            )
            if can_triton:
                o, S_new = _triton_recurrent_efla(
                    q.float(), k.float(), v.float(), alpha.float(), S, output_final_state=True
                )
                o = o.to(v.dtype)
                comp_new = None
            else:
                out = []
                for i in range(t):
                    S, comp = _efla_forward_step(S, k[:, i].float(), v[:, i].float(), beta_dispatch[:, i], alpha[:, i], comp)
                    out.append(torch.einsum("bhk,bhkv->bhv", q[:, i].float() * scale, S))
                o = torch.stack(out, dim=1).to(v.dtype)
                S_new = S
                comp_new = comp

        if self.D is not None:
            o = o + self.D[None, None, :, None] * v

        g = self.g_proj(x).view(b, t, self.num_heads, self.v_head)
        o = self._apply_output_gate(o, g)
        y = self.o_proj(o.reshape(b, t, self.value_dim))

        if use_cache and past_key_values is not None:
            past_key_values.update(
                self.layer_idx,
                LayerState(
                    conv_q=conv_q,
                    conv_k=conv_k,
                    conv_v=conv_v,
                    recurrent=S_new.detach(),
                    recurrent_comp=comp_new.detach() if comp_new is not None else None,
                ),
            )

        return y, None, past_key_values


class StreamingContext:
    def __init__(self, layers: Sequence[GatedDeltaNetX], batch_size: int = 1, device=None, dtype=None):
        self.layers = list(layers)
        self.cache = ModelCache()
        for layer in self.layers:
            self.cache.update(layer.layer_idx, layer.init_state(batch_size, device=device, dtype=dtype))

    def byte_size(self) -> int:
        return self.cache.bytes()

    def clear(self) -> None:
        self.cache.clear()


def build_cache_for_layers(layers: Iterable[GatedDeltaNetX], batch_size: int) -> ModelCache:
    cache = ModelCache()
    for layer in layers:
        cache.update(layer.layer_idx, layer.init_state(batch_size))
    return cache


def _self_test() -> None:
    """Smoke + invariants:
    - forward/backward;
    - streaming vs full equivalence;
    - state bytes invariant across sequence lengths.
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfgs = [
        dict(name="baseline", channelwise_beta=False, gate_fn="swish", use_mamba_gate=False),
        dict(name="fg2", channelwise_beta=True, gate_fn="swish", use_mamba_gate=True),
        dict(name="fg2+engram+gqa", channelwise_beta=True, gate_fn="engram", num_kv_heads=2, use_decoupled_beta=True),
    ]
    for cfg in cfgs:
        name = cfg.pop("name")
        layer = GatedDeltaNetX(hidden_size=128, num_heads=4, use_residual=True, **cfg).to(device=device, dtype=dtype)
        x = torch.randn(2, 48, 128, device=device, dtype=dtype, requires_grad=True)

        y, _, _ = layer(x)
        y.float().mean().backward()
        assert y.shape == x.shape and not torch.isnan(y).any(), f"{name}: fwd/bwd invalid"

        with torch.no_grad():
            y_full, _, _ = layer(x.detach(), use_cache=False)
            cache = ModelCache()
            ys = []
            for chunk in x.detach().split(7, dim=1):
                y_chunk, _, cache = layer(chunk, use_cache=True, past_key_values=cache)
                ys.append(y_chunk)
            y_stream = torch.cat(ys, dim=1)
            rel = (y_full - y_stream).float().abs().max() / (y_full.float().abs().max() + 1e-8)
            assert rel < 1e-3, f"{name}: streaming mismatch: {float(rel):.3e}"

    layers = nn.ModuleList([
        GatedDeltaNetX(hidden_size=128, num_heads=4, num_kv_heads=2, channelwise_beta=True, use_decoupled_beta=True)
        .to(device=device, dtype=dtype)
        for _ in range(3)
    ])
    baseline = build_cache_for_layers(list(layers), batch_size=2).bytes()
    with torch.no_grad():
        for l in (8, 64, 256, 1024):
            cache = build_cache_for_layers(list(layers), batch_size=2)
            h = torch.randn(2, l, 128, device=device, dtype=dtype)
            for i in range(l):
                ht = h[:, i : i + 1]
                for layer in layers:
                    ht, _, cache = layer(ht, use_cache=True, past_key_values=cache)
            assert cache.bytes() == baseline, f"state bytes changed at L={l}"

    # Gradient parity smoke: reversible path vs non-reversible on same weights.
    torch.manual_seed(123)
    layer_ref = GatedDeltaNetX(hidden_size=64, num_heads=4, reversible_backprop=False, use_kahan_state=False).to(device=device, dtype=dtype)
    layer_rev = GatedDeltaNetX(hidden_size=64, num_heads=4, reversible_backprop=True, use_kahan_state=False).to(device=device, dtype=dtype)
    layer_rev.load_state_dict(layer_ref.state_dict())
    x0 = torch.randn(2, 16, 64, device=device, dtype=dtype)
    x1 = x0.clone().detach().requires_grad_(True)
    x2 = x0.clone().detach().requires_grad_(True)
    y1, _, _ = layer_ref(x1)
    y2, _, _ = layer_rev(x2)
    loss1 = y1.float().pow(2).mean()
    loss2 = y2.float().pow(2).mean()
    loss1.backward()
    loss2.backward()
    gx_err = (x1.grad.float() - x2.grad.float()).abs().max()
    gx_den = x1.grad.float().abs().max() + 1e-8
    assert (gx_err / gx_den).item() < 5e-2, "reversible gradient mismatch too large"

    print("self-test passed (streaming + O(1) bytes)")


if __name__ == "__main__":
    _self_test()
