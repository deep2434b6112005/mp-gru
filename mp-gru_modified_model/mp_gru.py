"""
MP-GRU: Motion-Predictive GRU for continuous sign-language translation
with occlusion handling.

Implements the finalized 13-equation specification:
  1. Reset Gate
  2. Update Gate
  3. Visibility State (hysteresis)
  4. Occlusion Counter
  5. Motion Velocity
  6. Motion Prediction
  7. Motion Confidence (decay)
  8. Motion Trust Gate (MTG)              [formerly "Visibility Gate"]
  9. Candidate Hidden State
 10. Quality Gate (extensible with optional quality-metric vector m_t)
 11. GRU Memory Update
 12. Neutral Blend Weight
 13. Final Hidden State

State carried across timesteps: (h_prev, h_prev2, o_prev, s_prev, nu_ema_prev)
  h_prev, h_prev2 : previous two hidden states, shape (batch, hidden_size)
  o_prev          : occlusion counter,          shape (batch,)
  s_prev          : visibility flag (1=visible, 0=occluded), shape (batch,)
  nu_ema_prev     : EMA-smoothed neutral-blend weight, shape (batch,)
                    (added so downstream consumers, e.g. live_detect.py, get a
                    steadier occlusion/neutral signal instead of having to
                    smooth the instantaneous nu_t themselves)

h_neutral is parameterized as tanh(theta) so it stays bounded in the same
range as every other hidden state produced by the cell.

NOTE ON SCOPE: classifier-level stability logic (top-2 margin, confidence
threshold, majority-vote stability window, cooldown) intentionally does NOT
live here. This cell only sees landmarks + detection confidence; it has no
notion of gesture classes, which only exist after GestureClassifier's head
runs on h_t. Keeping that logic in the inference script (live_detect.py)
preserves MPGRUCell as a reusable, task-agnostic sequence encoder.
"""

import torch
import torch.nn as nn
from typing import NamedTuple, Optional


class MPGRUState(NamedTuple):
    h_prev: torch.Tensor       # (batch, hidden_size)
    h_prev2: torch.Tensor      # (batch, hidden_size)
    o_prev: torch.Tensor       # (batch,)  float, occlusion frame count
    s_prev: torch.Tensor       # (batch,)  float, 1.0 = visible, 0.0 = occluded
    nu_ema_prev: torch.Tensor  # (batch,)  float, EMA-smoothed neutral-blend weight


class MPGRUCell(nn.Module):
    """
    Single-timestep Motion-Predictive GRU cell.

    Args:
        input_size: dimensionality of x_t (126 for 21 landmarks x 3 coords x 2 hands)
        hidden_size: dimensionality of the hidden state
        quality_metric_size: dimensionality of optional m_t (0 disables the term)
        lambda_decay: lambda, motion-confidence decay rate (gamma_t = exp(-lambda * o_t))
        alpha: steepness of the neutral-blend sigmoid ramp (Eq. 12)
        Tc: occlusion-count midpoint of the neutral-blend ramp (Eq. 12)
        vis_high: hysteresis upper threshold (c_t above this -> visible)
        vis_low: hysteresis lower threshold (c_t below this -> occluded)
        cold_start_steps: number of initial timesteps where p_t is forced to 0
                           (global step count, since v_t needs h_{t-2})
        nu_ema_beta: smoothing factor for the EMA of nu_t (higher = smoother /
                     slower to react). 0.0 disables smoothing (nu_ema == nu_t).
    """

    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 128,
        quality_metric_size: int = 0,
        lambda_decay: float = 0.15,
        alpha: float = 1.0,
        Tc: float = 15.0,
        vis_high: float = 0.6,
        vis_low: float = 0.4,
        cold_start_steps: int = 2,
        nu_ema_beta: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.quality_metric_size = quality_metric_size
        self.lambda_decay = lambda_decay
        self.alpha = alpha
        self.Tc = Tc
        self.vis_high = vis_high
        self.vis_low = vis_low
        self.cold_start_steps = cold_start_steps
        self.nu_ema_beta = nu_ema_beta

        H, X = hidden_size, input_size

        # --- Eq. 1: Reset Gate ---
        self.W_r = nn.Linear(X, H, bias=False)
        self.U_r = nn.Linear(H, H, bias=True)
        self.ln_r = nn.LayerNorm(H)

        # --- Eq. 2: Update Gate ---
        self.W_z = nn.Linear(X, H, bias=False)
        self.U_z = nn.Linear(H, H, bias=True)
        self.ln_z = nn.LayerNorm(H)

        # --- Eq. 5: Motion Velocity normalization ---
        self.ln_v = nn.LayerNorm(H)

        # --- Eq. 6: Motion Prediction ---
        self.M_p = nn.Linear(H, H, bias=False)
        self.M_v = nn.Linear(H, H, bias=True)

        # --- Eq. 8: Motion Trust Gate (MTG) ---
        self.U_p = nn.Linear(H, H, bias=False)
        self.V_p = nn.Linear(1, H, bias=True)  # c_t is a scalar confidence

        # --- Eq. 9: Candidate Hidden State ---
        self.W_h = nn.Linear(X, H, bias=False)
        self.U_h = nn.Linear(H, H, bias=True)

        # --- Eq. 10: Quality Gate (extensible) ---
        self.W_q = nn.Linear(X, H, bias=False)
        self.U_q = nn.Linear(H, H, bias=True)
        if quality_metric_size > 0:
            self.R_q = nn.Linear(quality_metric_size, H, bias=False)
        else:
            self.R_q = None
        self.ln_q = nn.LayerNorm(H)

        # --- Learnable bounded neutral state: h_neutral = tanh(theta) ---
        self.theta = nn.Parameter(torch.zeros(H))

    def h_neutral(self) -> torch.Tensor:
        return torch.tanh(self.theta)

    def init_state(self, batch_size: int, device=None, dtype=None) -> MPGRUState:
        """h_0 = h_{-1} = h_{-2} = h_neutral ; o_0 = 0 ; s_0 = occluded ; nu_ema_0 = 1.0."""
        device = device or self.theta.device
        dtype = dtype or self.theta.dtype
        h0 = self.h_neutral().to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1).contiguous()
        o0 = torch.zeros(batch_size, device=device, dtype=dtype)
        s0 = torch.zeros(batch_size, device=device, dtype=dtype)  # 0 = occluded
        # Start "neutral" (nu=1) since s_0 is occluded — consistent with h_0 = h_neutral.
        nu_ema0 = torch.ones(batch_size, device=device, dtype=dtype)
        return MPGRUState(h_prev=h0, h_prev2=h0.clone(), o_prev=o0, s_prev=s0, nu_ema_prev=nu_ema0)

    def forward(
        self,
        x_t: torch.Tensor,
        c_t: torch.Tensor,
        state: MPGRUState,
        step: int,
        m_t: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x_t: (batch, input_size) landmark features
            c_t: (batch,) or (batch, 1) MediaPipe detection confidence in [0, 1]
            state: MPGRUState from the previous timestep
            step: global timestep index (0-based), used for cold-start gating
            m_t: optional (batch, quality_metric_size) quality-metric vector

        Returns:
            h_t: (batch, hidden_size) new hidden state
            new_state: MPGRUState to pass into the next call
            diagnostics: dict, see below (now includes "nu_ema")
        """
        h_prev, h_prev2, o_prev, s_prev, nu_ema_prev = state
        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(-1)  # (batch, 1)

        # --- Eq. 3: Visibility State (hysteresis) ---
        c_flat = c_t.squeeze(-1)
        is_visible = c_flat > self.vis_high
        is_occluded = c_flat < self.vis_low
        s_t = torch.where(is_visible, torch.ones_like(s_prev),
                torch.where(is_occluded, torch.zeros_like(s_prev), s_prev))

        # --- Eq. 4: Occlusion Counter ---
        o_t = torch.where(s_t > 0.5, torch.zeros_like(o_prev), o_prev + 1.0)

        # --- Eq. 1: Reset Gate ---
        r_t = torch.sigmoid(self.ln_r(self.W_r(x_t) + self.U_r(h_prev)))

        # --- Eq. 2: Update Gate ---
        z_t = torch.sigmoid(self.ln_z(self.W_z(x_t) + self.U_z(h_prev)))

        # --- Eq. 5: Motion Velocity (normalized) ---
        v_t = self.ln_v(h_prev - h_prev2)

        # --- Eq. 6: Motion Prediction ---
        h_hat_t = torch.tanh(self.M_p(h_prev) + self.M_v(v_t))

        # --- Eq. 7: Motion Confidence ---
        gamma_t = torch.exp(-self.lambda_decay * o_t).unsqueeze(-1)  # (batch, 1)

        # --- Eq. 8: Motion Trust Gate (MTG) ---
        mtg_raw = torch.sigmoid(self.U_p(h_hat_t) + self.V_p(c_t))
        p_t = gamma_t * mtg_raw
        if step < self.cold_start_steps:
            p_t = torch.zeros_like(p_t)

        # --- Eq. 9: Candidate Hidden State ---
        h_tilde_t = torch.tanh(
            self.W_h(x_t)
            + (1 - p_t) * self.U_h(r_t * h_prev)
            + p_t * h_hat_t
        )

        # --- Eq. 10: Quality Gate (extensible) ---
        q_pre = self.W_q(x_t) + self.U_q(h_tilde_t)
        if self.R_q is not None and m_t is not None:
            q_pre = q_pre + self.R_q(m_t)
        q_t = torch.sigmoid(self.ln_q(q_pre))

        # --- Eq. 11: GRU Memory Update ---
        zq = z_t * q_t
        h_gru_t = (1 - zq) * h_prev + zq * h_tilde_t

        # --- Eq. 12: Neutral Blend Weight ---
        nu_t_raw = torch.sigmoid(self.alpha * (o_t - self.Tc))  # (batch,)

        # --- EMA smoothing of nu_t (new): reduces frame-to-frame jitter in the
        # occlusion/neutral signal without changing h_t's own math (h_t below
        # still uses the instantaneous nu_t_raw, so hidden-state dynamics are
        # unchanged/backward-compatible). nu_ema is an *additional* diagnostic
        # for downstream display/gating decisions, e.g. live_detect.py can use
        # nu_ema instead of instantaneous nu_t for its neutral_threshold check
        # to avoid flicker right at the threshold boundary.
        beta = self.nu_ema_beta
        nu_ema_t = beta * nu_ema_prev + (1 - beta) * nu_t_raw

        nu_t = nu_t_raw.unsqueeze(-1)  # (batch, 1), kept for h_t math below

        # --- Eq. 13: Final Hidden State ---
        h_neutral = self.h_neutral().unsqueeze(0)
        h_t = (1 - nu_t) * h_gru_t + nu_t * h_neutral

        new_state = MPGRUState(h_prev=h_t, h_prev2=h_prev, o_prev=o_t, s_prev=s_t,
                                nu_ema_prev=nu_ema_t)

        diagnostics = {
            "gamma": gamma_t.detach(),      # (batch, 1)
            "nu": nu_t.detach(),            # (batch, 1) instantaneous
            "nu_ema": nu_ema_t.detach(),    # (batch,)   smoothed — prefer this for display gating
            "p": p_t.detach(),              # (batch, hidden_size)
            "o": o_t.detach(),              # (batch,)
            "s": s_t.detach(),              # (batch,)
        }
        return h_t, new_state, diagnostics


class MPGRU(nn.Module):
    """
    Sequence-level wrapper: scans MPGRUCell over a (batch, seq_len, input_size)
    tensor of landmark features and a (batch, seq_len) tensor of confidences.
    """

    def __init__(self, **cell_kwargs):
        super().__init__()
        self.cell = MPGRUCell(**cell_kwargs)

    def forward(self, x: torch.Tensor, c: torch.Tensor, m: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, input_size)
            c: (batch, seq_len)
            m: optional (batch, seq_len, quality_metric_size)

        Returns:
            outputs: (batch, seq_len, hidden_size) — h_t at every timestep
            final_state: MPGRUState after the last timestep
            diagnostics: dict of stacked per-timestep tensors, each
                         (batch, seq_len) or (batch, seq_len, hidden_size):
                         "gamma", "nu", "nu_ema", "p", "o", "s"
        """
        batch_size, seq_len, _ = x.shape
        state = self.cell.init_state(batch_size, device=x.device, dtype=x.dtype)
        outputs = []
        diag_accum = {"gamma": [], "nu": [], "nu_ema": [], "p": [], "o": [], "s": []}
        for t in range(seq_len):
            m_t = m[:, t, :] if m is not None else None
            h_t, state, diagnostics = self.cell(x[:, t, :], c[:, t], state, step=t, m_t=m_t)
            outputs.append(h_t)
            for k in diag_accum:
                diag_accum[k].append(diagnostics[k])

        stacked_diag = {
            "gamma": torch.stack(diag_accum["gamma"], dim=1).squeeze(-1),   # (batch, seq_len)
            "nu": torch.stack(diag_accum["nu"], dim=1).squeeze(-1),         # (batch, seq_len)
            "nu_ema": torch.stack(diag_accum["nu_ema"], dim=1),             # (batch, seq_len)
            "p": torch.stack(diag_accum["p"], dim=1),                       # (batch, seq_len, hidden)
            "o": torch.stack(diag_accum["o"], dim=1),                       # (batch, seq_len)
            "s": torch.stack(diag_accum["s"], dim=1),                       # (batch, seq_len)
        }
        return torch.stack(outputs, dim=1), state, stacked_diag


if __name__ == "__main__":
    # --- Sanity check: 20-frame sequence, occlusion from frame 5 to frame 14 ---
    torch.manual_seed(0)
    batch, seq_len, input_size, hidden_size = 2, 20, 126, 32

    model = MPGRU(input_size=input_size, hidden_size=hidden_size,
                   lambda_decay=0.15, alpha=1.0, Tc=15.0)

    x = torch.randn(batch, seq_len, input_size)
    c = torch.ones(batch, seq_len)
    c[:, 5:15] = 0.1  # simulate hand disappearing for frames 5-14

    outputs, final_state, diagnostics = model(x, c)

    print("outputs shape:", outputs.shape)          # (2, 20, 32)
    print("final o_t:", final_state.o_prev)          # occlusion counter after seq
    print("final s_t:", final_state.s_prev)          # visibility state after seq
    print("diagnostics keys:", list(diagnostics.keys()))
    print("gamma_t trace (sample 0):", diagnostics["gamma"][0].tolist())
    print("nu_t trace (sample 0):    ", diagnostics["nu"][0].tolist())
    print("nu_ema trace (sample 0):  ", diagnostics["nu_ema"][0].tolist())

    # Check that hidden state norm shrinks toward h_neutral during long occlusion
    h_neutral = model.cell.h_neutral()
    dists = (outputs[0] - h_neutral).norm(dim=-1)
    print("\nframe : distance to h_neutral")
    for t in range(seq_len):
        print(f"{t:5d} : {dists[t].item():.4f}")