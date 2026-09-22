"""
VoxBridge Production MP-GRU
============================

Drop-in compatible with the existing VoxBridge GestureClassifier checkpoint.

IMPORTANT:
    This version intentionally preserves the existing learnable parameter
    structure and tensor dimensions:

        input_size  = 126
        hidden_size = 64
        embedding   = 32 (handled by GestureClassifier)

    Therefore the existing gesture_classifier.pt can still be loaded.

Main improvements:
    1. Streaming-first state handling.
    2. No artificial cold-start suppression.
    3. Safe first-hand acquisition.
    4. Confidence-aware motion trust.
    5. Quality-aware neutral recovery.
    6. EMA neutral diagnostic.
    7. Numerical safety/clamping.
    8. Backward-compatible MPGRU sequence wrapper.

The classifier-level decision logic remains outside this module.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Dict, Tuple

import torch
import torch.nn as nn


# ============================================================
# STATE
# ============================================================

class MPGRUState(NamedTuple):
    h_prev: torch.Tensor
    h_prev2: torch.Tensor
    o_prev: torch.Tensor
    s_prev: torch.Tensor
    nu_ema_prev: torch.Tensor


# ============================================================
# CELL
# ============================================================

class MPGRUCell(nn.Module):
    """
    Production streaming MP-GRU cell.

    State:
        h_prev
        h_prev2
        o_prev
        s_prev
        nu_ema_prev

    Input:
        x_t : (B, 126)
        c_t : (B,) or (B,1)

    Optional:
        m_t : quality metrics

    IMPORTANT:
        Existing learned layers are preserved so the old checkpoint
        remains loadable.
    """

    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 64,
        quality_metric_size: int = 0,

        lambda_decay: float = 0.15,

        alpha: float = 1.0,
        Tc: float = 15.0,

        vis_high: float = 0.60,
        vis_low: float = 0.40,

        # Kept for checkpoint/API compatibility.
        # Production streaming no longer suppresses prediction simply
        # because the global timestep is small.
        cold_start_steps: int = 0,

        nu_ema_beta: float = 0.30,

        # Production quality-aware recovery.
        quality_weight: float = 0.35,
        confidence_weight: float = 0.35,

        # Prevent neutral recovery from becoming too aggressive too early.
        neutral_floor: float = 0.0,
        neutral_ceiling: float = 1.0,
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

        self.quality_weight = quality_weight
        self.confidence_weight = confidence_weight

        self.neutral_floor = neutral_floor
        self.neutral_ceiling = neutral_ceiling

        H = hidden_size
        X = input_size

        # --------------------------------------------------------
        # Eq. 1: Reset gate
        # --------------------------------------------------------

        self.W_r = nn.Linear(X, H, bias=False)
        self.U_r = nn.Linear(H, H, bias=True)
        self.ln_r = nn.LayerNorm(H)

        # --------------------------------------------------------
        # Eq. 2: Update gate
        # --------------------------------------------------------

        self.W_z = nn.Linear(X, H, bias=False)
        self.U_z = nn.Linear(H, H, bias=True)
        self.ln_z = nn.LayerNorm(H)

        # --------------------------------------------------------
        # Eq. 5: Motion velocity
        # --------------------------------------------------------

        self.ln_v = nn.LayerNorm(H)

        # --------------------------------------------------------
        # Eq. 6: Motion prediction
        # --------------------------------------------------------

        self.M_p = nn.Linear(H, H, bias=False)
        self.M_v = nn.Linear(H, H, bias=True)

        # --------------------------------------------------------
        # Eq. 8: Motion Trust Gate
        # --------------------------------------------------------

        self.U_p = nn.Linear(H, H, bias=False)
        self.V_p = nn.Linear(1, H, bias=True)

        # --------------------------------------------------------
        # Eq. 9: Candidate hidden state
        # --------------------------------------------------------

        self.W_h = nn.Linear(X, H, bias=False)
        self.U_h = nn.Linear(H, H, bias=True)

        # --------------------------------------------------------
        # Eq. 10: Quality gate
        # --------------------------------------------------------

        self.W_q = nn.Linear(X, H, bias=False)
        self.U_q = nn.Linear(H, H, bias=True)

        if quality_metric_size > 0:
            self.R_q = nn.Linear(
                quality_metric_size,
                H,
                bias=False,
            )
        else:
            self.R_q = None

        self.ln_q = nn.LayerNorm(H)

        # --------------------------------------------------------
        # Neutral state
        # --------------------------------------------------------

        self.theta = nn.Parameter(torch.zeros(H))

    # ============================================================
    # NEUTRAL
    # ============================================================

    def h_neutral(self) -> torch.Tensor:
        """
        Bounded neutral state.

        tanh guarantees:

            -1 <= h_neutral <= 1
        """

        return torch.tanh(self.theta)

    # ============================================================
    # STATE INITIALIZATION
    # ============================================================

    def init_state(
        self,
        batch_size: int,
        device=None,
        dtype=None,
    ) -> MPGRUState:

        device = device or self.theta.device
        dtype = dtype or self.theta.dtype

        neutral = self.h_neutral().to(
            device=device,
            dtype=dtype,
        )

        h0 = (
            neutral
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )

        o0 = torch.zeros(
            batch_size,
            device=device,
            dtype=dtype,
        )

        # Start as "not visible".
        s0 = torch.zeros(
            batch_size,
            device=device,
            dtype=dtype,
        )

        # Neutral state is already active initially.
        nu0 = torch.ones(
            batch_size,
            device=device,
            dtype=dtype,
        )

        return MPGRUState(
            h_prev=h0,
            h_prev2=h0.clone(),
            o_prev=o0,
            s_prev=s0,
            nu_ema_prev=nu0,
        )

    # ============================================================
    # FIRST HAND ACQUISITION
    # ============================================================

    def initialize_from_observation(
        self,
        x_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> MPGRUState:
        """
        Fast acquisition initialization.

        Once a real hand is detected, seed both previous states from
        the current observation.

        This avoids the old artificial cold-start delay.
        """

        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(-1)

        B = x_t.shape[0]

        # We cannot know the eventual hidden representation before
        # passing through the cell. Neutral initialization is therefore
        # retained as the safe base state.

        state = self.init_state(
            batch_size=B,
            device=x_t.device,
            dtype=x_t.dtype,
        )

        return state

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        x_t: torch.Tensor,
        c_t: torch.Tensor,
        state: MPGRUState,
        step: int = 0,
        m_t: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        MPGRUState,
        Dict[str, torch.Tensor],
    ]:

        h_prev = state.h_prev
        h_prev2 = state.h_prev2
        o_prev = state.o_prev
        s_prev = state.s_prev
        nu_ema_prev = state.nu_ema_prev

        # --------------------------------------------------------
        # Validate confidence shape
        # --------------------------------------------------------

        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(-1)

        c_t = torch.nan_to_num(
            c_t,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        c_t = c_t.clamp(0.0, 1.0)

        c_flat = c_t.squeeze(-1)

        # --------------------------------------------------------
        # Eq. 3: Visibility hysteresis
        # --------------------------------------------------------

        is_visible = c_flat >= self.vis_high
        is_occluded = c_flat <= self.vis_low

        s_t = torch.where(
            is_visible,
            torch.ones_like(s_prev),
            torch.where(
                is_occluded,
                torch.zeros_like(s_prev),
                s_prev,
            ),
        )

        # --------------------------------------------------------
        # Eq. 4: Occlusion counter
        # --------------------------------------------------------

        o_t = torch.where(
            s_t > 0.5,
            torch.zeros_like(o_prev),
            o_prev + 1.0,
        )

        # --------------------------------------------------------
        # Eq. 1: Reset gate
        # --------------------------------------------------------

        r_t = torch.sigmoid(
            self.ln_r(
                self.W_r(x_t)
                + self.U_r(h_prev)
            )
        )

        # --------------------------------------------------------
        # Eq. 2: Update gate
        # --------------------------------------------------------

        z_t = torch.sigmoid(
            self.ln_z(
                self.W_z(x_t)
                + self.U_z(h_prev)
            )
        )

        # --------------------------------------------------------
        # Eq. 5: Motion velocity
        # --------------------------------------------------------

        velocity = h_prev - h_prev2

        velocity = torch.nan_to_num(
            velocity,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        v_t = self.ln_v(velocity)

        # --------------------------------------------------------
        # Eq. 6: Motion prediction
        # --------------------------------------------------------

        h_hat_t = torch.tanh(
            self.M_p(h_prev)
            + self.M_v(v_t)
        )

        # --------------------------------------------------------
        # Eq. 7: Motion confidence
        # --------------------------------------------------------

        gamma_t = torch.exp(
            -self.lambda_decay * o_t
        ).unsqueeze(-1)

        gamma_t = gamma_t.clamp(0.0, 1.0)

        # --------------------------------------------------------
        # Eq. 8: Motion Trust Gate
        # --------------------------------------------------------

        mtg_raw = torch.sigmoid(
            self.U_p(h_hat_t)
            + self.V_p(c_t)
        )

        p_t = gamma_t * mtg_raw

        # No artificial startup suppression.
        #
        # The first frame can use the model immediately.
        #
        # The initial velocity is zero because h_prev == h_prev2.

        # --------------------------------------------------------
        # Eq. 9: Candidate hidden state
        # --------------------------------------------------------

        h_tilde_t = torch.tanh(
            self.W_h(x_t)
            +
            (1.0 - p_t)
            * self.U_h(r_t * h_prev)
            +
            p_t * h_hat_t
        )

        # --------------------------------------------------------
        # Eq. 10: Quality gate
        # --------------------------------------------------------

        q_pre = (
            self.W_q(x_t)
            + self.U_q(h_tilde_t)
        )

        if self.R_q is not None and m_t is not None:
            q_pre = q_pre + self.R_q(m_t)

        q_t = torch.sigmoid(
            self.ln_q(q_pre)
        )

        # --------------------------------------------------------
        # Eq. 11: GRU memory update
        # --------------------------------------------------------

        zq = z_t * q_t

        h_gru_t = (
            (1.0 - zq) * h_prev
            + zq * h_tilde_t
        )

        # --------------------------------------------------------
        # Eq. 12: Base neutral blend
        # --------------------------------------------------------

        nu_base = torch.sigmoid(
            self.alpha * (o_t - self.Tc)
        )

        # --------------------------------------------------------
        # Production quality/confidence recovery
        # --------------------------------------------------------

        #
        # confidence_loss:
        #
        #   0 = excellent confidence
        #   1 = confidence approaching zero
        #

        confidence_loss = 1.0 - c_flat

        # Quality estimate from q_t.
        #
        # q_t is H-dimensional.
        # Mean converts it into a scalar quality estimate.
        #

        quality_score = q_t.mean(dim=-1)

        quality_loss = 1.0 - quality_score

        # Confidence/quality should not overwhelm normal operation.
        #

        adaptive_recovery = (
            self.confidence_weight
            * confidence_loss
            +
            self.quality_weight
            * quality_loss
        )

        adaptive_recovery = adaptive_recovery.clamp(
            0.0,
            1.0,
        )

        # --------------------------------------------------------
        # Important:
        #
        # We don't replace the trained neutral sigmoid.
        # We add a bounded reliability contribution.
        #
        # This means long occlusion still dominates.
        # Poor observation quality can accelerate recovery.
        # --------------------------------------------------------

        nu_t_raw = (
            nu_base
            +
            (1.0 - nu_base)
            * adaptive_recovery
        )

        nu_t_raw = nu_t_raw.clamp(
            self.neutral_floor,
            self.neutral_ceiling,
        )

        # --------------------------------------------------------
        # EMA diagnostic
        # --------------------------------------------------------

        beta = float(self.nu_ema_beta)

        beta = max(
            0.0,
            min(1.0, beta),
        )

        nu_ema_t = (
            beta * nu_ema_prev
            +
            (1.0 - beta) * nu_t_raw
        )

        # --------------------------------------------------------
        # Eq. 13: Final hidden state
        # --------------------------------------------------------

        nu_t = nu_t_raw.unsqueeze(-1)

        h_neutral = (
            self.h_neutral()
            .to(device=h_gru_t.device, dtype=h_gru_t.dtype)
            .unsqueeze(0)
        )

        h_t = (
            (1.0 - nu_t) * h_gru_t
            +
            nu_t * h_neutral
        )

        # --------------------------------------------------------
        # Numerical safety
        # --------------------------------------------------------

        h_t = torch.nan_to_num(
            h_t,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        # --------------------------------------------------------
        # New state
        # --------------------------------------------------------

        new_state = MPGRUState(
            h_prev=h_t,
            h_prev2=h_prev,
            o_prev=o_t,
            s_prev=s_t,
            nu_ema_prev=nu_ema_t,
        )

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------

        diagnostics = {
            "gamma": gamma_t.detach(),
            "nu": nu_t.detach(),
            "nu_ema": nu_ema_t.detach(),

            "nu_base": nu_base.detach(),

            "quality": quality_score.detach(),
            "quality_loss": quality_loss.detach(),

            "confidence_loss": confidence_loss.detach(),

            "p": p_t.detach(),
            "o": o_t.detach(),
            "s": s_t.detach(),

            "velocity_norm": velocity.norm(
                dim=-1
            ).detach(),
        }

        return (
            h_t,
            new_state,
            diagnostics,
        )


# ============================================================
# SEQUENCE WRAPPER
# ============================================================

class MPGRU(nn.Module):

    def __init__(self, **cell_kwargs):
        super().__init__()

        self.cell = MPGRUCell(
            **cell_kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        m: Optional[torch.Tensor] = None,
    ):

        B, T, _ = x.shape

        if lengths is not None:
            if lengths.shape != (B,):
                raise ValueError(
                    f"lengths must have shape ({B},), got {tuple(lengths.shape)}"
                )
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=0, max=T)

        state = self.cell.init_state(
            batch_size=B,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []

        diag_accum = {
            "gamma": [],
            "nu": [],
            "nu_ema": [],
            "nu_base": [],
            "quality": [],
            "quality_loss": [],
            "confidence_loss": [],
            "p": [],
            "o": [],
            "s": [],
            "velocity_norm": [],
        }

        for t in range(T):

            # A padded timestep is not an occlusion: its sequence has ended.
            # Run the cell for the batched tensor, then retain every component
            # of the prior state for finished samples. This keeps one batched
            # loop while ensuring padding cannot alter recurrent state.
            active = (
                torch.ones(B, dtype=torch.bool, device=x.device)
                if lengths is None
                else t < lengths
            )

            m_t = (
                m[:, t, :]
                if m is not None
                else None
            )

            h_candidate, candidate_state, diag = self.cell(
                x[:, t, :],
                c[:, t],
                state,
                step=t,
                m_t=m_t,
            )

            active_vector = active.unsqueeze(-1)
            h_t = torch.where(active_vector, h_candidate, state.h_prev)
            state = MPGRUState(
                h_prev=h_t,
                h_prev2=torch.where(active_vector, candidate_state.h_prev2, state.h_prev2),
                o_prev=torch.where(active, candidate_state.o_prev, state.o_prev),
                s_prev=torch.where(active, candidate_state.s_prev, state.s_prev),
                nu_ema_prev=torch.where(active, candidate_state.nu_ema_prev, state.nu_ema_prev),
            )

            outputs.append(h_t)

            for key in diag_accum:
                diag_accum[key].append(
                    diag[key]
                )

        stacked = {
            "gamma": torch.stack(
                diag_accum["gamma"],
                dim=1,
            ).squeeze(-1),

            "nu": torch.stack(
                diag_accum["nu"],
                dim=1,
            ).squeeze(-1),

            "nu_ema": torch.stack(
                diag_accum["nu_ema"],
                dim=1,
            ),

            "nu_base": torch.stack(
                diag_accum["nu_base"],
                dim=1,
            ),

            "quality": torch.stack(
                diag_accum["quality"],
                dim=1,
            ),

            "quality_loss": torch.stack(
                diag_accum["quality_loss"],
                dim=1,
            ),

            "confidence_loss": torch.stack(
                diag_accum["confidence_loss"],
                dim=1,
            ),

            "p": torch.stack(
                diag_accum["p"],
                dim=1,
            ),

            "o": torch.stack(
                diag_accum["o"],
                dim=1,
            ),

            "s": torch.stack(
                diag_accum["s"],
                dim=1,
            ),

            "velocity_norm": torch.stack(
                diag_accum["velocity_norm"],
                dim=1,
            ),
        }

        return (
            torch.stack(outputs, dim=1),
            state,
            stacked,
        )


# ============================================================
# SANITY TEST
# ============================================================

def _sanity_test():

    torch.manual_seed(0)

    model = MPGRU(
        input_size=126,
        hidden_size=64,
        lambda_decay=0.15,
        alpha=1.0,
        Tc=15.0,
    )

    model.eval()

    B = 1
    T = 35

    x = torch.randn(
        B,
        T,
        126,
    )

    c = torch.ones(
        B,
        T,
    )

    # 20-frame occlusion.
    c[:, 5:25] = 0.1

    with torch.inference_mode():

        outputs, state, diag = model(
            x,
            c,
        )

    print("=" * 60)
    print("MP-GRU SANITY TEST")
    print("=" * 60)

    print(
        "outputs:",
        tuple(outputs.shape),
    )

    print(
        "final occlusion:",
        state.o_prev.tolist(),
    )

    print(
        "final visibility:",
        state.s_prev.tolist(),
    )

    print(
        "max occlusion:",
        diag["o"].max().item(),
    )

    print(
        "nu at frame 24:",
        diag["nu"][0, 24].item(),
    )

    # We expect 20 occluded frames.
    assert diag["o"][0, 24].item() == 20.0

    # Neutral blend should be strong at o=20.
    assert diag["nu"][0, 24].item() > 0.90

    # After hand returns, occlusion should reset.
    assert diag["o"][0, 25].item() == 0.0

    print()
    print("✓ Shapes OK")
    print("✓ Occlusion counter OK")
    print("✓ Neutral recovery OK")
    print("✓ Visibility recovery OK")
    print()
    print("MP-GRU SANITY TEST PASSED")


if __name__ == "__main__":
    _sanity_test()
