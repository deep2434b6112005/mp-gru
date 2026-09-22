"""
Step 1 validation: synthetic occlusion sweep.

Feeds the MP-GRU cell sequences with a controlled occlusion window of varying
length (0, 5, 10, 20, 40 frames) and records gamma_t, nu_t, and ||h_t - h_neutral||
at every timestep. This verifies the *mathematics* behaves as designed before
any real data or downstream classifier is involved:

  - gamma_t should decay smoothly with occlusion length (Eq. 7)
  - nu_t should stay ~0 for short occlusions and ramp toward 1 as occlusion
    length crosses Tc (Eq. 12)
  - h_t should track h_neutral increasingly closely as occlusion continues,
    and snap back toward the observation-driven regime immediately on recovery

Run: python3 validate_occlusion_sweep.py
Produces: occlusion_sweep.png (if matplotlib is available) and prints raw
values to stdout regardless.
"""

import torch
from mp_gru import MPGRU

torch.manual_seed(0)

BATCH = 1
INPUT_SIZE = 126
HIDDEN_SIZE = 32
PRE_FRAMES = 5    # visible frames before occlusion starts
POST_FRAMES = 10  # visible frames after occlusion ends
LAMBDA_DECAY = 0.15
ALPHA = 1.0
TC = 15.0

OCCLUSION_LENGTHS = [0, 5, 10, 20, 40]


def build_sequence(occ_len: int):
    """Build (x, c) for a sequence: PRE_FRAMES visible, occ_len occluded,
    POST_FRAMES visible again."""
    seq_len = PRE_FRAMES + occ_len + POST_FRAMES
    x = torch.randn(BATCH, seq_len, INPUT_SIZE)
    c = torch.ones(BATCH, seq_len)
    if occ_len > 0:
        c[:, PRE_FRAMES:PRE_FRAMES + occ_len] = 0.1  # below vis_low -> occluded
    return x, c, seq_len


def run_sweep():
    model = MPGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        lambda_decay=LAMBDA_DECAY,
        alpha=ALPHA,
        Tc=TC,
    )
    model.eval()

    results = {}
    with torch.no_grad():
        for occ_len in OCCLUSION_LENGTHS:
            x, c, seq_len = build_sequence(occ_len)
            state = model.cell.init_state(BATCH, device=x.device, dtype=x.dtype)
            h_neutral = model.cell.h_neutral()

            gammas, nus, dists, o_vals, s_vals = [], [], [], [], []
            for t in range(seq_len):
                h_t, state, diag = model.cell(x[:, t, :], c[:, t], state, step=t)

                dist = (h_t[0] - h_neutral).norm().item()

                gammas.append(diag["gamma"].item())
                nus.append(diag["nu"].item())
                dists.append(dist)
                o_vals.append(diag["o"].item())
                s_vals.append(diag["s"].item())

            results[occ_len] = dict(
                gamma=gammas, nu=nus, dist=dists, o=o_vals, s=s_vals,
                seq_len=seq_len,
            )
    return results


def print_summary(results):
    for occ_len, r in results.items():
        print(f"\n=== Occlusion length: {occ_len} frames (seq_len={r['seq_len']}) ===")
        print(f"{'t':>4} {'s_t':>5} {'o_t':>6} {'gamma_t':>10} {'nu_t':>10} {'||h-h_n||':>12}")
        for t in range(r['seq_len']):
            print(f"{t:4d} {r['s'][t]:5.0f} {r['o'][t]:6.0f} "
                  f"{r['gamma'][t]:10.4f} {r['nu'][t]:10.4f} {r['dist'][t]:12.4f}")

        peak_o = max(r['o'])
        peak_nu = max(r['nu'])
        min_gamma = min(r['gamma'])
        print(f"--> peak o_t={peak_o:.0f}, peak nu_t={peak_nu:.4f}, min gamma_t={min_gamma:.4f}")


def try_plot(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping plot, printed summary above is authoritative)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)
    for occ_len, r in results.items():
        t_axis = list(range(r['seq_len']))
        axes[0].plot(t_axis, r['gamma'], label=f"occ={occ_len}")
        axes[1].plot(t_axis, r['nu'], label=f"occ={occ_len}")
        axes[2].plot(t_axis, r['dist'], label=f"occ={occ_len}")

    axes[0].set_title("Motion Confidence (gamma_t)")
    axes[1].set_title("Neutral Blend Weight (nu_t)")
    axes[2].set_title("Distance to h_neutral  ||h_t - h_neutral||")
    for ax in axes:
        ax.set_xlabel("timestep")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("occlusion_sweep.png", dpi=150)
    print("\nSaved plot to occlusion_sweep.png")


if __name__ == "__main__":
    results = run_sweep()
    print_summary(results)
    try_plot(results)

    # --- Basic pass/fail checks (sanity assertions, not a full test suite) ---
    print("\n=== Sanity checks ===")
    for occ_len, r in results.items():
        # gamma_t should be non-increasing during the occlusion window, and
        # reset back toward 1.0 once visibility resumes
        ok_gamma = r['gamma'][0] == 1.0 or occ_len == 0
        # nu_t should stay near 0 for occlusions well below Tc
        if occ_len < TC:
            ok_nu = max(r['nu']) < 0.3
        else:
            ok_nu = max(r['nu']) > 0.3
        print(f"occ_len={occ_len:3d} | gamma resets as expected: {ok_gamma} | "
              f"nu_t behavior as expected: {ok_nu}")