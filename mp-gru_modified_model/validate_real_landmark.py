"""
Step 2 validation: real landmarks -> MP-GRU -> hidden-state dynamics.

No classifier yet. This script only checks that the architecture behaves
sensibly on real hand motion and real occlusion (hand leaving frame, hand
covered, etc.) instead of synthetic random noise.

Pipeline:
    Camera or video file
        -> MediaPipe HandLandmarker (Tasks API)
        -> 126-D landmark vector (21 landmarks x 3 coords x 2 hands, zero-filled
           for any hand not detected) + confidence score c_t
        -> MPGRUCell (one step at a time, streaming)
        -> live-collected diagnostics: o_t, gamma_t, nu_t, p_t, ||h_t - h_neutral||

--------------------------------------------------------------------------
SETUP REQUIRED (one-time):
1. pip install mediapipe opencv-python
2. Download the hand landmarker model file:
     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   and save it next to this script (or pass --model /path/to/hand_landmarker.task).
--------------------------------------------------------------------------

Usage:
    python3 validate_real_landmarks.py --source 0                # webcam
    python3 validate_real_landmarks.py --source path/to/video.mp4 # video file
    python3 validate_real_landmarks.py --source 0 --max-frames 300

Controls (webcam mode):
    q - quit and plot results
"""

import argparse
import time
from collections import deque

import numpy as np
import torch

from mp_gru import MPGRU

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
MAX_HANDS = 2
INPUT_SIZE = NUM_LANDMARKS * COORDS_PER_LANDMARK * MAX_HANDS  # 126


def build_hand_landmarker(model_path: str):
    """Build a MediaPipe HandLandmarker (Tasks API) in VIDEO running mode."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options), mp


def landmarks_to_vector(result):
    """
    Convert a HandLandmarkerResult into:
        x_t: (126,) float32 vector, zero-filled for any missing hand
        c_t: scalar confidence in [0, 1] (max handedness score across
             detected hands, or 0.0 if nothing detected)

    Hand ordering (left slot / right slot of the 126-dim vector) is assigned
    by MediaPipe's handedness label so the same physical hand tends to land
    in the same feature slot across frames.
    """
    x = np.zeros(INPUT_SIZE, dtype=np.float32)
    conf = 0.0

    if not result.hand_landmarks:
        return x, conf

    slot_size = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
    confidences = []

    for hand_idx, hand_lms in enumerate(result.hand_landmarks):
        # handedness[hand_idx][0].category_name is "Left" or "Right"
        try:
            label = result.handedness[hand_idx][0].category_name
            score = result.handedness[hand_idx][0].score
        except (IndexError, AttributeError):
            label, score = ("Left" if hand_idx == 0 else "Right"), 1.0

        slot = 0 if label == "Left" else 1
        offset = slot * slot_size
        for i, lm in enumerate(hand_lms):
            base = offset + i * COORDS_PER_LANDMARK
            x[base + 0] = lm.x
            x[base + 1] = lm.y
            x[base + 2] = lm.z
        confidences.append(score)

    conf = max(confidences) if confidences else 0.0
    return x, conf


def run(source, model_path, max_frames=None, headless=False):
    import cv2

    landmarker, mp = build_hand_landmarker(model_path)

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    model = MPGRU(input_size=INPUT_SIZE, hidden_size=64,
                   lambda_decay=0.15, alpha=1.0, Tc=15.0)
    model.eval()
    state = model.cell.init_state(batch_size=1)
    h_neutral = model.cell.h_neutral()

    trace = {"o": [], "gamma": [], "nu": [], "dist": [], "c": []}
    frame_idx = 0
    start_time = time.time()

    print("Running. Press 'q' in the video window to stop (or wait for max_frames / EOF).")

    with torch.no_grad():
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            x_np, c_val = landmarks_to_vector(result)
            x_t = torch.from_numpy(x_np).unsqueeze(0)          # (1, 126)
            c_t = torch.tensor([c_val], dtype=torch.float32)    # (1,)

            h_t, state, diag = model.cell(x_t, c_t, state, step=frame_idx)
            dist = (h_t[0] - h_neutral).norm().item()

            trace["o"].append(diag["o"].item())
            trace["gamma"].append(diag["gamma"].item())
            trace["nu"].append(diag["nu"].item())
            trace["dist"].append(dist)
            trace["c"].append(c_val)

            if not headless:
                cv2.putText(frame, f"c_t={c_val:.2f} o_t={diag['o'].item():.0f} "
                                    f"nu_t={diag['nu'].item():.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("MP-GRU live diagnostics", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames.")
    return trace


def plot_trace(trace):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - printing raw trace instead")
        for k, v in trace.items():
            print(k, v[:20], "..." if len(v) > 20 else "")
        return

    t_axis = list(range(len(trace["o"])))
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(t_axis, trace["c"], color="tab:gray")
    axes[0].set_title("Detector confidence c_t")

    axes[1].plot(t_axis, trace["gamma"], color="tab:blue")
    axes[1].set_title("Motion Confidence gamma_t")

    axes[2].plot(t_axis, trace["nu"], color="tab:orange")
    axes[2].set_title("Neutral Blend Weight nu_t")

    axes[3].plot(t_axis, trace["dist"], color="tab:red")
    axes[3].set_title("||h_t - h_neutral||")
    axes[3].set_xlabel("frame")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("real_landmark_trace.png", dpi=150)
    print("Saved plot to real_landmark_trace.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                         help="Webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--model", default="hand_landmarker.task",
                         help="Path to the MediaPipe hand_landmarker.task model file")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true",
                         help="Don't open a display window (useful for video files / servers)")
    args = parser.parse_args()

    trace = run(args.source, args.model, max_frames=args.max_frames, headless=args.headless)
    plot_trace(trace)
