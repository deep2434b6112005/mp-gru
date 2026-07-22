"""
Step 4: Live gesture detection.

Pipeline (streaming, one frame at a time):
    Camera
        -> MediaPipe HandLandmarker (VIDEO mode, single continuous stream)
        -> 126-D landmark vector + confidence c_t
        -> MPGRUCell.forward (one step, state carried across frames)
        -> classifier head (same weights as train_classifier.py)
        -> Softmax
        -> Top-2 margin check
        -> Confidence check
        -> Stability voting
        -> Cooldown
        -> Display label

Uses the neutral-blend weight nu_t to decide whether to *show* a prediction:
when nu_t is high (long occlusion / hands down / no active gesture), the
overlay shows "..." instead of a stale label, since the hidden state has
drifted toward h_neutral and a classification at that point isn't meaningful.

Setup (one-time, same as validate_real_landmarks.py):
    pip install mediapipe opencv-python
    Download hand_landmarker.task:
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

Usage:
    python3 live_detect.py --source 0 --checkpoint gesture_classifier.pt --model hand_landmarker.task

Controls:
    q - quit
"""

import argparse
import time
from collections import deque, Counter

import numpy as np
import torch
import torch.nn.functional as F

from train_classifier import GestureClassifier
from validate_real_landmarks import build_hand_landmarker, landmarks_to_vector, INPUT_SIZE


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    hidden_size = ckpt.get("hidden_size", 64)
    num_classes = len(label_to_idx)

    model = GestureClassifier(num_classes=num_classes, hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, idx_to_label


def run(source, model_path, checkpoint_path, conf_threshold, neutral_threshold,
        margin_threshold, stability_frames, cooldown_time, history_len):
    import cv2
    import mediapipe as mp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_label = load_model(checkpoint_path, device)

    landmarker, mp = build_hand_landmarker(model_path)

    # Use the actual --source arg (webcam index or a stream URL), instead of
    # a hardcoded address.
    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    state = model.mpgru.cell.init_state(batch_size=1, device=device)
    h_neutral = model.mpgru.cell.h_neutral()

    # --- (1) stability history + (5) majority voting ---
    pred_history = deque(maxlen=history_len)

    # --- (3) cooldown ---
    last_prediction_time = 0.0

    # --- (9) last detected gesture, persists through neutral gaps ---
    last_label = ""
    last_conf = 0.0

    frame_idx = 0
    start_time = time.time()

    print("Running live detection. Press 'q' to quit.")

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
            x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
            c_t = torch.tensor([c_val], dtype=torch.float32, device=device)

            # --- (4) keep MP-GRU occlusion logic (nu_t, o_t) ---
            h_t, state, diag = model.mpgru.cell(x_t, c_t, state, step=frame_idx)
            nu_t = diag["nu_ema"].item()   # was diag["nu"].item()

            display_label = "..."
            display_conf = 0.0
            top1_name, top1_conf = "-", 0.0
            top2_name, top2_conf = "-", 0.0
            margin = 0.0

            if nu_t < neutral_threshold:
                z = F.relu(model.fc1(h_t))
                logits = model.fc2(z)
                probs = F.softmax(logits, dim=-1).squeeze(0)  # (num_classes,)

                # --- (2) top-2 confidence margin ---
                top2 = torch.topk(probs, 2)
                idx1 = top2.indices[0].item()
                idx2 = top2.indices[1].item()
                conf1 = top2.values[0].item()
                conf2 = top2.values[1].item()
                margin = conf1 - conf2

                top1_name, top1_conf = idx_to_label[idx1], conf1
                top2_name, top2_conf = idx_to_label[idx2], conf2

                if conf1 >= conf_threshold and margin >= margin_threshold:
                    pred_history.append(idx1)
                else:
                    pred_history.clear()

                # --- (1) stability voting: require enough matching votes ---
                if len(pred_history) >= stability_frames:
                    common_idx, _ = Counter(pred_history).most_common(1)[0]
                    candidate_label = idx_to_label[common_idx]

                    # --- (3) cooldown before accepting a new label ---
                    now = time.time()
                    if now - last_prediction_time > cooldown_time:
                        display_label = candidate_label
                        display_conf = conf1
                        last_label = candidate_label
                        last_conf = conf1
                        last_prediction_time = now
                    else:
                        # still on cooldown: keep showing the label that just fired
                        display_label = last_label
                        display_conf = last_conf
                else:
                    display_label = "..."
            else:
                # --- (7) reset history during long occlusion / neutral state ---
                pred_history.clear()
                display_label = "..."

            # --- (9) show last detected gesture instead of a bare "..." ---
            overlay_main = f"{display_label}  ({display_conf:.2f})" if display_label != "..." \
                else (f"...  |  Last: {last_label} ({last_conf:.2f})" if last_label else "...")

            cv2.putText(frame, overlay_main,
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"c_t={c_val:.2f}  o_t={diag['o'].item():.0f}  nu_t={nu_t:.2f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # --- (6) top-2 debug overlay ---
            cv2.putText(frame, f"Top1: {top1_name:12s} {top1_conf:.2f}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cv2.putText(frame, f"Top2: {top2_name:12s} {top2_conf:.2f}",
                        (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cv2.putText(frame, f"Margin: {margin:.2f}",
                        (10, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

            # --- (8) FPS display ---
            elapsed = time.time() - start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0.0
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Live Gesture Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Webcam index or video file/stream URL")
    parser.add_argument("--model", default="hand_landmarker.task")
    parser.add_argument("--checkpoint", default="gesture_classifier.pt")
    parser.add_argument("--conf-threshold", type=float, default=0.90,
                         help="Minimum softmax confidence to accept a prediction")
    parser.add_argument("--neutral-threshold", type=float, default=0.5,
                         help="If nu_t exceeds this, treat as 'no active gesture' and show '...'")
    parser.add_argument("--margin-threshold", type=float, default=0.18,
                         help="Required gap between top-1 and top-2 confidence to accept")
    parser.add_argument("--stability-frames", type=int, default=5,
                         help="Minimum matching votes (within history window) required to accept a label")
    parser.add_argument("--history-len", type=int, default=8,
                         help="Length of the rolling prediction history used for majority voting")
    parser.add_argument("--cooldown", type=float, default=1.2,
                         help="Seconds to wait after a label fires before accepting a new one")
    args = parser.parse_args()

    run(args.source, args.model, args.checkpoint, args.conf_threshold,
        args.neutral_threshold, args.margin_threshold, args.stability_frames,
        args.cooldown, args.history_len)