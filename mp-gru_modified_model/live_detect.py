"""
Step 4: Live MP-GRU gesture detection.

Pipeline:
    Camera
        -> MediaPipe HandLandmarker
        -> 126-D landmark vector + confidence
        -> MP-GRU one-step inference
        -> fc1 -> ReLU -> embedding -> classifier
        -> Softmax
        -> Top-2 confidence + margin
        -> Stability voting
        -> Cooldown
        -> Display prediction

This version matches the current GestureClassifier in train_classifier.py.

Controls:
    q - quit

Usage:
    py live_detect.py

CPU test:
    py live_detect.py --device cpu

GPU test:
    py live_detect.py --device cuda
"""

import argparse
import time
from collections import deque, Counter

import cv2
import firebase_admin
import numpy as np
import torch
import torch.nn.functional as F
from firebase_admin import credentials, firestore

from train_classifier import GestureClassifier
from validate_real_landmarks import (
    build_hand_landmarker,
    landmarks_to_vector,
)

cred = credentials.Certificate(
    "voxbridge-cf7be-firebase-adminsdk-fbsvc-6cf834f5bd.json"
)

firebase_admin.initialize_app(cred)

db = firestore.client()


def send_to_firestore(sentence):
    db.collection("gestures").document("latest").set({
        "text": sentence,
        "processed": False,
        "timestamp": firestore.SERVER_TIMESTAMP,
    })

    print("✓ Sent to Firestore:", sentence)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path, device):

    print("=" * 50)
    print("Loading checkpoint:", checkpoint_path)
    print("Using device:", device)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device
    )

    label_to_idx = ckpt["label_to_idx"]

    idx_to_label = {
        v: k for k, v in label_to_idx.items()
    }

    hidden_size = ckpt.get(
        "hidden_size",
        64
    )

    embedding_size = ckpt.get(
        "embedding_size",
        32
    )

    num_classes = len(label_to_idx)

    print("Number of classes:", num_classes)
    print("Hidden size:", hidden_size)
    print("Embedding size:", embedding_size)

    model = GestureClassifier(
        num_classes=num_classes,
        hidden_size=hidden_size,
        embedding_size=embedding_size
    ).to(device)

    model.load_state_dict(
        ckpt["model_state"]
    )

    model.eval()

    print("Model loaded successfully.")

    return model, idx_to_label


# ============================================================
# DRAW MEDIAPIPE LANDMARKS
# ============================================================

def draw_landmarks(frame, result):
    """
    Draw MediaPipe hand landmarks and connections.
    Compatible with MediaPipe Tasks API.
    """

    if result is None or not hasattr(result, "hand_landmarks"):
        return

    h, w = frame.shape[:2]

    # Standard MediaPipe 21-landmark hand connections
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),

        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),

        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),

        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),

        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),

        # Palm connections
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    for hand_landmarks in result.hand_landmarks:

        points = []

        # Convert normalized coordinates to pixel coordinates
        for landmark in hand_landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            points.append((x, y))

        # Draw connections first
        for start_idx, end_idx in connections:

            cv2.line(
                frame,
                points[start_idx],
                points[end_idx],
                (255, 255, 255),
                2
            )

        # Draw landmarks
        for x, y in points:

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


# ============================================================
# MAIN LIVE DETECTION
# ============================================================

def run(
    source,
    model_path,
    checkpoint_path,
    device_name,
    conf_threshold,
    neutral_threshold,
    margin_threshold,
    stability_frames,
    cooldown_time,
    history_len
):

    import cv2
    import mediapipe as mp

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if device_name == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            device_name
        )

    print("=" * 50)

    print("MP-GRU LIVE DETECTION")

    print("=" * 50)

    print(
        "PyTorch version:",
        torch.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    print(
        "Using device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("=" * 50)

    # --------------------------------------------------------
    # LOAD PYTORCH MODEL
    # --------------------------------------------------------

    model, idx_to_label = load_model(
        checkpoint_path,
        device
    )

    # --------------------------------------------------------
    # BUILD MEDIAPIPE HAND LANDMARKER
    # --------------------------------------------------------

    landmarker, mp = build_hand_landmarker(
        model_path
    )

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    if str(source).isdigit():

        cap_source = int(source)

    else:

        cap_source = source

    print(
        "Opening camera:",
        cap_source
    )

    cap = cv2.VideoCapture(
        cap_source
    )

    # Try to force a reasonable resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    if not cap.isOpened():

        print()
        print("=" * 50)
        print("ERROR: Could not open camera")
        print("=" * 50)
        print()
        print("Try another camera index:")
        print()
        print("py live_detect.py --source 1")
        print()
        return

    print(
        "Camera opened successfully."
    )

    # --------------------------------------------------------
    # INITIALIZE MP-GRU STATE
    # --------------------------------------------------------

    state = (
        model.mpgru.cell.init_state(
            batch_size=1,
            device=device
        )
    )

    # --------------------------------------------------------
    # PREDICTION HISTORY
    # --------------------------------------------------------

    pred_history = deque(
        maxlen=history_len
    )

    last_prediction_time = 0.0

    last_label = ""
    last_conf = 0.0

    sentence_buffer = []
    last_added_word = ""
    last_hand_seen_time = time.time()
    END_OF_SIGN_DELAY = 1.0

    frame_idx = 0

    start_time = time.time()

    print()

    print(
        "Running live detection."
    )

    print(
        "Press 'q' to quit."
    )

    print()

    # ========================================================
    # LIVE LOOP
    # ========================================================

    with torch.no_grad():

        while True:

            # ------------------------------------------------
            # READ CAMERA FRAME
            # ------------------------------------------------

            ok, frame = cap.read()

            if not ok:

                print(
                    "ERROR: Failed to read frame "
                    "from camera."
                )

                break

            # ------------------------------------------------
            # FLIP CAMERA IMAGE
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )

            # ------------------------------------------------
            # CONVERT BGR -> RGB
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # MEDIAPIPE IMAGE
            # ------------------------------------------------

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb
            )

            # ------------------------------------------------
            # TIMESTAMP
            # ------------------------------------------------

            timestamp_ms = int(
                (time.time() - start_time)
                * 1000
            )

            # ------------------------------------------------
            # RUN MEDIAPIPE
            # ------------------------------------------------

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            # ------------------------------------------------
            # DRAW HAND LANDMARKS
            # ------------------------------------------------

            draw_landmarks(
                frame,
                result
            )

            hand_present = (
                result is not None
                and hasattr(result, "hand_landmarks")
                and len(result.hand_landmarks) > 0
            )

            if hand_present:
                last_hand_seen_time = time.time()

            # ------------------------------------------------
            # EXTRACT 126 LANDMARK FEATURES
            # ------------------------------------------------

            x_np, c_val = (
                landmarks_to_vector(
                    result
                )
            )

            # ------------------------------------------------
            # CONVERT TO PYTORCH
            # ------------------------------------------------

            x_t = (
                torch.from_numpy(
                    x_np
                )
                .float()
                .unsqueeze(0)
                .to(device)
            )

            c_t = torch.tensor(
                [c_val],
                dtype=torch.float32,
                device=device
            )

            # ------------------------------------------------
            # MP-GRU ONE TIMESTEP
            # ------------------------------------------------

            h_t, state, diag = (
                model.mpgru.cell(
                    x_t,
                    c_t,
                    state,
                    step=frame_idx
                )
            )

            # ------------------------------------------------
            # GET NEUTRAL BLEND VALUE
            # ------------------------------------------------

            if "nu_ema" in diag:

                nu_t = (
                    diag["nu_ema"]
                    .item()
                )

            else:

                nu_t = (
                    diag["nu"]
                    .item()
                )

            # ------------------------------------------------
            # DEFAULT DISPLAY VALUES
            # ------------------------------------------------

            display_label = "..."

            display_conf = 0.0

            top1_name = "-"

            top1_conf = 0.0

            top2_name = "-"

            top2_conf = 0.0

            margin = 0.0

            # =================================================
            # CLASSIFICATION
            # =================================================

            if nu_t < neutral_threshold:

                # ---------------------------------------------
                # CURRENT GestureClassifier ARCHITECTURE
                #
                # h_t
                #   -> fc1
                #   -> ReLU
                #   -> embedding
                #   -> classifier
                # ---------------------------------------------

                z = F.relu(model.fc1(h_t))
                embedding_raw = model.embedding(z)
                logits = model.classifier(embedding_raw)
                probs = F.softmax(logits, dim=-1).squeeze(0)

                # ---------------------------------------------
                # TOP 2 PREDICTIONS
                # ---------------------------------------------

                k = min(
                    2,
                    probs.numel()
                )

                top2 = torch.topk(
                    probs,
                    k
                )

                idx1 = (
                    top2.indices[0]
                    .item()
                )

                conf1 = (
                    top2.values[0]
                    .item()
                )

                top1_name = (
                    idx_to_label[
                        idx1
                    ]
                )

                top1_conf = conf1

                if k >= 2:

                    idx2 = (
                        top2.indices[1]
                        .item()
                    )

                    conf2 = (
                        top2.values[1]
                        .item()
                    )

                    top2_name = (
                        idx_to_label[
                            idx2
                        ]
                    )

                    top2_conf = conf2

                    margin = (
                        conf1
                        - conf2
                    )

                # ---------------------------------------------
                # CONFIDENCE + MARGIN
                # ---------------------------------------------

                if (
                    conf1
                    >= conf_threshold
                    and
                    margin
                    >= margin_threshold
                ):

                    pred_history.append(
                        idx1
                    )

                else:

                    pred_history.clear()

                # ---------------------------------------------
                # STABILITY VOTING
                # ---------------------------------------------

                if (
                    len(pred_history)
                    >= stability_frames
                ):

                    common_idx, _ = (
                        Counter(
                            pred_history
                        )
                        .most_common(1)[0]
                    )

                    candidate_label = (
                        idx_to_label[
                            common_idx
                        ]
                    )

                    # -----------------------------------------
                    # COOLDOWN
                    # -----------------------------------------

                    now = time.time()

                    if (
                        now
                        - last_prediction_time
                        > cooldown_time
                    ):

                        display_label = (
                            candidate_label
                        )

                        display_conf = conf1

                        last_label = candidate_label
                        last_conf = conf1

                        if candidate_label != last_added_word:
                            sentence_buffer.append(candidate_label)
                            last_added_word = candidate_label

                        last_prediction_time = now

                    else:

                        display_label = (
                            last_label
                        )

                        display_conf = (
                            last_conf
                        )

            else:

                # ------------------------------------------------
                # NEUTRAL / OCCLUSION
                # ------------------------------------------------

                pred_history.clear()

            # =================================================
            # DISPLAY
            # =================================================

            if display_label != "...":

                overlay_main = (
                    f"{display_label} "
                    f"({display_conf:.2f})"
                )

            elif last_label:

                overlay_main = (
                    f"... | Last: "
                    f"{last_label} "
                    f"({last_conf:.2f})"
                )

            else:

                overlay_main = "..."

            # ------------------------------------------------
            # MAIN PREDICTION
            # ------------------------------------------------

            cv2.putText(
                frame,
                overlay_main,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # MP-GRU DEBUG
            # ------------------------------------------------

            occlusion_value = (
                diag["o"]
                .item()
                if "o" in diag
                else 0.0
            )

            cv2.putText(
                frame,
                (
                    f"c_t={c_val:.2f} "
                    f"o_t={occlusion_value:.0f} "
                    f"nu_t={nu_t:.2f}"
                ),
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )

            # ------------------------------------------------
            # TOP 1
            # ------------------------------------------------

            cv2.putText(
                frame,
                (
                    f"Top1: "
                    f"{top1_name:15s} "
                    f"{top1_conf:.2f}"
                ),
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                1
            )

            # ------------------------------------------------
            # TOP 2
            # ------------------------------------------------

            cv2.putText(
                frame,
                (
                    f"Top2: "
                    f"{top2_name:15s} "
                    f"{top2_conf:.2f}"
                ),
                (10, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                1
            )

            # ------------------------------------------------
            # MARGIN
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Margin: {margin:.2f}",
                (10, 144),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                1
            )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            elapsed = (
                time.time()
                - start_time
            )

            fps = (
                frame_idx
                / elapsed
                if elapsed > 0
                else 0.0
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # ------------------------------------------------
            # DEVICE
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Device: {device}",
                (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            if (
                not hand_present
                and len(sentence_buffer) > 0
                and (time.time() - last_hand_seen_time) > END_OF_SIGN_DELAY
            ):
                final_sentence = " ".join(sentence_buffer)

                print("\n" + "=" * 40)
                print("FINAL SENTENCE")
                print(final_sentence)
                send_to_firestore(final_sentence)
                print("=" * 40)

                sentence_buffer.clear()
                last_added_word = ""

            current_sentence = " ".join(sentence_buffer)

            cv2.putText(
                frame,
                "Sentence: " + current_sentence,
                (10, 440),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # ------------------------------------------------
            # SHOW WINDOW
            # ------------------------------------------------

            cv2.imshow(
                "MP-GRU Live Gesture Detection",
                frame
            )

            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                break

            frame_idx += 1

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print()

    print(
        "Live detection stopped."
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Webcam index or "
            "video file/stream URL"
        )
    )

    parser.add_argument(
        "--model",
        default="hand_landmarker.task"
    )

    parser.add_argument(
        "--checkpoint",
        default="gesture_classifier.pt"
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda"
        ],
        help=(
            "Inference device. "
            "'auto' uses CUDA if available."
        )
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.90,
        help=(
            "Minimum softmax confidence "
            "to accept prediction."
        )
    )

    parser.add_argument(
        "--neutral-threshold",
        type=float,
        default=0.5,
        help=(
            "If nu_t exceeds this, "
            "treat as neutral."
        )
    )

    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.18,
        help=(
            "Minimum top1-top2 "
            "confidence margin."
        )
    )

    parser.add_argument(
        "--stability-frames",
        type=int,
        default=5,
        help=(
            "Number of stable predictions "
            "required."
        )
    )

    parser.add_argument(
        "--history-len",
        type=int,
        default=8,
        help=(
            "Prediction history length."
        )
    )

    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.2,
        help=(
            "Cooldown between accepted "
            "predictions."
        )
    )

    args = parser.parse_args()

    run(
        source=args.source,
        model_path=args.model,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        conf_threshold=args.conf_threshold,
        neutral_threshold=args.neutral_threshold,
        margin_threshold=args.margin_threshold,
        stability_frames=args.stability_frames,
        cooldown_time=args.cooldown,
        history_len=args.history_len
    )
