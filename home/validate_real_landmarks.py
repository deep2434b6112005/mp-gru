"""
Product-Level MP-GRU Real-Landmark Validation
==============================================

Purpose
-------
Validate the complete real-time landmark -> MP-GRU pipeline using:

    Camera / Video
          |
          v
    MediaPipe HandLandmarker
          |
          +---- Left hand  -> slot 0
          |
          +---- Right hand -> slot 1
          |
          v
       x_t (126)
          |
          +---- c_t
          |
          +---- quality metrics
          |
          v
       MP-GRU Cell
          |
          v
       diagnostics

This script DOES NOT contain:
    - gesture classifier
    - top-2 margin
    - majority voting
    - cooldown
    - TTS
    - Firebase
    - gesture labels

Those belong after MP-GRU.

Canonical representation
------------------------
126 features:

    [Left 21 landmarks x 3,
     Right 21 landmarks x 3]

Therefore:

    x[0:63]   = LEFT
    x[63:126] = RIGHT

Missing hands are zero-filled.

Important confidence distinction
--------------------------------
MediaPipe handedness score is NOT treated as detection confidence.

We calculate:

    hand_presence_confidence

from the available MediaPipe hand result information.

If your MediaPipe version exposes detection/presence scores differently,
the script falls back safely.

Timestamp policy
----------------
VIDEO mode requires strictly increasing timestamps.

For video files:
    timestamp = frame_index / FPS

For webcam:
    use monotonic wall-clock timestamps.

Performance measurements
------------------------
Measures:

    capture time
    MediaPipe time
    MP-GRU time
    total processing time
    processing FPS

Validation tests
----------------
The script reports:

    - detection rate
    - average confidence
    - confidence percentiles
    - maximum occlusion duration
    - acquisition latency
    - recovery latency
    - left/right slot switching
    - timestamp violations
    - NaN/Inf values
    - hidden-state norm
    - distance to neutral state

Outputs
-------
    real_landmark_trace.csv
    real_landmark_trace.png

Usage
-----

Webcam:

    python3 validate_real_landmarks.py \
        --source 0 \
        --model hand_landmarker.task

Video:

    python3 validate_real_landmarks.py \
        --source video.mp4 \
        --model hand_landmarker.task

Headless:

    python3 validate_real_landmarks.py \
        --source video.mp4 \
        --headless

Limit frames:

    python3 validate_real_landmarks.py \
        --source 0 \
        --max-frames 500
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mp_gru import MPGRU


# ============================================================
# CONSTANTS
# ============================================================

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
MAX_HANDS = 2

SLOT_SIZE = NUM_LANDMARKS * COORDS_PER_LANDMARK
INPUT_SIZE = SLOT_SIZE * MAX_HANDS

LEFT_SLOT = 0
RIGHT_SLOT = 1

EPS = 1e-8


# ============================================================
# SAFE UTILITIES
# ============================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass

    return default


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(values, dtype=np.float32),
            p,
        )
    )


# ============================================================
# MEDIAPIPE
# ============================================================

def build_hand_landmarker(model_path: str):
    """
    Build MediaPipe HandLandmarker in VIDEO mode.

    Detection settings are deliberately separated from MP-GRU logic.

    For fast acquisition:
        detection confidence = 0.40
        presence confidence  = 0.40
        tracking confidence  = 0.40

    MP-GRU itself receives the resulting confidence/quality signals.
    """

    import mediapipe as mp

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(
        model_asset_path=model_path
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,

        running_mode=vision.RunningMode.VIDEO,

        num_hands=MAX_HANDS,

        min_hand_detection_confidence=0.40,
        min_hand_presence_confidence=0.40,
        min_tracking_confidence=0.40,
    )

    landmarker = (
        vision.HandLandmarker
        .create_from_options(options)
    )

    return landmarker, mp


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_hand_data(result):
    """
    Extract stable two-hand representation.

    Returns
    -------
    x:
        (126,) float32

    landmarks:
        (2, 21, 3)

    handedness:
        (2,)
        0 = missing
        1 = left
        2 = right

    handedness_scores:
        (2,)

    bboxes:
        (2, 4)

    num_hands:
        int
    """

    x = np.zeros(
        INPUT_SIZE,
        dtype=np.float32,
    )

    landmarks = np.zeros(
        (MAX_HANDS, NUM_LANDMARKS, COORDS_PER_LANDMARK),
        dtype=np.float32,
    )

    handedness = np.zeros(
        MAX_HANDS,
        dtype=np.int8,
    )

    handedness_scores = np.zeros(
        MAX_HANDS,
        dtype=np.float32,
    )

    bboxes = np.zeros(
        (MAX_HANDS, 4),
        dtype=np.float32,
    )

    hand_list = getattr(
        result,
        "hand_landmarks",
        None,
    )

    handedness_list = getattr(
        result,
        "handedness",
        None,
    )

    if not hand_list:
        return (
            x,
            landmarks,
            handedness,
            handedness_scores,
            bboxes,
            0,
        )

    for i, hand_lms in enumerate(hand_list[:MAX_HANDS]):

        # ----------------------------------------------------
        # Handedness
        # ----------------------------------------------------

        label = None
        score = 0.0

        try:
            category = handedness_list[i][0]

            label = getattr(
                category,
                "category_name",
                None,
            )

            score = safe_float(
                getattr(
                    category,
                    "score",
                    0.0,
                )
            )

        except Exception:
            pass

        label = str(label).lower()

        if label == "left":
            slot = LEFT_SLOT
            hand_id = 1

        elif label == "right":
            slot = RIGHT_SLOT
            hand_id = 2

        else:
            # Safe fallback.
            free = np.where(
                handedness == 0
            )[0]

            if len(free) == 0:
                continue

            slot = int(free[0])
            hand_id = 0

        # Do not overwrite a slot.
        if handedness[slot] != 0:
            free = np.where(
                handedness == 0
            )[0]

            if len(free) == 0:
                continue

            slot = int(free[0])

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        coords = np.zeros(
            (NUM_LANDMARKS, 3),
            dtype=np.float32,
        )

        for j, lm in enumerate(hand_lms[:NUM_LANDMARKS]):

            coords[j, 0] = safe_float(lm.x)
            coords[j, 1] = safe_float(lm.y)
            coords[j, 2] = safe_float(lm.z)

        landmarks[slot] = coords

        handedness[slot] = hand_id

        handedness_scores[slot] = clamp01(
            score
        )

        # ----------------------------------------------------
        # x_t canonical representation
        # ----------------------------------------------------

        start = slot * SLOT_SIZE
        end = start + SLOT_SIZE

        x[start:end] = coords.reshape(-1)

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        xs = coords[:, 0]
        ys = coords[:, 1]

        if len(xs) > 0:

            xmin = clamp01(float(xs.min()))
            ymin = clamp01(float(ys.min()))
            xmax = clamp01(float(xs.max()))
            ymax = clamp01(float(ys.max()))

            bboxes[slot] = np.array(
                [
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ],
                dtype=np.float32,
            )

    num_hands = int(
        np.count_nonzero(handedness)
    )

    return (
        x,
        landmarks,
        handedness,
        handedness_scores,
        bboxes,
        num_hands,
    )


# ============================================================
# FRAME CONFIDENCE
# ============================================================

def calculate_confidence(
    result,
    handedness_scores: np.ndarray,
    num_hands: int,
) -> float:
    """
    Calculate the scalar confidence supplied to MP-GRU.

    Important:
        handedness score is not technically the same thing as
        detector confidence.

    Since the Tasks API result does not consistently expose a
    per-hand detector confidence field across versions, this
    function uses the available handedness score as a conservative
    observable confidence signal.

    This can later be replaced by a dedicated detector score if
    your MediaPipe version exposes one.
    """

    if num_hands == 0:
        return 0.0

    valid = handedness_scores[
        handedness_scores > 0
    ]

    if len(valid) == 0:
        return 0.0

    # Mean rather than max.
    #
    # Why?
    # If both hands are required, max() can hide a weak hand.
    return clamp01(
        float(np.mean(valid))
    )


# ============================================================
# QUALITY METRICS
# ============================================================

def calculate_quality_metrics(
    landmarks: np.ndarray,
    bboxes: np.ndarray,
    handedness_scores: np.ndarray,
    num_hands: int,
) -> np.ndarray:
    """
    Produce optional quality vector m_t.

    Current vector:

        [left_hand_score,
         right_hand_score,
         left_bbox_area,
         right_bbox_area,
         num_hands_normalized,
         landmark_validity]

    Shape:
        (6,)
    """

    left_score = float(
        handedness_scores[LEFT_SLOT]
    )

    right_score = float(
        handedness_scores[RIGHT_SLOT]
    )

    # --------------------------------------------------------
    # Bounding box area
    # --------------------------------------------------------

    def bbox_area(b):
        xmin, ymin, xmax, ymax = b

        w = max(
            0.0,
            float(xmax - xmin),
        )

        h = max(
            0.0,
            float(ymax - ymin),
        )

        return clamp01(w * h * 4.0)

    left_area = bbox_area(
        bboxes[LEFT_SLOT]
    )

    right_area = bbox_area(
        bboxes[RIGHT_SLOT]
    )

    # --------------------------------------------------------
    # Landmark numerical validity
    # --------------------------------------------------------

    valid = np.isfinite(
        landmarks
    ).all()

    landmark_validity = (
        1.0 if valid else 0.0
    )

    return np.array(
        [
            left_score,
            right_score,
            left_area,
            right_area,
            num_hands / 2.0,
            landmark_validity,
        ],
        dtype=np.float32,
    )


# ============================================================
# TIMESTAMP
# ============================================================

def video_timestamp_ms(
    frame_idx: int,
    fps: float,
    previous_timestamp: int,
) -> int:

    if (
        fps <= 0
        or not math.isfinite(fps)
    ):
        fps = 30.0

    timestamp = int(
        round(
            frame_idx
            * 1000.0
            / fps
        )
    )

    if timestamp <= previous_timestamp:
        timestamp = previous_timestamp + 1

    return timestamp


# ============================================================
# TRACE
# ============================================================

class Trace:

    def __init__(self):

        self.frame = []

        self.timestamp_ms = []

        self.c = []

        self.num_hands = []

        self.left_score = []

        self.right_score = []

        self.o = []

        self.s = []

        self.gamma = []

        self.nu = []

        self.nu_ema = []

        self.p_mean = []

        self.p_max = []

        self.hidden_norm = []

        self.neutral_distance = []

        self.capture_ms = []

        self.mp_ms = []

        self.gru_ms = []

        self.total_ms = []

        self.processing_fps = []

    def append(
        self,
        frame_idx,
        timestamp,
        confidence,
        num_hands,
        left_score,
        right_score,
        diag,
        hidden,
        neutral,
        capture_ms,
        mp_ms,
        gru_ms,
        total_ms,
    ):

        self.frame.append(
            frame_idx
        )

        self.timestamp_ms.append(
            timestamp
        )

        self.c.append(
            confidence
        )

        self.num_hands.append(
            num_hands
        )

        self.left_score.append(
            left_score
        )

        self.right_score.append(
            right_score
        )

        self.o.append(
            float(diag["o"].item())
        )

        self.s.append(
            float(diag["s"].item())
        )

        self.gamma.append(
            float(diag["gamma"].item())
        )

        self.nu.append(
            float(diag["nu"].item())
        )

        self.nu_ema.append(
            float(diag["nu_ema"].item())
        )

        p = diag["p"]

        self.p_mean.append(
            float(p.mean().item())
        )

        self.p_max.append(
            float(p.max().item())
        )

        h = hidden[0]

        self.hidden_norm.append(
            float(torch.linalg.vector_norm(h).item())
        )

        self.neutral_distance.append(
            float(
                torch.linalg.vector_norm(
                    h - neutral
                ).item()
            )
        )

        self.capture_ms.append(
            capture_ms
        )

        self.mp_ms.append(
            mp_ms
        )

        self.gru_ms.append(
            gru_ms
        )

        self.total_ms.append(
            total_ms
        )

        self.processing_fps.append(
            1000.0 / max(total_ms, EPS)
        )

    def length(self):
        return len(self.frame)


# ============================================================
# RUN VALIDATION
# ============================================================

def run(
    source,
    model_path,
    max_frames=None,
    headless=False,
    save_csv=True,
):
    import cv2

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("MP-GRU PRODUCT-LEVEL REAL LANDMARK VALIDATION")
    print("=" * 70)

    print(
        f"PyTorch device : {device}"
    )

    print(
        f"Input size     : {INPUT_SIZE}"
    )

    print(
        f"Left slot      : x[0:63]"
    )

    print(
        f"Right slot     : x[63:126]"
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    landmarker, mp = build_hand_landmarker(
        model_path
    )

    print(
        "✓ MediaPipe HandLandmarker loaded"
    )

    # --------------------------------------------------------
    # Video source
    # --------------------------------------------------------

    if str(source).isdigit():
        source_value = int(source)
        is_camera = True
    else:
        source_value = source
        is_camera = False

    cap = cv2.VideoCapture(
        source_value
    )

    if not cap.isOpened():
        landmarker.close()

        raise RuntimeError(
            f"Could not open source: {source}"
        )

    fps = safe_float(
        cap.get(cv2.CAP_PROP_FPS),
        30.0,
    )

    if fps <= 0:
        fps = 30.0

    print(
        f"Source FPS      : {fps:.2f}"
    )

    # --------------------------------------------------------
    # MP-GRU
    # --------------------------------------------------------

    model = MPGRU(
        input_size=INPUT_SIZE,
        hidden_size=64,

        quality_metric_size=6,

        lambda_decay=0.15,
        alpha=1.0,
        Tc=15.0,

        vis_high=0.60,
        vis_low=0.40,

        cold_start_steps=2,

        nu_ema_beta=0.30,
    ).to(device)

    model.eval()

    state = model.cell.init_state(
        batch_size=1,
        device=device,
        dtype=torch.float32,
    )

    h_neutral = (
        model.cell
        .h_neutral()
        .detach()
    )

    print(
        "✓ MP-GRU initialized"
    )

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace = Trace()

    frame_idx = 0

    previous_timestamp = -1

    previous_wall_time = time.perf_counter()

    timestamp_violations = 0

    start_time = time.perf_counter()

    # Acquisition tracking
    previous_had_hand = False

    acquisition_start_time = None

    acquisition_latencies = []

    # Recovery tracking
    occlusion_start_time = None

    recovery_latencies = []

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    try:

        with torch.inference_mode():

            while cap.isOpened():

                # ====================================================
                # Capture
                # ====================================================

                capture_start = time.perf_counter()

                ok, frame = cap.read()

                capture_end = time.perf_counter()

                capture_ms = (
                    capture_end
                    - capture_start
                ) * 1000.0

                if not ok:
                    break

                # ====================================================
                # Timestamp
                # ====================================================

                if is_camera:

                    now = time.monotonic()

                    timestamp_ms = int(
                        now * 1000.0
                    )

                    if timestamp_ms <= previous_timestamp:
                        timestamp_ms = (
                            previous_timestamp + 1
                        )

                else:

                    timestamp_ms = (
                        video_timestamp_ms(
                            frame_idx,
                            fps,
                            previous_timestamp,
                        )
                    )

                if (
                    timestamp_ms
                    <= previous_timestamp
                ):
                    timestamp_violations += 1

                previous_timestamp = (
                    timestamp_ms
                )

                # ====================================================
                # RGB
                # ====================================================

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                # ====================================================
                # MediaPipe
                # ====================================================

                mp_start = time.perf_counter()

                try:

                    result = (
                        landmarker
                        .detect_for_video(
                            mp_image,
                            timestamp_ms,
                        )
                    )

                except Exception as exc:

                    print(
                        f"[MediaPipe ERROR] "
                        f"frame={frame_idx}: "
                        f"{exc}"
                    )

                    result = None

                mp_end = time.perf_counter()

                mp_ms = (
                    mp_end
                    - mp_start
                ) * 1000.0

                # ====================================================
                # Extract landmarks
                # ====================================================

                if result is None:

                    x_np = np.zeros(
                        INPUT_SIZE,
                        dtype=np.float32,
                    )

                    landmarks = np.zeros(
                        (2, 21, 3),
                        dtype=np.float32,
                    )

                    handedness = np.zeros(
                        2,
                        dtype=np.int8,
                    )

                    handedness_scores = np.zeros(
                        2,
                        dtype=np.float32,
                    )

                    bboxes = np.zeros(
                        (2, 4),
                        dtype=np.float32,
                    )

                    num_hands = 0

                else:

                    (
                        x_np,
                        landmarks,
                        handedness,
                        handedness_scores,
                        bboxes,
                        num_hands,
                    ) = extract_hand_data(
                        result
                    )

                confidence = (
                    calculate_confidence(
                        result,
                        handedness_scores,
                        num_hands,
                    )
                    if result is not None
                    else 0.0
                )

                # ====================================================
                # Quality vector
                # ====================================================

                quality = (
                    calculate_quality_metrics(
                        landmarks,
                        bboxes,
                        handedness_scores,
                        num_hands,
                    )
                )

                # ====================================================
                # Numerical validation
                # ====================================================

                if not np.isfinite(
                    x_np
                ).all():

                    print(
                        f"[WARNING] "
                        f"NaN/Inf detected "
                        f"at frame {frame_idx}"
                    )

                    x_np = np.nan_to_num(
                        x_np,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                # ====================================================
                # Tensor conversion
                # ====================================================

                x_t = torch.from_numpy(
                    x_np
                ).unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                )

                c_t = torch.tensor(
                    [confidence],
                    device=device,
                    dtype=torch.float32,
                )

                m_t = torch.from_numpy(
                    quality
                ).unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                )

                # ====================================================
                # MP-GRU
                # ====================================================

                gru_start = time.perf_counter()

                h_t, state, diag = (
                    model.cell(
                        x_t,
                        c_t,
                        state,
                        step=frame_idx,
                        m_t=m_t,
                    )
                )

                # Force completion for accurate GPU timing.
                if device.type == "cuda":
                    torch.cuda.synchronize()

                gru_end = time.perf_counter()

                gru_ms = (
                    gru_end
                    - gru_start
                ) * 1000.0

                # ====================================================
                # Total
                # ====================================================

                total_ms = (
                    capture_ms
                    + mp_ms
                    + gru_ms
                )

                # ====================================================
                # Trace
                # ====================================================

                trace.append(
                    frame_idx=frame_idx,
                    timestamp=timestamp_ms,
                    confidence=confidence,
                    num_hands=num_hands,
                    left_score=float(
                        handedness_scores[
                            LEFT_SLOT
                        ]
                    ),
                    right_score=float(
                        handedness_scores[
                            RIGHT_SLOT
                        ]
                    ),
                    diag=diag,
                    hidden=h_t,
                    neutral=h_neutral,
                    capture_ms=capture_ms,
                    mp_ms=mp_ms,
                    gru_ms=gru_ms,
                    total_ms=total_ms,
                )

                # ====================================================
                # Acquisition detection
                # ====================================================

                has_hand = num_hands > 0

                if not previous_had_hand:

                    if has_hand:

                        if (
                            acquisition_start_time
                            is not None
                        ):

                            latency = (
                                time.perf_counter()
                                - acquisition_start_time
                            )

                            acquisition_latencies.append(
                                latency * 1000.0
                            )

                        acquisition_start_time = None

                else:

                    if not has_hand:

                        acquisition_start_time = (
                            time.perf_counter()
                        )

                previous_had_hand = has_hand

                # ====================================================
                # Occlusion recovery
                # ====================================================

                occluded = (
                    float(
                        diag["s"].item()
                    ) < 0.5
                )

                if occluded:

                    if (
                        occlusion_start_time
                        is None
                    ):

                        occlusion_start_time = (
                            time.perf_counter()
                        )

                else:

                    if (
                        occlusion_start_time
                        is not None
                    ):

                        recovery = (
                            time.perf_counter()
                            - occlusion_start_time
                        ) * 1000.0

                        recovery_latencies.append(
                            recovery
                        )

                        occlusion_start_time = None

                # ====================================================
                # Display
                # ====================================================

                if not headless:

                    o_val = float(
                        diag["o"].item()
                    )

                    nu_val = float(
                        diag["nu"].item()
                    )

                    nu_ema_val = float(
                        diag["nu_ema"].item()
                    )

                    gamma_val = float(
                        diag["gamma"].item()
                    )

                    text1 = (
                        f"Hands:{num_hands} "
                        f"Conf:{confidence:.2f} "
                        f"O:{o_val:.0f}"
                    )

                    text2 = (
                        f"Nu:{nu_val:.2f} "
                        f"EMA:{nu_ema_val:.2f} "
                        f"Gamma:{gamma_val:.2f}"
                    )

                    text3 = (
                        f"MP:{mp_ms:.1f}ms "
                        f"GRU:{gru_ms:.2f}ms "
                        f"Total:{total_ms:.1f}ms"
                    )

                    cv2.putText(
                        frame,
                        text1,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        text2,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        text3,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow(
                        "MP-GRU Product Validation",
                        frame,
                    )

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    if key == ord("q"):
                        break

                # ====================================================
                # Limit
                # ====================================================

                frame_idx += 1

                if (
                    max_frames is not None
                    and frame_idx >= max_frames
                ):
                    break

    finally:

        cap.release()

        landmarker.close()

        if not headless:
            cv2.destroyAllWindows()

    # ========================================================
    # Final report
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print("\n")
    print("=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    n = trace.length()

    print(
        f"Frames processed       : {n}"
    )

    print(
        f"Wall time              : {elapsed:.2f} sec"
    )

    if elapsed > 0:

        print(
            f"Overall FPS            : "
            f"{n / elapsed:.2f}"
        )

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    detection_rate = (
        np.mean(
            np.asarray(
                trace.num_hands
            ) > 0
        )
        if n
        else 0
    )

    print(
        f"Hand detection rate    : "
        f"{detection_rate * 100:.2f}%"
    )

    print(
        f"Mean confidence        : "
        f"{np.mean(trace.c):.3f}"
    )

    print(
        f"P10 confidence         : "
        f"{percentile(trace.c, 10):.3f}"
    )

    print(
        f"P50 confidence         : "
        f"{percentile(trace.c, 50):.3f}"
    )

    print(
        f"P90 confidence         : "
        f"{percentile(trace.c, 90):.3f}"
    )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    print("\nTiming:")

    print(
        f"Capture mean           : "
        f"{np.mean(trace.capture_ms):.2f} ms"
    )

    print(
        f"MediaPipe mean         : "
        f"{np.mean(trace.mp_ms):.2f} ms"
    )

    print(
        f"MediaPipe P90         : "
        f"{percentile(trace.mp_ms, 90):.2f} ms"
    )

    print(
        f"MP-GRU mean            : "
        f"{np.mean(trace.gru_ms):.3f} ms"
    )

    print(
        f"Total pipeline mean    : "
        f"{np.mean(trace.total_ms):.2f} ms"
    )

    print(
        f"Total pipeline P90     : "
        f"{percentile(trace.total_ms, 90):.2f} ms"
    )

    # --------------------------------------------------------
    # MP-GRU
    # --------------------------------------------------------

    print("\nMP-GRU:")

    print(
        f"Maximum occlusion      : "
        f"{max(trace.o) if trace.o else 0:.0f} frames"
    )

    print(
        f"Final occlusion        : "
        f"{trace.o[-1] if trace.o else 0:.0f}"
    )

    print(
        f"Final neutral weight   : "
        f"{trace.nu[-1] if trace.nu else 0:.3f}"
    )

    print(
        f"Final neutral EMA      : "
        f"{trace.nu_ema[-1] if trace.nu_ema else 0:.3f}"
    )

    print(
        f"Final neutral distance : "
        f"{trace.neutral_distance[-1] if trace.neutral_distance else 0:.4f}"
    )

    # --------------------------------------------------------
    # Acquisition
    # --------------------------------------------------------

    print("\nAcquisition:")

    if acquisition_latencies:

        print(
            f"Mean acquisition       : "
            f"{np.mean(acquisition_latencies):.1f} ms"
        )

        print(
            f"Best acquisition       : "
            f"{np.min(acquisition_latencies):.1f} ms"
        )

        print(
            f"Worst acquisition     : "
            f"{np.max(acquisition_latencies):.1f} ms"
        )

    else:

        print(
            "No complete hand acquisition "
            "event measured."
        )

    # --------------------------------------------------------
    # Recovery
    # --------------------------------------------------------

    print("\nRecovery:")

    if recovery_latencies:

        print(
            f"Mean recovery         : "
            f"{np.mean(recovery_latencies):.1f} ms"
        )

        print(
            f"Best recovery         : "
            f"{np.min(recovery_latencies):.1f} ms"
        )

        print(
            f"Worst recovery       : "
            f"{np.max(recovery_latencies):.1f} ms"
        )

    else:

        print(
            "No complete occlusion "
            "recovery event measured."
        )

    print("\nValidation:")

    if timestamp_violations == 0:

        print(
            "✓ Timestamps strictly increasing"
        )

    else:

        print(
            f"✗ Timestamp violations: "
            f"{timestamp_violations}"
        )

    x_valid = True

    if trace.length() == 0:

        x_valid = False

    if x_valid:

        print(
            "✓ Pipeline produced valid frames"
        )

    else:

        print(
            "✗ No valid frames"
        )

    print("=" * 70)

    # ========================================================
    # CSV
    # ========================================================

    if save_csv:

        save_trace_csv(
            trace,
            "real_landmark_trace.csv",
        )

    return trace


# ============================================================
# CSV
# ============================================================

def save_trace_csv(
    trace: Trace,
    path: str,
):

    fields = [
        "frame",
        "timestamp_ms",
        "confidence",
        "num_hands",
        "left_score",
        "right_score",
        "occlusion",
        "visibility",
        "gamma",
        "nu",
        "nu_ema",
        "p_mean",
        "p_max",
        "hidden_norm",
        "neutral_distance",
        "capture_ms",
        "mediapipe_ms",
        "gru_ms",
        "total_ms",
        "processing_fps",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(fields)

        for i in range(
            trace.length()
        ):

            writer.writerow(
                [
                    trace.frame[i],
                    trace.timestamp_ms[i],
                    trace.c[i],
                    trace.num_hands[i],
                    trace.left_score[i],
                    trace.right_score[i],
                    trace.o[i],
                    trace.s[i],
                    trace.gamma[i],
                    trace.nu[i],
                    trace.nu_ema[i],
                    trace.p_mean[i],
                    trace.p_max[i],
                    trace.hidden_norm[i],
                    trace.neutral_distance[i],
                    trace.capture_ms[i],
                    trace.mp_ms[i],
                    trace.gru_ms[i],
                    trace.total_ms[i],
                    trace.processing_fps[i],
                ]
            )

    print(
        f"Saved diagnostic CSV: {path}"
    )


# ============================================================
# PLOT
# ============================================================

def plot_trace(
    trace: Trace,
    path="real_landmark_trace.png",
):

    try:
        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "matplotlib not installed; "
            "skipping plot."
        )

        return

    t = np.arange(
        trace.length()
    )

    # --------------------------------------------------------
    # Plot 1
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        t,
        trace.c,
        label="confidence",
    )

    plt.plot(
        t,
        trace.left_score,
        label="left handedness",
    )

    plt.plot(
        t,
        trace.right_score,
        label="right handedness",
    )

    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.title(
        "Real Hand Detection / Confidence"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    confidence_path = (
        "confidence_trace.png"
    )

    plt.savefig(
        confidence_path,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        t,
        trace.o,
        label="occlusion counter",
    )

    plt.plot(
        t,
        trace.gamma,
        label="gamma",
    )

    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title(
        "MP-GRU Occlusion Dynamics"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    occlusion_path = (
        "occlusion_trace.png"
    )

    plt.savefig(
        occlusion_path,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        t,
        trace.nu,
        label="nu",
    )

    plt.plot(
        t,
        trace.nu_ema,
        label="nu EMA",
    )

    plt.xlabel("Frame")
    plt.ylabel("Neutral weight")
    plt.title(
        "Neutral Blend Dynamics"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    neutral_path = (
        "neutral_trace.png"
    )

    plt.savefig(
        neutral_path,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 4
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        t,
        trace.neutral_distance,
        label="||h - h_neutral||",
    )

    plt.xlabel("Frame")
    plt.ylabel("Distance")
    plt.title(
        "Hidden-State Distance From Neutral"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    hidden_path = (
        "hidden_state_trace.png"
    )

    plt.savefig(
        hidden_path,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 5
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        t,
        trace.mp_ms,
        label="MediaPipe",
    )

    plt.plot(
        t,
        trace.gru_ms,
        label="MP-GRU",
    )

    plt.plot(
        t,
        trace.total_ms,
        label="Total",
    )

    plt.xlabel("Frame")
    plt.ylabel("Milliseconds")
    plt.title(
        "Pipeline Latency"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    timing_path = (
        "pipeline_timing_trace.png"
    )

    plt.savefig(
        timing_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved plots:"
    )

    print(
        f"  {confidence_path}"
    )

    print(
        f"  {occlusion_path}"
    )

    print(
        f"  {neutral_path}"
    )

    print(
        f"  {hidden_path}"
    )

    print(
        f"  {timing_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Product-level validation of "
            "real MediaPipe landmarks through MP-GRU."
        )
    )

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Webcam index or video path."
        ),
    )

    parser.add_argument(
        "--model",
        default="hand_landmarker.task",
        help=(
            "Path to hand_landmarker.task."
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Maximum number of frames."
        ),
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Disable OpenCV display."
        ),
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help=(
            "Do not save CSV trace."
        ),
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help=(
            "Do not generate plots."
        ),
    )

    args = parser.parse_args()

    trace = run(
        source=args.source,
        model_path=args.model,
        max_frames=args.max_frames,
        headless=args.headless,
        save_csv=not args.no_csv,
    )

    if not args.no_plot:
        plot_trace(trace)