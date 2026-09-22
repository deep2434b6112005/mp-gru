"""
Advanced dataset preprocessing for MP-GRU.

Input:
    dataset/
        I_am/
            video1.mp4
            video2.mp4
        Hungry/
            video1.mp4
            ...

Output:
    cache/
        labels.json
        manifest.json
        I_am/
            video1.npz
            video2.npz
        Hungry/
            video1.npz

Each .npz contains:

    x:
        (T, 126)
        Model-ready landmark vector.

    c:
        (T,)
        Frame-level detection confidence.

    landmarks:
        (T, 2, 21, 3)
        Raw normalized MediaPipe landmarks.
        Slot 0 = left hand
        Slot 1 = right hand

    handedness:
        (T, 2)
        0 = no hand
        1 = left
        2 = right

    hand_scores:
        (T, 2)
        Per-hand MediaPipe confidence.

    bboxes:
        (T, 2, 4)
        [xmin, ymin, xmax, ymax], normalized.

    num_hands:
        (T,)
        Number of detected hands.

    timestamps_ms:
        (T,)
        Strictly increasing MediaPipe timestamps.

    frame_indices:
        (T,)
        Original video frame index.

    frame_quality:
        (T,)
        Additional frame-quality estimate.

    label:
        Label name.

    fps:
        Original video FPS.

    width:
        Original video width.

    height:
        Original video height.

    duration_sec:
        Video duration.

NEW BEHAVIOR:
    Before processing each gesture label, the script asks:

        How many hands are used for this gesture?
        1 = One hand
        2 = Two hands

    Every video is then checked against that expectation.

    A video is NOT rejected because of one or two temporary MediaPipe
    detection failures.

    Instead, the percentage of frames matching the expected hand count
    is calculated.

    If the percentage is below EXPECTED_HAND_MATCH_THRESHOLD,
    the video is flagged.

    The user is then asked:

        Delete this video? [y/n]

    The original video is deleted ONLY if the user explicitly answers
    yes.

IMPORTANT:
    This script processes EVERY frame.
    No frame skipping is used during primary cache generation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from landmark_extractor import (
    build_hand_landmarker,
    landmarks_to_vector,
)


# ============================================================
# Configuration
# ============================================================

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
LANDMARK_DIM = 3
FEATURE_SIZE = NUM_HANDS * LANDMARKS_PER_HAND * LANDMARK_DIM

# ------------------------------------------------------------
# NEW:
# Percentage of frames that must contain the expected number
# of hands for a video to be automatically accepted.
#
# Example:
#     Expected = 2 hands
#     Threshold = 0.80
#
#     85% frames have 2 hands -> ACCEPT
#     62% frames have 2 hands -> FLAG
#
# You can change this to 0.70, 0.75, 0.85, etc.
# ------------------------------------------------------------

EXPECTED_HAND_MATCH_THRESHOLD = 0.80


# ============================================================
# Utility functions
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value safely to float."""
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except Exception:
        pass

    return default


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_bbox(
    hand_landmarks,
) -> np.ndarray:
    """
    Calculate normalized bounding box from 21 landmarks.

    Returns:
        [xmin, ymin, xmax, ymax]
    """

    xs = [safe_float(p.x) for p in hand_landmarks]
    ys = [safe_float(p.y) for p in hand_landmarks]

    if not xs or not ys:
        return np.zeros(4, dtype=np.float32)

    xmin = clamp01(min(xs))
    ymin = clamp01(min(ys))
    xmax = clamp01(max(xs))
    ymax = clamp01(max(ys))

    return np.array(
        [xmin, ymin, xmax, ymax],
        dtype=np.float32,
    )


def bbox_quality(bbox: np.ndarray) -> float:
    """
    Simple geometric quality measure.

    Larger, valid hand boxes receive higher quality.
    """

    if bbox.shape != (4,):
        return 0.0

    xmin, ymin, xmax, ymax = bbox

    w = max(0.0, float(xmax - xmin))
    h = max(0.0, float(ymax - ymin))

    area = w * h

    # Avoid making large hands dominate completely.
    quality = math.sqrt(area)

    return clamp01(quality * 4.0)


def landmark_quality(
    landmarks: np.ndarray,
) -> float:
    """
    Checks landmark numerical validity.

    landmarks:
        (21, 3)
    """

    if landmarks.shape != (21, 3):
        return 0.0

    if not np.isfinite(landmarks).all():
        return 0.0

    # Normalized x/y should normally be close to [0, 1].
    xy = landmarks[:, :2]

    finite_ratio = np.isfinite(xy).all(axis=1).mean()

    # Penalize extreme numerical values.
    reasonable = (
        (xy[:, 0] > -0.5)
        & (xy[:, 0] < 1.5)
        & (xy[:, 1] > -0.5)
        & (xy[:, 1] < 1.5)
    ).mean()

    return float(
        0.5 * finite_ratio
        + 0.5 * reasonable
    )


def frame_quality_score(
    num_hands: int,
    confidence: float,
    landmarks: np.ndarray,
    bboxes: np.ndarray,
) -> float:
    """
    Composite frame-quality score.

    This is NOT used as a replacement for MediaPipe confidence.
    It is stored as auxiliary information for future experiments.
    """

    confidence = clamp01(confidence)

    if num_hands <= 0:
        return 0.0

    valid_hand_scores = []

    for h in range(min(num_hands, NUM_HANDS)):

        lq = landmark_quality(
            landmarks[h]
        )

        bq = bbox_quality(
            bboxes[h]
        )

        valid_hand_scores.append(
            0.5 * lq
            + 0.5 * bq
        )

    if not valid_hand_scores:
        return 0.0

    geometry_quality = float(
        np.mean(valid_hand_scores)
    )

    # Detection confidence is the most important component.
    score = (
        0.60 * confidence
        + 0.40 * geometry_quality
    )

    return clamp01(score)


# ============================================================
# Hand extraction
# ============================================================

def extract_hands(result):
    """
    Convert MediaPipe result into stable two-hand representation.

    Output:

        landmarks:
            (2, 21, 3)

        handedness:
            (2,)
            0 = missing
            1 = left
            2 = right

        hand_scores:
            (2,)

        bboxes:
            (2, 4)

        num_hands:
            int

    Slot convention:

        slot 0 -> LEFT
        slot 1 -> RIGHT

    This is important because the GRU should not see the same physical
    hand jumping between feature positions from frame to frame.
    """

    landmarks_out = np.zeros(
        (
            NUM_HANDS,
            LANDMARKS_PER_HAND,
            LANDMARK_DIM,
        ),
        dtype=np.float32,
    )

    handedness_out = np.zeros(
        (NUM_HANDS,),
        dtype=np.int8,
    )

    scores_out = np.zeros(
        (NUM_HANDS,),
        dtype=np.float32,
    )

    bboxes_out = np.zeros(
        (NUM_HANDS, 4),
        dtype=np.float32,
    )

    hand_landmarks_list = getattr(
        result,
        "hand_landmarks",
        None,
    )

    handedness_list = getattr(
        result,
        "handedness",
        None,
    )

    if not hand_landmarks_list:

        return (
            landmarks_out,
            handedness_out,
            scores_out,
            bboxes_out,
            0,
        )

    detected_count = min(
        len(hand_landmarks_list),
        NUM_HANDS,
    )

    for i in range(detected_count):

        hand = hand_landmarks_list[i]

        coords = np.array(
            [
                [
                    safe_float(p.x),
                    safe_float(p.y),
                    safe_float(p.z),
                ]
                for p in hand
            ],
            dtype=np.float32,
        )

        if coords.shape != (21, 3):
            continue

        # ----------------------------------------------------
        # Determine handedness
        # ----------------------------------------------------

        label = None
        score = 0.0

        if handedness_list is not None:

            try:

                categories = handedness_list[i]

                if categories:

                    category = categories[0]

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

        label_lower = str(label).lower()

        if label_lower == "left":

            slot = 0
            handedness_value = 1

        elif label_lower == "right":

            slot = 1
            handedness_value = 2

        else:

            # Fallback:
            # MediaPipe handedness is normally available,
            # but if it isn't, use the first free slot.

            if handedness_out[0] == 0:

                slot = 0

            elif handedness_out[1] == 0:

                slot = 1

            else:

                continue

            handedness_value = 0

        # Avoid overwriting a previously assigned slot.

        if handedness_out[slot] != 0:

            free_slots = np.where(
                handedness_out == 0
            )[0]

            if len(free_slots) == 0:
                continue

            slot = int(
                free_slots[0]
            )

        landmarks_out[slot] = coords

        handedness_out[slot] = (
            handedness_value
        )

        scores_out[slot] = clamp01(
            score
        )

        bboxes_out[slot] = (
            normalize_bbox(hand)
        )

    num_hands = int(
        np.count_nonzero(
            handedness_out
        )
    )

    return (
        landmarks_out,
        handedness_out,
        scores_out,
        bboxes_out,
        num_hands,
    )


# ============================================================
# Confidence extraction
# ============================================================

def get_frame_confidence(
    result,
    hand_scores: np.ndarray,
    num_hands: int,
) -> float:
    """
    Calculate one frame-level confidence.

    MediaPipe's hand detector may provide per-hand scores.
    For two hands, use the mean of available hand scores.

    This gives MP-GRU a stable scalar c_t.
    """

    if num_hands <= 0:
        return 0.0

    scores = hand_scores[
        hand_scores > 0
    ]

    if len(scores) == 0:
        return 0.0

    # Mean is preferable to max because both hands matter.
    return clamp01(
        float(
            np.mean(scores)
        )
    )


# ============================================================
# Timestamp handling
# ============================================================

def make_timestamp(
    frame_idx: int,
    fps: float,
    previous_timestamp: int,
) -> int:
    """
    Generate a strictly increasing timestamp.

    MediaPipe VIDEO mode requires:
        timestamp[t] > timestamp[t-1]

    Some videos have broken/missing timestamps, so we construct a
    reliable timeline from FPS.
    """

    if (
        not math.isfinite(fps)
        or fps <= 1.0
        or fps > 240.0
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
        timestamp = (
            previous_timestamp
            + 1
        )

    return timestamp


# ============================================================
# NEW:
# Ask expected number of hands for a label
# ============================================================

def ask_expected_hands(
    label: str,
) -> int:
    """
    Ask the user how many hands the gesture uses.

    Returns:
        1 or 2
    """

    print()
    print("=" * 70)
    print(
        f"GESTURE HAND REQUIREMENT: {label}"
    )
    print("=" * 70)

    while True:

        print()
        print(
            "How many hands are used "
            "for this gesture?"
        )

        print(
            "  [1] One hand"
        )

        print(
            "  [2] Two hands"
        )

        choice = input(
            "\nEnter 1 or 2: "
        ).strip()

        if choice == "1":

            print(
                f"\nExpected hands for "
                f"'{label}': 1"
            )

            return 1

        if choice == "2":

            print(
                f"\nExpected hands for "
                f"'{label}': 2"
            )

            return 2

        print(
            "\n[ERROR] Please enter "
            "1 or 2."
        )


# ============================================================
# NEW:
# Check whether a video satisfies the expected hand count
# ============================================================

def evaluate_hand_requirement(
    num_hands_array: np.ndarray,
    expected_hands: int,
) -> Dict[str, Any]:
    """
    Evaluate how consistently a video contains the expected
    number of hands.

    A temporary MediaPipe miss does not automatically reject
    the video.

    Returns statistics describing the match.
    """

    total_frames = int(
        len(num_hands_array)
    )

    if total_frames == 0:

        return {
            "expected_hands": expected_hands,
            "total_frames": 0,
            "matching_frames": 0,
            "matching_rate": 0.0,
            "accepted": False,
            "count_0_frames": 0,
            "count_1_frames": 0,
            "count_2_frames": 0,
        }

    matching_frames = int(
        np.sum(
            num_hands_array
            == expected_hands
        )
    )

    matching_rate = (
        matching_frames
        / total_frames
    )

    count_0 = int(
        np.sum(
            num_hands_array == 0
        )
    )

    count_1 = int(
        np.sum(
            num_hands_array == 1
        )
    )

    count_2 = int(
        np.sum(
            num_hands_array == 2
        )
    )

    accepted = (
        matching_rate
        >= EXPECTED_HAND_MATCH_THRESHOLD
    )

    return {
        "expected_hands": int(
            expected_hands
        ),
        "total_frames": total_frames,
        "matching_frames": matching_frames,
        "matching_rate": float(
            matching_rate
        ),
        "accepted": bool(
            accepted
        ),
        "count_0_frames": count_0,
        "count_1_frames": count_1,
        "count_2_frames": count_2,
    }


# ============================================================
# NEW:
# Ask permission to delete a bad video
# ============================================================

def ask_delete_video(
    video_path: Path,
    hand_stats: Dict[str, Any],
) -> bool:
    """
    Ask the user whether a video should be deleted.

    The file is NEVER deleted without explicit confirmation.
    """

    expected = hand_stats[
        "expected_hands"
    ]

    matching_rate = (
        hand_stats["matching_rate"]
        * 100.0
    )

    count_0 = hand_stats[
        "count_0_frames"
    ]

    count_1 = hand_stats[
        "count_1_frames"
    ]

    count_2 = hand_stats[
        "count_2_frames"
    ]

    print()
    print("!" * 70)
    print("VIDEO DOES NOT MATCH EXPECTED HAND COUNT")
    print("!" * 70)

    print(
        f"Video          : {video_path}"
    )

    print(
        f"Expected hands : {expected}"
    )

    print(
        f"Matching       : "
        f"{matching_rate:.1f}% of frames"
    )

    print(
        f"0-hand frames  : {count_0}"
    )

    print(
        f"1-hand frames  : {count_1}"
    )

    print(
        f"2-hand frames  : {count_2}"
    )

    print(
        f"Required       : "
        f"{EXPECTED_HAND_MATCH_THRESHOLD * 100:.1f}%"
    )

    print()
    print(
        "This video may be unsuitable "
        "for this gesture."
    )

    while True:

        choice = input(
            "\nDelete this video? [y/n]: "
        ).strip().lower()

        if choice in (
            "y",
            "yes",
        ):
            return True

        if choice in (
            "n",
            "no",
        ):
            return False

        print(
            "Please enter y or n."
        )


# ============================================================
# Video processing
# ============================================================

def process_video(
    model_path: str,
    video_path: Path,
    save_debug: bool = False,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, Any],
]:
    """
    Process one complete video.

    A new MediaPipe VIDEO landmarker is created
    for every video.
    """

    print(
        f"\nProcessing: {video_path}"
    )

    landmarker, mp = (
        build_hand_landmarker(
            model_path
        )
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        landmarker.close()

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    fps = safe_float(
        cap.get(
            cv2.CAP_PROP_FPS
        ),
        30.0,
    )

    if (
        fps <= 1.0
        or fps > 240.0
    ):
        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames_estimate = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    xs: List[np.ndarray] = []
    cs: List[float] = []

    landmarks_all: List[
        np.ndarray
    ] = []

    handedness_all: List[
        np.ndarray
    ] = []

    hand_scores_all: List[
        np.ndarray
    ] = []

    bboxes_all: List[
        np.ndarray
    ] = []

    num_hands_all: List[
        int
    ] = []

    quality_all: List[
        float
    ] = []

    timestamps_all: List[
        int
    ] = []

    frame_indices_all: List[
        int
    ] = []

    frame_idx = 0
    previous_timestamp = -1

    valid_frames = 0
    empty_frames = 0
    processing_errors = 0

    start_time = time.perf_counter()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # ------------------------------------------------
            # RGB conversion
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb,
            )

            # ------------------------------------------------
            # Robust timestamp
            # ------------------------------------------------

            timestamp_ms = (
                make_timestamp(
                    frame_idx,
                    fps,
                    previous_timestamp,
                )
            )

            previous_timestamp = (
                timestamp_ms
            )

            # ------------------------------------------------
            # MediaPipe inference
            # ------------------------------------------------

            try:

                result = (
                    landmarker
                    .detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )
                )

            except Exception as exc:

                processing_errors += 1

                print(
                    f"  [WARN] frame "
                    f"{frame_idx}: "
                    f"MediaPipe error: "
                    f"{exc}"
                )

                # Preserve temporal alignment.

                x_np = np.zeros(
                    FEATURE_SIZE,
                    dtype=np.float32,
                )

                c_val = 0.0

                lm = np.zeros(
                    (
                        NUM_HANDS,
                        21,
                        3,
                    ),
                    dtype=np.float32,
                )

                handedness = np.zeros(
                    (NUM_HANDS,),
                    dtype=np.int8,
                )

                hand_scores = np.zeros(
                    (NUM_HANDS,),
                    dtype=np.float32,
                )

                bboxes = np.zeros(
                    (NUM_HANDS, 4),
                    dtype=np.float32,
                )

                num_hands = 0

                quality = 0.0

            else:

                # ------------------------------------------------
                # Model-ready vector
                # ------------------------------------------------

                x_np, c_val = (
                    landmarks_to_vector(
                        result
                    )
                )

                x_np = np.asarray(
                    x_np,
                    dtype=np.float32,
                )

                if x_np.shape != (
                    FEATURE_SIZE,
                ):

                    raise RuntimeError(
                        "Unexpected landmark "
                        f"vector shape "
                        f"{x_np.shape}; "
                        f"expected "
                        f"{(FEATURE_SIZE,)}"
                    )

                c_val = clamp01(
                    safe_float(
                        c_val
                    )
                )

                # ------------------------------------------------
                # Raw detailed hand information
                # ------------------------------------------------

                (
                    lm,
                    handedness,
                    hand_scores,
                    bboxes,
                    num_hands,
                ) = extract_hands(
                    result
                )

                # ------------------------------------------------
                # Quality
                # ------------------------------------------------

                quality = (
                    frame_quality_score(
                        num_hands=num_hands,
                        confidence=c_val,
                        landmarks=lm,
                        bboxes=bboxes,
                    )
                )

            # ----------------------------------------------------
            # Save frame
            # ----------------------------------------------------

            xs.append(x_np)
            cs.append(c_val)

            landmarks_all.append(lm)

            handedness_all.append(
                handedness
            )

            hand_scores_all.append(
                hand_scores
            )

            bboxes_all.append(
                bboxes
            )

            num_hands_all.append(
                num_hands
            )

            quality_all.append(
                quality
            )

            timestamps_all.append(
                timestamp_ms
            )

            frame_indices_all.append(
                frame_idx
            )

            if num_hands > 0:

                valid_frames += 1

            else:

                empty_frames += 1

            frame_idx += 1

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if frame_idx % 100 == 0:

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                proc_fps = (
                    frame_idx / elapsed
                    if elapsed > 0
                    else 0
                )

                if (
                    total_frames_estimate
                    > 0
                ):

                    percent = (
                        100.0
                        * frame_idx
                        / total_frames_estimate
                    )

                    print(
                        f"  {frame_idx}/"
                        f"{total_frames_estimate} "
                        f"({percent:.1f}%) "
                        f"| "
                        f"{proc_fps:.1f} "
                        f"frames/s"
                    )

                else:

                    print(
                        f"  {frame_idx} "
                        f"frames "
                        f"| "
                        f"{proc_fps:.1f} "
                        f"frames/s"
                    )

    finally:

        cap.release()
        landmarker.close()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # Validate
    # ========================================================

    if len(xs) == 0:

        raise RuntimeError(
            f"No frames extracted "
            f"from {video_path}"
        )

    x = np.stack(
        xs
    ).astype(
        np.float32,
        copy=False,
    )

    c = np.asarray(
        cs,
        dtype=np.float32,
    )

    landmarks = np.stack(
        landmarks_all
    ).astype(
        np.float32,
        copy=False,
    )

    handedness = np.stack(
        handedness_all
    ).astype(
        np.int8,
        copy=False,
    )

    hand_scores = np.stack(
        hand_scores_all
    ).astype(
        np.float32,
        copy=False,
    )

    bboxes = np.stack(
        bboxes_all
    ).astype(
        np.float32,
        copy=False,
    )

    num_hands = np.asarray(
        num_hands_all,
        dtype=np.int8,
    )

    frame_quality = np.asarray(
        quality_all,
        dtype=np.float32,
    )

    timestamps_ms = np.asarray(
        timestamps_all,
        dtype=np.int64,
    )

    frame_indices = np.asarray(
        frame_indices_all,
        dtype=np.int32,
    )

    # --------------------------------------------------------
    # Strict timestamp validation
    # --------------------------------------------------------

    if len(timestamps_ms) > 1:

        diffs = np.diff(
            timestamps_ms
        )

        if not np.all(
            diffs > 0
        ):

            raise RuntimeError(
                "Timestamp validation "
                "failed: timestamps are "
                "not strictly increasing."
            )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    assert (
        x.shape[1]
        == FEATURE_SIZE
    )

    assert (
        landmarks.shape[1:]
        == (
            NUM_HANDS,
            21,
            3,
        )
    )

    assert (
        c.shape[0]
        == x.shape[0]
    )

    assert (
        timestamps_ms.shape[0]
        == x.shape[0]
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    duration_sec = (
        float(
            timestamps_ms[-1]
        )
        / 1000.0
        if len(timestamps_ms)
        else 0.0
    )

    detection_rate = (
        valid_frames
        / len(x)
        if len(x)
        else 0.0
    )

    mean_confidence = (
        float(np.mean(c))
        if len(c)
        else 0.0
    )

    mean_quality = (
        float(
            np.mean(
                frame_quality
            )
        )
        if len(frame_quality)
        else 0.0
    )

    max_hands = int(
        np.max(num_hands)
        if len(num_hands)
        else 0
    )

    metadata = {
        "video": str(video_path),
        "frames": int(len(x)),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "duration_sec": duration_sec,
        "detection_rate": detection_rate,
        "mean_confidence": mean_confidence,
        "mean_quality": mean_quality,
        "max_hands": max_hands,
        "empty_frames": int(
            empty_frames
        ),
        "processing_errors": int(
            processing_errors
        ),
        "processing_time_sec": float(
            elapsed
        ),
        "processing_fps": (
            float(
                len(x) / elapsed
            )
            if elapsed > 0
            else 0.0
        ),
    }

    data = {
        "x": x,
        "c": c,
        "landmarks": landmarks,
        "handedness": handedness,
        "hand_scores": hand_scores,
        "bboxes": bboxes,
        "num_hands": num_hands,
        "frame_quality": frame_quality,
        "timestamps_ms": timestamps_ms,
        "frame_indices": frame_indices,
    }

    return data, metadata


# ============================================================
# Cache validation
# ============================================================

def validate_cache_file(
    path: Path,
) -> Tuple[bool, str]:
    """
    Validate one generated .npz file.
    """

    try:

        data = np.load(
            path,
            allow_pickle=False,
        )

        required = [
            "x",
            "c",
            "landmarks",
            "handedness",
            "hand_scores",
            "bboxes",
            "num_hands",
            "frame_quality",
            "timestamps_ms",
            "frame_indices",
        ]

        for key in required:

            if key not in data:

                return (
                    False,
                    f"missing key '{key}'",
                )

        x = data["x"]
        c = data["c"]
        lm = data["landmarks"]
        ts = data["timestamps_ms"]

        if x.ndim != 2:

            return (
                False,
                "x must be 2-D",
            )

        if x.shape[1] != FEATURE_SIZE:

            return (
                False,
                f"x feature size="
                f"{x.shape[1]}, "
                f"expected="
                f"{FEATURE_SIZE}",
            )

        if lm.ndim != 4:

            return (
                False,
                "landmarks must be 4-D",
            )

        if lm.shape[1:] != (
            2,
            21,
            3,
        ):

            return (
                False,
                f"invalid landmarks "
                f"shape {lm.shape}",
            )

        if c.shape[0] != x.shape[0]:

            return (
                False,
                "c length mismatch",
            )

        if ts.shape[0] != x.shape[0]:

            return (
                False,
                "timestamp length mismatch",
            )

        if len(ts) > 1:

            if not np.all(
                np.diff(ts) > 0
            ):

                return (
                    False,
                    "timestamps are not "
                    "strictly increasing",
                )

        if not np.isfinite(
            x
        ).all():

            return (
                False,
                "x contains NaN/Inf",
            )

        if not np.isfinite(
            c
        ).all():

            return (
                False,
                "c contains NaN/Inf",
            )

        return True, "OK"

    except Exception as exc:

        return False, str(exc)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Advanced full-frame "
            "MediaPipe preprocessing "
            "for MP-GRU gesture datasets."
        )
    )

    parser.add_argument(
        "--dataset",
        default="dataset",
        help=(
            "Root dataset directory: "
            "<dataset>/<label>/*.mp4"
        ),
    )

    parser.add_argument(
        "--cache",
        default="cache",
        help="Output cache directory.",
    )

    parser.add_argument(
        "--model",
        default="hand_landmarker.task",
        help=(
            "MediaPipe hand landmarker "
            "model."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Reprocess existing .npz "
            "files."
        ),
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Validate generated cache "
            "files."
        ),
    )

    parser.add_argument(
        "--video-ext",
        default=(
            ".mp4,.mov,.avi,.mkv,.webm"
        ),
        help=(
            "Comma-separated video "
            "extensions."
        ),
    )

    args = parser.parse_args()

    dataset_root = Path(
        args.dataset
    )

    cache_root = Path(
        args.cache
    )

    model_path = Path(
        args.model
    )

    # ========================================================
    # Validate paths
    # ========================================================

    if not dataset_root.is_dir():

        parser.error(
            "Dataset directory does not "
            f"exist: {dataset_root}"
        )

    if not model_path.is_file():

        parser.error(
            "MediaPipe model does not "
            f"exist: {model_path}"
        )

    cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    extensions = tuple(
        e.strip().lower()
        for e in args.video_ext.split(",")
        if e.strip()
    )

    # ========================================================
    # Find labels
    # ========================================================

    label_dirs = sorted(
        [
            p
            for p in dataset_root.iterdir()
            if p.is_dir()
        ],
        key=lambda p:
            p.name.lower(),
    )

    if not label_dirs:

        raise RuntimeError(
            "No label folders found "
            f"under {dataset_root}"
        )

    labels = [
        p.name
        for p in label_dirs
    ]

    label_to_idx = {
        label: idx
        for idx, label in enumerate(
            labels
        )
    }

    with open(
        cache_root / "labels.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            label_to_idx,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 70)
    print(
        "MP-GRU ADVANCED DATASET "
        "PREPROCESSOR"
    )
    print("=" * 70)

    print(
        f"Dataset : {dataset_root}"
    )

    print(
        f"Cache   : {cache_root}"
    )

    print(
        f"Model   : {model_path}"
    )

    print(
        f"Labels  : {len(labels)}"
    )

    print(
        "Hand match threshold : "
        f"{EXPECTED_HAND_MATCH_THRESHOLD * 100:.1f}%"
    )

    print()

    for idx, label in enumerate(
        labels
    ):

        print(
            f"  {idx:2d} -> {label}"
        )

    print("=" * 70)

    # ========================================================
    # Dataset statistics
    # ========================================================

    total_processed = 0
    total_skipped = 0
    total_failed = 0
    total_frames = 0

    total_deleted = 0
    total_flagged_kept = 0

    manifest = {
        "version": 3,
        "feature_size": FEATURE_SIZE,
        "num_hands": NUM_HANDS,
        "landmarks_per_hand":
            LANDMARKS_PER_HAND,
        "landmark_dimensions":
            LANDMARK_DIM,
        "expected_hand_match_threshold":
            EXPECTED_HAND_MATCH_THRESHOLD,
        "labels": label_to_idx,
        "videos": [],
    }

    global_start = time.perf_counter()

    # ========================================================
    # Process labels
    # ========================================================

    for label_dir in label_dirs:

        label = label_dir.name

        # ----------------------------------------------------
        # NEW:
        # Ask once for this label
        # ----------------------------------------------------

        expected_hands = (
            ask_expected_hands(
                label
            )
        )

        output_dir = (
            cache_root / label
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        videos = sorted(
            [
                p
                for p in label_dir.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in extensions
            ],
            key=lambda p:
                p.name.lower(),
        )

        print()
        print(
            f"[{label}] "
            f"{len(videos)} videos"
        )

        print(
            f"Expected hand count: "
            f"{expected_hands}"
        )

        # ====================================================
        # Process videos
        # ====================================================

        for video_path in videos:

            output_path = (
                output_dir
                / f"{video_path.stem}.npz"
            )

            # ------------------------------------------------
            # Existing cache
            # ------------------------------------------------

            if (
                output_path.exists()
                and not args.overwrite
            ):

                total_skipped += 1

                print(
                    f"  [SKIP] "
                    f"{video_path.name}"
                )

                if args.validate:

                    ok, message = (
                        validate_cache_file(
                            output_path
                        )
                    )

                    print(
                        f"         validation: "
                        f"{message}"
                    )

                continue

            # ------------------------------------------------
            # Process video
            # ------------------------------------------------

            try:

                data, metadata = (
                    process_video(
                        model_path=str(
                            model_path
                        ),
                        video_path=video_path,
                    )
                )

                # ====================================================
                # NEW:
                # Evaluate expected hand count
                # ====================================================

                hand_stats = (
                    evaluate_hand_requirement(
                        num_hands_array=(
                            data[
                                "num_hands"
                            ]
                        ),
                        expected_hands=(
                            expected_hands
                        ),
                    )
                )

                matching_rate = (
                    hand_stats[
                        "matching_rate"
                    ]
                )

                print()
                print(
                    f"  Hand check: "
                    f"{matching_rate * 100:.1f}% "
                    f"of frames match "
                    f"expected {expected_hands} "
                    f"hand(s)"
                )

                # ------------------------------------------------
                # Bad hand-count video
                # ------------------------------------------------

                if not hand_stats[
                    "accepted"
                ]:

                    delete_video = (
                        ask_delete_video(
                            video_path,
                            hand_stats,
                        )
                    )

                    if delete_video:

                        # --------------------------------------------
                        # Delete original video
                        # --------------------------------------------

                        try:

                            video_path.unlink()

                        except Exception as delete_exc:

                            print(
                                "\n[ERROR] "
                                "Could not delete "
                                f"{video_path}: "
                                f"{delete_exc}"
                            )

                            # Do NOT create cache if deletion
                            # was requested but failed.

                            total_failed += 1

                            continue

                        # --------------------------------------------
                        # Remove stale cache if it exists
                        # --------------------------------------------

                        if output_path.exists():

                            try:

                                output_path.unlink()

                            except Exception as cache_exc:

                                print(
                                    "\n[WARN] "
                                    "Could not remove "
                                    "existing cache: "
                                    f"{cache_exc}"
                                )

                        total_deleted += 1

                        print()
                        print(
                            f"  [DELETED] "
                            f"{video_path}"
                        )

                        print(
                            "  No cache file "
                            "was created."
                        )

                        continue

                    else:

                        # User chose to keep it.
                        # Process/cache it, but record that
                        # it failed the expected hand check.

                        total_flagged_kept += 1

                        print()
                        print(
                            "  [KEPT BY USER]"
                        )

                        print(
                            "  The video will "
                            "still be cached."
                        )

                # ====================================================
                # Add label information
                # ====================================================

                data["label"] = np.array(
                    label
                )

                # ------------------------------------------------
                # Add metadata arrays/scalars
                # ------------------------------------------------

                data["fps"] = np.array(
                    metadata["fps"],
                    dtype=np.float32,
                )

                data["width"] = np.array(
                    metadata["width"],
                    dtype=np.int32,
                )

                data["height"] = np.array(
                    metadata["height"],
                    dtype=np.int32,
                )

                data["duration_sec"] = (
                    np.array(
                        metadata[
                            "duration_sec"
                        ],
                        dtype=np.float32,
                    )
                )

                # ------------------------------------------------
                # NEW:
                # Store expected hand information inside cache
                # ------------------------------------------------

                data[
                    "expected_hands"
                ] = np.array(
                    expected_hands,
                    dtype=np.int8,
                )

                data[
                    "hand_match_rate"
                ] = np.array(
                    hand_stats[
                        "matching_rate"
                    ],
                    dtype=np.float32,
                )

                data[
                    "hand_requirement_passed"
                ] = np.array(
                    hand_stats[
                        "accepted"
                    ],
                    dtype=np.bool_,
                )

                # ------------------------------------------------
                # Save compressed cache
                # ------------------------------------------------

                np.savez_compressed(
                    output_path,
                    **data,
                )

                # ------------------------------------------------
                # Validate newly written cache
                # ------------------------------------------------

                if args.validate:

                    ok, message = (
                        validate_cache_file(
                            output_path
                        )
                    )

                    if not ok:

                        raise RuntimeError(
                            "Cache validation "
                            "failed: "
                            f"{message}"
                        )

                # ------------------------------------------------
                # Manifest metadata
                # ------------------------------------------------

                metadata["label"] = label

                metadata[
                    "label_index"
                ] = label_to_idx[
                    label
                ]

                metadata[
                    "expected_hands"
                ] = expected_hands

                metadata[
                    "hand_match_rate"
                ] = hand_stats[
                    "matching_rate"
                ]

                metadata[
                    "hand_requirement_passed"
                ] = hand_stats[
                    "accepted"
                ]

                metadata[
                    "hand_count_0_frames"
                ] = hand_stats[
                    "count_0_frames"
                ]

                metadata[
                    "hand_count_1_frames"
                ] = hand_stats[
                    "count_1_frames"
                ]

                metadata[
                    "hand_count_2_frames"
                ] = hand_stats[
                    "count_2_frames"
                ]

                metadata[
                    "cache_file"
                ] = str(
                    output_path
                )

                manifest[
                    "videos"
                ].append(
                    metadata
                )

                total_processed += 1

                total_frames += (
                    metadata[
                        "frames"
                    ]
                )

                # ------------------------------------------------
                # Print result
                # ------------------------------------------------

                print()

                if hand_stats[
                    "accepted"
                ]:

                    print(
                        f"  [OK] "
                        f"{video_path.name}"
                    )

                else:

                    print(
                        f"  [OK - KEPT] "
                        f"{video_path.name}"
                    )

                print(
                    f"       frames      : "
                    f"{metadata['frames']}"
                )

                print(
                    f"       duration    : "
                    f"{metadata['duration_sec']:.2f}s"
                )

                print(
                    f"       detection   : "
                    f"{metadata['detection_rate'] * 100:.1f}%"
                )

                print(
                    f"       confidence  : "
                    f"{metadata['mean_confidence']:.3f}"
                )

                print(
                    f"       quality     : "
                    f"{metadata['mean_quality']:.3f}"
                )

                print(
                    f"       hands max   : "
                    f"{metadata['max_hands']}"
                )

                print(
                    f"       expected    : "
                    f"{expected_hands}"
                )

                print(
                    f"       hand match  : "
                    f"{hand_stats['matching_rate'] * 100:.1f}%"
                )

            except Exception as exc:

                total_failed += 1

                print(
                    f"  [ERROR] "
                    f"{video_path}: "
                    f"{exc}"
                )

    # ========================================================
    # Save manifest
    # ========================================================

    elapsed = (
        time.perf_counter()
        - global_start
    )

    manifest[
        "summary"
    ] = {
        "processed_videos":
            total_processed,

        "skipped_videos":
            total_skipped,

        "failed_videos":
            total_failed,

        "deleted_videos":
            total_deleted,

        "flagged_but_kept_videos":
            total_flagged_kept,

        "total_frames":
            total_frames,

        "elapsed_sec":
            elapsed,
    }

    with open(
        cache_root / "manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Final report
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "PREPROCESSING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Processed videos       : "
        f"{total_processed}"
    )

    print(
        f"Skipped videos         : "
        f"{total_skipped}"
    )

    print(
        f"Failed videos          : "
        f"{total_failed}"
    )

    print(
        f"Deleted videos         : "
        f"{total_deleted}"
    )

    print(
        f"Flagged but kept       : "
        f"{total_flagged_kept}"
    )

    print(
        f"Total frames           : "
        f"{total_frames}"
    )

    print(
        f"Time                   : "
        f"{elapsed:.1f} sec"
    )

    print(
        f"Labels                 : "
        f"{len(labels)}"
    )

    print(
        f"Feature size           : "
        f"{FEATURE_SIZE}"
    )

    print(
        f"Hand match threshold   : "
        f"{EXPECTED_HAND_MATCH_THRESHOLD * 100:.1f}%"
    )

    print(
        f"Cache directory        : "
        f"{cache_root}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()