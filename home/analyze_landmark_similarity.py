"""
Step 3b / Stage 1 v2: Raw vs Wrist-Relative vs Motion similarity analysis.

READ-ONLY ANALYSIS
------------------
This script reads existing cached MediaPipe landmark sequences:

    cache/
        <label>/
            video1.npz
            video2.npz
            ...
        labels.json

Each .npz is expected to contain:

    x : (T, 126) float32
        2 hands x 21 landmarks x (x,y,z)

        [0:63]   = left hand
        [63:126] = right hand

        A missing hand is represented by 63 zeros.

    c : (T,) float32
        MediaPipe detection confidence.

The script does NOT modify:

    preprocess_dataset.py
    validate_real_landmarks.py
    mp_gru.py
    gesture_classifier.pt
    cache/*.npz

It only reads the existing cache and optionally writes a JSON report.

THREE REPRESENTATIONS
---------------------

1. RAW POSITION
   Original MediaPipe coordinates.

   This answers:
       "Are these gestures physically located/configured similarly?"

2. WRIST-RELATIVE POSITION
   For each detected hand:

       landmark_relative = landmark - wrist

   This removes the absolute hand position from the image and focuses
   more on the actual hand shape/finger configuration.

   This answers:
       "Do these gestures have similar hand shapes?"

3. MOTION
   Frame-to-frame landmark movement:

       delta[t] = x[t] - x[t-1]

   Only consecutive reliable frames are used.

   This answers:
       "Do these gestures move in a similar way?"

RELIABILITY
-----------
A frame is reliable when:

    confidence >= --conf-threshold
    AND
    at least one hand is detected.

For motion, BOTH frames in a consecutive pair must be reliable.

LOW-SAMPLE CLASSES
------------------
Classes with fewer than --min-samples are flagged.

Their similarity scores should be treated as hypotheses rather than
strong conclusions.

USAGE
-----

    python analyze_landmark_similarity_v2.py --cache-dir cache

More detailed:

    python analyze_landmark_similarity_v2.py ^
        --cache-dir cache ^
        --min-samples 10 ^
        --conf-threshold 0.5 ^
        --top-k 15 ^
        --save-json report_v2.json

"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

N_LANDMARKS = 21
COORDS = 3

HAND_DIM = N_LANDMARKS * COORDS       # 63
TOTAL_DIM = HAND_DIM * 2              # 126

LEFT_START = 0
LEFT_END = HAND_DIM

RIGHT_START = HAND_DIM
RIGHT_END = TOTAL_DIM


# ============================================================
# LABEL HELPERS
# ============================================================

def strip_numeric_prefix(folder_name: str) -> str:
    """
    Convert:

        '1_help me' -> 'help me'
        '2-water'   -> 'water'

    If there is no numeric prefix, leave the name unchanged.
    """
    m = re.match(r"^\d+[_\-\s]+(.*)$", folder_name)

    if m:
        return m.group(1)

    return folder_name


# ============================================================
# CACHE LOADING
# ============================================================

def load_cache(cache_dir: Path):
    """
    Load all usable .npz files.

    Returns:

        data[label] = [
            (x, c, filename),
            ...
        ]

    where:

        x = (T,126)
        c = (T,)
    """

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache directory not found: {cache_dir}"
        )

    data = {}

    for class_dir in sorted(cache_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        label = strip_numeric_prefix(class_dir.name)

        videos = []

        for npz_path in sorted(class_dir.glob("*.npz")):

            try:

                with np.load(npz_path, allow_pickle=True) as npz:

                    if "x" not in npz or "c" not in npz:

                        print(
                            f"  ! skipping {npz_path} "
                            f"(missing x/c keys)"
                        )
                        continue

                    x = npz["x"].astype(np.float32)
                    c = npz["c"].astype(np.float32)

            except Exception as e:

                print(
                    f"  ! skipping {npz_path}: {e}"
                )
                continue

            # Validate x

            if x.ndim != 2 or x.shape[1] != TOTAL_DIM:

                print(
                    f"  ! skipping {npz_path} "
                    f"(unexpected x shape {x.shape})"
                )
                continue

            # Validate c

            if c.ndim != 1 or c.shape[0] != x.shape[0]:

                print(
                    f"  ! skipping {npz_path} "
                    f"(x/c length mismatch: "
                    f"{x.shape[0]} vs {c.shape[0]})"
                )
                continue

            videos.append(
                (x, c, npz_path.name)
            )

        if videos:

            data[label] = videos

        else:

            print(
                f"  ! no usable .npz files found under "
                f"{class_dir}"
            )

    return data


# ============================================================
# HAND DETECTION
# ============================================================

def hand_present_mask(
    x: np.ndarray,
    hand: str
) -> np.ndarray:
    """
    Return a boolean mask indicating whether a hand exists
    in each frame.

    Missing hands are represented by 63 zeros.
    """

    if hand == "left":

        block = x[:, LEFT_START:LEFT_END]

    elif hand == "right":

        block = x[:, RIGHT_START:RIGHT_END]

    else:

        raise ValueError(
            "hand must be 'left' or 'right'"
        )

    return np.any(block != 0.0, axis=1)


def any_hand_present(x: np.ndarray) -> np.ndarray:
    """
    True if at least one hand is detected.
    """

    left = hand_present_mask(x, "left")
    right = hand_present_mask(x, "right")

    return left | right


# ============================================================
# RELIABLE FRAMES
# ============================================================

def reliable_frame_mask(
    x: np.ndarray,
    c: np.ndarray,
    conf_threshold: float
) -> np.ndarray:
    """
    A frame is reliable when:

        confidence >= threshold
        AND
        at least one hand exists.
    """

    confidence_ok = c >= conf_threshold

    hand_ok = any_hand_present(x)

    return confidence_ok & hand_ok


# ============================================================
# WRIST-RELATIVE REPRESENTATION
# ============================================================

def wrist_relative_sequence(
    x: np.ndarray
) -> np.ndarray:
    """
    Convert absolute landmark coordinates into coordinates
    relative to the wrist.

    MediaPipe landmark 0 is the wrist.

    For each detected hand:

        relative_landmark =
            landmark_position - wrist_position

    Missing hands remain zero.

    Input:
        x = (T,126)

    Output:
        relative = (T,126)
    """

    relative = np.zeros_like(x)

    # --------------------------------------------------------
    # LEFT HAND
    # --------------------------------------------------------

    left = x[:, LEFT_START:LEFT_END].reshape(
        -1,
        N_LANDMARKS,
        COORDS
    )

    left_present = np.any(
        left != 0.0,
        axis=(1, 2)
    )

    for t in range(left.shape[0]):

        if not left_present[t]:
            continue

        wrist = left[t, 0].copy()

        left[t] = left[t] - wrist

    relative[:, LEFT_START:LEFT_END] = left.reshape(
        -1,
        HAND_DIM
    )

    # --------------------------------------------------------
    # RIGHT HAND
    # --------------------------------------------------------

    right = x[:, RIGHT_START:RIGHT_END].reshape(
        -1,
        N_LANDMARKS,
        COORDS
    )

    right_present = np.any(
        right != 0.0,
        axis=(1, 2)
    )

    for t in range(right.shape[0]):

        if not right_present[t]:
            continue

        wrist = right[t, 0].copy()

        right[t] = right[t] - wrist

    relative[:, RIGHT_START:RIGHT_END] = right.reshape(
        -1,
        HAND_DIM
    )

    return relative.astype(np.float32)


# ============================================================
# POSITION SUMMARY
# ============================================================

def position_summary(
    x: np.ndarray,
    reliable: np.ndarray
) -> np.ndarray:
    """
    Compute mean pose across reliable frames.

    Output:

        (126,)
    """

    if reliable.any():

        return x[reliable].mean(
            axis=0
        ).astype(np.float32)

    # Fallback for bad clips

    return x.mean(
        axis=0
    ).astype(np.float32)


# ============================================================
# MOTION SUMMARY
# ============================================================

def motion_summary(
    x: np.ndarray,
    reliable: np.ndarray
) -> np.ndarray:
    """
    Create a fixed-size motion descriptor.

    For each of 126 coordinates:

        mean(delta)
        std(delta)
        mean(abs(delta))
        max(abs(delta))

    = 126 * 4 = 504

    Plus:

        left-hand mean landmark motion
        right-hand mean landmark motion

    = 2

    Total:

        506 dimensions
    """

    T = x.shape[0]

    deltas = []

    left_mags = []
    right_mags = []

    for t in range(1, T):

        # BOTH frames must be reliable.

        if not (
            reliable[t]
            and reliable[t - 1]
        ):
            continue

        previous = x[t - 1]
        current = x[t]

        d = current - previous

        deltas.append(d)

        # ----------------------------------------------------
        # LEFT HAND MOTION
        # ----------------------------------------------------

        left_d = d[
            LEFT_START:LEFT_END
        ].reshape(
            N_LANDMARKS,
            COORDS
        )

        left_mag = np.sqrt(
            (left_d ** 2).sum(axis=1)
        ).mean()

        left_mags.append(
            left_mag
        )

        # ----------------------------------------------------
        # RIGHT HAND MOTION
        # ----------------------------------------------------

        right_d = d[
            RIGHT_START:RIGHT_END
        ].reshape(
            N_LANDMARKS,
            COORDS
        )

        right_mag = np.sqrt(
            (right_d ** 2).sum(axis=1)
        ).mean()

        right_mags.append(
            right_mag
        )

    # No valid motion pairs

    if not deltas:

        return np.zeros(
            TOTAL_DIM * 4 + 2,
            dtype=np.float32
        )

    deltas = np.stack(
        deltas,
        axis=0
    )

    mean_d = deltas.mean(
        axis=0
    )

    std_d = deltas.std(
        axis=0
    )

    mean_abs_d = np.abs(
        deltas
    ).mean(
        axis=0
    )

    max_abs_d = np.abs(
        deltas
    ).max(
        axis=0
    )

    left_mag = float(
        np.mean(left_mags)
    )

    right_mag = float(
        np.mean(right_mags)
    )

    descriptor = np.concatenate(
        [
            mean_d,
            std_d,
            mean_abs_d,
            max_abs_d,
            [
                left_mag,
                right_mag
            ]
        ]
    )

    return descriptor.astype(
        np.float32
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_sim(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """
    Cosine similarity.
    """

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0.0 or nb == 0.0:

        return 0.0

    value = np.dot(
        a,
        b
    ) / (
        na * nb
    )

    # Numerical safety

    return float(
        np.clip(
            value,
            -1.0,
            1.0
        )
    )


def average_pairwise_similarity(
    vectors
) -> float:
    """
    Average cosine similarity between
    every distinct pair.
    """

    n = len(vectors)

    if n < 2:

        return float("nan")

    total = 0.0
    count = 0

    for i in range(n):

        for j in range(
            i + 1,
            n
        ):

            total += cosine_sim(
                vectors[i],
                vectors[j]
            )

            count += 1

    return total / count


# ============================================================
# BUILD REPRESENTATIONS
# ============================================================

def build_video_representations(
    x: np.ndarray,
    c: np.ndarray,
    conf_threshold: float
):
    """
    Build:

        raw position
        wrist-relative position
        raw motion
        wrist-relative motion

    for one video.
    """

    reliable = reliable_frame_mask(
        x,
        c,
        conf_threshold
    )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    raw_position = position_summary(
        x,
        reliable
    )

    # --------------------------------------------------------
    # WRIST-RELATIVE
    # --------------------------------------------------------

    relative_x = wrist_relative_sequence(
        x
    )

    relative_position = position_summary(
        relative_x,
        reliable
    )

    # --------------------------------------------------------
    # MOTION
    # --------------------------------------------------------

    raw_motion = motion_summary(
        x,
        reliable
    )

    relative_motion = motion_summary(
        relative_x,
        reliable
    )

    return {
        "raw_position": raw_position,
        "relative_position": relative_position,
        "raw_motion": raw_motion,
        "relative_motion": relative_motion,
        "reliable_frames": int(
            reliable.sum()
        ),
        "total_frames": int(
            len(reliable)
        ),
    }


# ============================================================
# PAIR GENERATION
# ============================================================

def make_pair_results(
    labels,
    class_vectors,
    sample_counts,
    min_samples
):
    """
    Compare every class pair.

    class_vectors contains:

        raw_position
        relative_position
        raw_motion
        relative_motion
    """

    results = []

    for i in range(
        len(labels)
    ):

        for j in range(
            i + 1,
            len(labels)
        ):

            a = labels[i]
            b = labels[j]

            result = {

                "a": a,

                "b": b,

                "raw_position":
                    cosine_sim(
                        class_vectors[a]["raw_position"],
                        class_vectors[b]["raw_position"]
                    ),

                "wrist_relative_position":
                    cosine_sim(
                        class_vectors[a]["relative_position"],
                        class_vectors[b]["relative_position"]
                    ),

                "raw_motion":
                    cosine_sim(
                        class_vectors[a]["raw_motion"],
                        class_vectors[b]["raw_motion"]
                    ),

                "wrist_relative_motion":
                    cosine_sim(
                        class_vectors[a]["relative_motion"],
                        class_vectors[b]["relative_motion"]
                    ),

                "low_sample":
                    (
                        sample_counts[a] < min_samples
                        or
                        sample_counts[b] < min_samples
                    ),
            }

            results.append(
                result
            )

    return results


# ============================================================
# PRINT RANKING
# ============================================================

def print_ranking(
    pair_results,
    key,
    title,
    top_k
):
    """
    Print one ranking.
    """

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"{'Pair':45s}"
        f"{'Similarity':>12s}"
        f"{'Flag':>18s}"
    )

    print("-" * 80)

    ranked = sorted(
        pair_results,
        key=lambda r: r[key],
        reverse=True
    )

    for r in ranked[:top_k]:

        pair = (
            f"{r['a']} <-> {r['b']}"
        )

        flag = (
            "LOW SAMPLE"
            if r["low_sample"]
            else ""
        )

        print(
            f"{pair:45s}"
            f"{r[key]:12.3f}"
            f"{flag:>18s}"
        )


# ============================================================
# COMBINED CONFUSION RANKING
# ============================================================

def add_combined_score(
    pair_results
):
    """
    Calculate a simple combined score.

    We deliberately do NOT train a model here.

    Combined score:

        average(
            raw position,
            wrist-relative position,
            raw motion,
            wrist-relative motion
        )

    This gives a general "overall similarity" ranking.
    """

    for r in pair_results:

        values = [

            r["raw_position"],

            r["wrist_relative_position"],

            r["raw_motion"],

            r["wrist_relative_motion"],

        ]

        r["combined"] = float(
            np.mean(values)
        )

    return pair_results


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze(
    cache_dir: Path,
    conf_threshold: float,
    min_samples: int,
    top_k: int,
    save_json=None
):

    print("=" * 80)
    print("VoxBridge Landmark Similarity Analysis v2")
    print("=" * 80)

    print(
        "Cache:",
        cache_dir
    )

    print(
        "Confidence threshold:",
        conf_threshold
    )

    print(
        "Minimum samples:",
        min_samples
    )

    print("=" * 80)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = load_cache(
        cache_dir
    )

    print(
        f"Loaded {len(data)} classes."
    )

    if len(data) < 2:

        print(
            "Need at least 2 classes."
        )

        return

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    per_video = {}

    sample_counts = {}

    reliable_frame_counts = {}

    # --------------------------------------------------------
    # PROCESS EVERY VIDEO
    # --------------------------------------------------------

    print()
    print("Building representations...")
    print("-" * 80)

    for label, videos in data.items():

        per_video[label] = []

        sample_counts[label] = len(
            videos
        )

        reliable_frame_counts[label] = []

        for x, c, name in videos:

            representation = build_video_representations(
                x,
                c,
                conf_threshold
            )

            representation["video"] = name

            per_video[label].append(
                representation
            )

            reliable_frame_counts[label].append(
                representation[
                    "reliable_frames"
                ]
            )

    # --------------------------------------------------------
    # SAMPLE COUNTS
    # --------------------------------------------------------

    labels = sorted(
        data.keys()
    )

    print()
    print("=" * 80)
    print("CLASS SAMPLE COUNTS")
    print("=" * 80)

    for label in labels:

        n = sample_counts[label]

        if n < min_samples:

            print(
                f"{label:30s}"
                f"{n:5d}"
                f"  <-- LOW SAMPLE COUNT"
            )

        else:

            print(
                f"{label:30s}"
                f"{n:5d}"
            )

    # --------------------------------------------------------
    # CLASS-LEVEL VECTORS
    # --------------------------------------------------------

    class_vectors = {}

    for label in labels:

        videos = per_video[label]

        class_vectors[label] = {

            "raw_position":
                np.mean(
                    [
                        v["raw_position"]
                        for v in videos
                    ],
                    axis=0
                ),

            "relative_position":
                np.mean(
                    [
                        v["relative_position"]
                        for v in videos
                    ],
                    axis=0
                ),

            "raw_motion":
                np.mean(
                    [
                        v["raw_motion"]
                        for v in videos
                    ],
                    axis=0
                ),

            "relative_motion":
                np.mean(
                    [
                        v["relative_motion"]
                        for v in videos
                    ],
                    axis=0
                ),
        }

    # --------------------------------------------------------
    # WITHIN CLASS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("WITHIN-CLASS SIMILARITY")
    print("=" * 100)

    print(
        f"{'Gesture':30s}"
        f"{'Raw':>10s}"
        f"{'Wrist-Rel':>12s}"
        f"{'Motion':>10s}"
        f"{'Rel-Motion':>12s}"
        f"{'N':>6s}"
    )

    print("-" * 100)

    within_class = {}

    for label in labels:

        videos = per_video[label]

        raw_pos = average_pairwise_similarity(
            [
                v["raw_position"]
                for v in videos
            ]
        )

        relative_pos = average_pairwise_similarity(
            [
                v["relative_position"]
                for v in videos
            ]
        )

        raw_motion = average_pairwise_similarity(
            [
                v["raw_motion"]
                for v in videos
            ]
        )

        relative_motion = average_pairwise_similarity(
            [
                v["relative_motion"]
                for v in videos
            ]
        )

        within_class[label] = {

            "raw_position":
                raw_pos,

            "wrist_relative_position":
                relative_pos,

            "raw_motion":
                raw_motion,

            "wrist_relative_motion":
                relative_motion,

            "samples":
                sample_counts[label],

            "mean_reliable_frames":
                float(
                    np.mean(
                        reliable_frame_counts[label]
                    )
                ),
        }

        flag = (
            " <-- LOW SAMPLE"
            if sample_counts[label] < min_samples
            else ""
        )

        print(
            f"{label:30s}"
            f"{raw_pos:10.3f}"
            f"{relative_pos:12.3f}"
            f"{raw_motion:10.3f}"
            f"{relative_motion:12.3f}"
            f"{sample_counts[label]:6d}"
            f"{flag}"
        )

    # --------------------------------------------------------
    # PAIRWISE RESULTS
    # --------------------------------------------------------

    pair_results = make_pair_results(
        labels,
        class_vectors,
        sample_counts,
        min_samples
    )

    pair_results = add_combined_score(
        pair_results
    )

    # --------------------------------------------------------
    # RAW POSITION
    # --------------------------------------------------------

    print_ranking(
        pair_results,
        "raw_position",
        "TOP GESTURE PAIRS - RAW POSITION SIMILARITY",
        top_k
    )

    # --------------------------------------------------------
    # WRIST RELATIVE POSITION
    # --------------------------------------------------------

    print_ranking(
        pair_results,
        "wrist_relative_position",
        "TOP GESTURE PAIRS - WRIST-RELATIVE POSITION SIMILARITY",
        top_k
    )

    # --------------------------------------------------------
    # RAW MOTION
    # --------------------------------------------------------

    print_ranking(
        pair_results,
        "raw_motion",
        "TOP GESTURE PAIRS - RAW MOTION SIMILARITY",
        top_k
    )

    # --------------------------------------------------------
    # WRIST RELATIVE MOTION
    # --------------------------------------------------------

    print_ranking(
        pair_results,
        "wrist_relative_motion",
        "TOP GESTURE PAIRS - WRIST-RELATIVE MOTION SIMILARITY",
        top_k
    )

    # --------------------------------------------------------
    # COMBINED
    # --------------------------------------------------------

    print_ranking(
        pair_results,
        "combined",
        "TOP GESTURE PAIRS - OVERALL COMBINED SIMILARITY",
        top_k
    )

    # --------------------------------------------------------
    # SPECIAL ANALYSIS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("RAW vs WRIST-RELATIVE COMPARISON")
    print("=" * 100)

    print(
        f"{'Pair':45s}"
        f"{'Raw Pos':>10s}"
        f"{'Rel Pos':>10s}"
        f"{'Change':>10s}"
    )

    print("-" * 100)

    # Sort by biggest improvement from raw -> wrist-relative

    comparison = []

    for r in pair_results:

        raw = r["raw_position"]

        relative = r[
            "wrist_relative_position"
        ]

        change = relative - raw

        comparison.append(
            (
                change,
                r
            )
        )

    comparison.sort(
        key=lambda item: item[0],
        reverse=True
    )

    for change, r in comparison[:top_k]:

        pair = (
            f"{r['a']} <-> {r['b']}"
        )

        print(
            f"{pair:45s}"
            f"{r['raw_position']:10.3f}"
            f"{r['wrist_relative_position']:10.3f}"
            f"{change:+10.3f}"
        )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("HOW TO INTERPRET THIS REPORT")
    print("=" * 100)

    print(
        """
RAW POSITION
------------
High similarity means the gestures occupy similar absolute
positions/configurations in the MediaPipe coordinate space.

WRIST-RELATIVE POSITION
-----------------------
High similarity means the actual hand shapes are similar after
removing the hand's absolute location.

This is usually more useful when two gestures can be performed
at different positions in front of the camera.

RAW MOTION
----------
High similarity means the gestures have similar frame-to-frame
landmark movement.

WRIST-RELATIVE MOTION
---------------------
High similarity means the internal hand movement is similar after
removing wrist translation.

IMPORTANT
---------
A high similarity score does NOT prove that the classifier confuses
the two gestures.

It only identifies pairs worth investigating.

The strongest candidates are usually pairs that are:

    high wrist-relative position
    +
    high wrist-relative motion

especially when both classes have enough samples.

LOW-SAMPLE CLASSES
------------------
Any class below --min-samples should be treated cautiously.

For example, a gesture with only 5 or 8 samples can produce a high
similarity score simply because the sample estimate is unstable.

COLLECT MORE DATA BEFORE CHANGING THE MODEL
-------------------------------------------
If a pair remains highly similar after you collect more examples,
THEN it becomes stronger evidence that the gesture definitions/data
are genuinely difficult to separate.
"""
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    if save_json:

        output = {

            "version":
                "analyze_landmark_similarity_v2",

            "cache_dir":
                str(cache_dir),

            "conf_threshold":
                conf_threshold,

            "min_samples":
                min_samples,

            "sample_counts":
                sample_counts,

            "within_class":
                within_class,

            "pairs":
                pair_results,
        }

        with open(
            save_json,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output,
                f,
                indent=2,
                allow_nan=True
            )

        print()
        print(
            f"Saved JSON report to: {save_json}"
        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "VoxBridge Stage 1 v2: "
            "Raw vs wrist-relative vs motion similarity analysis."
        )
    )

    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Path to cache directory."
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum MediaPipe confidence for a frame "
            "to be considered reliable."
        )
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help=(
            "Classes below this number of videos are "
            "flagged as low sample."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help=(
            "Number of top gesture pairs to display "
            "for each ranking."
        )
    )

    parser.add_argument(
        "--save-json",
        default=None,
        help=(
            "Optional JSON output path, e.g. report_v2.json"
        )
    )

    args = parser.parse_args()

    analyze(
        cache_dir=Path(
            args.cache_dir
        ),
        conf_threshold=args.conf_threshold,
        min_samples=args.min_samples,
        top_k=args.top_k,
        save_json=args.save_json)