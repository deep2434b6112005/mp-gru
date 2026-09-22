"""
VoxBridge Production Live Detector
==================================

Pipeline:

    rpicam-vid
        |
        v
    CameraWorker
        |
        v
    LatestFrameBuffer
        |
        v
    MediaPipe LIVE_STREAM
        |
        v
    callback
        |
        v
    LatestResultBuffer
        |
        v
    InferenceWorker
        |
        +--> MP-GRU
        +--> GestureClassifier
        +--> EMA
        +--> Top-2 margin
        +--> Stability voting
        +--> Cooldown
        +--> Sentence buffer
        |
        +--> TTS worker
        +--> Firebase worker

Designed for Raspberry Pi deployment.

IMPORTANT:
    This program is headless.
    Ctrl+C is handled by SIGINT.
    SIGTERM is also handled for clean shutdown.
"""

from __future__ import annotations

import argparse
import atexit
import faulthandler
import json
import os
import queue
import signal
import subprocess
import threading
import time

from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import GestureClassifier


# ============================================================
# THREAD DEADLOCK / BLOCKING DEBUG
# ============================================================

faulthandler.dump_traceback_later(
    10,
    repeat=True,
)

print("[DEBUG] Python thread stack watchdog enabled", flush=True)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODE_STATE_FILE = os.path.join(
    SCRIPT_DIR,
    "mode_state.json",
)

QUEUE_FILE = os.path.join(
    SCRIPT_DIR,
    "firebase_queue.json",
)

DEFAULT_MODEL = os.path.join(
    SCRIPT_DIR,
    "hand_landmarker.task",
)

DEFAULT_CHECKPOINT = os.path.join(
    SCRIPT_DIR,
    "gesture_classifier.pt",
)


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


# ============================================================
# MEDIAPIPE CONFIGURATION
# ============================================================

MAX_HANDS = 2

MP_DETECTION_CONFIDENCE = 0.45
MP_PRESENCE_CONFIDENCE = 0.45
MP_TRACKING_CONFIDENCE = 0.45

# Limit live-stream landmark requests so inference can keep up.
MIN_MP_INTERVAL = 0.25 # no artificial MediaPipe submission throttle; benchmark real throughput


# ============================================================
# CLASSIFIER CONFIGURATION
# ============================================================

CLASSIFIER_CONFIDENCE = 0.40
CLASSIFIER_MARGIN = 0.10

HIGH_CONFIDENCE = 0.5
HIGH_MARGIN = 0.9

LOW_CONFIDENCE = 0.20


# ============================================================
# TEMPORAL DECISION CONFIGURATION
# ============================================================

PROBABILITY_EMA = 0.45

FAST_STABILITY_FRAMES = 2
NORMAL_STABILITY_FRAMES = 3
SLOW_STABILITY_FRAMES = 4

HISTORY_LEN = 6

COOLDOWN_SECONDS = 0.80


# ============================================================
# HAND LOSS / SENTENCE CONFIGURATION
# ============================================================

HAND_LOST_RESET_SECONDS = 0.70

END_OF_SIGN_DELAY = 0.90

# Poll mode_state.json independently of inference-frame count.
MODE_POLL_INTERVAL = 0.75


# ============================================================
# MP-GRU NEUTRAL CONFIGURATION
# ============================================================

NEUTRAL_THRESHOLD = 0.50


# ============================================================
# CUSTOMISATION MODE
# ============================================================

FINGER_STABILITY_FRAMES = 4
FINGER_HISTORY_LEN = 6
FINGER_COOLDOWN = 1.0


DEFAULT_FINGER_PHRASES = {
    "1": "I am hungry",
    "2": "I need water",
    "3": "Please help me",
    "4": "I need my medicine",
    "5": "Take me to hospital",
    "fist": "No",
}


# ============================================================
# GLOBAL SHUTDOWN
# ============================================================

STOP_EVENT = threading.Event()

_SHUTDOWN_LOCK = threading.Lock()
_SHUTDOWN_STARTED = False


def request_shutdown(signum=None, frame=None):
    """
    Process-level shutdown handler.

    Handles:
        Ctrl+C  -> SIGINT
        systemd -> SIGTERM
    """

    global _SHUTDOWN_STARTED

    with _SHUTDOWN_LOCK:

        if _SHUTDOWN_STARTED:
            return

        _SHUTDOWN_STARTED = True

    name = {
        signal.SIGINT: "SIGINT / Ctrl+C",
        signal.SIGTERM: "SIGTERM",
    }.get(signum, "shutdown")

    print(
        f"\n\n>>> Shutdown requested ({name})"
    )

    STOP_EVENT.set()


# Register signal handlers.
signal.signal(
    signal.SIGINT,
    request_shutdown,
)

signal.signal(
    signal.SIGTERM,
    request_shutdown,
)


# ============================================================
# LATEST VALUE BUFFER
# ============================================================

class LatestValue:

    """
    Thread-safe latest-value buffer.

    New data replaces old data.

    This prevents stale frames/results from building
    up in memory.
    """

    def __init__(self):

        self._lock = threading.Lock()

        self._value = None

        self._version = 0

    def put(self, value):

        with self._lock:

            self._value = value

            self._version += 1

            return self._version

    def get(self):

        with self._lock:

            return (
                self._value,
                self._version,
            )


# ============================================================
# DETECTION PACKET
# ============================================================

@dataclass
class DetectionPacket:

    result: object

    timestamp_ms: int

    received_time: float


# ============================================================
# MODE STATE
# ============================================================

def load_mode_state():

    default = {
        "mode": "isl",
        "phrases": dict(
            DEFAULT_FINGER_PHRASES
        ),
    }

    try:

        with open(
            MODE_STATE_FILE,
            "r",
        ) as f:

            data = json.load(f)

        mode = data.get(
            "mode",
            "isl",
        )

        if mode not in (
            "isl",
            "customisation",
        ):

            mode = "isl"

        phrases = dict(
            DEFAULT_FINGER_PHRASES
        )

        user_phrases = data.get(
            "phrases",
            {},
        )

        if isinstance(
            user_phrases,
            dict,
        ):

            phrases.update(
                user_phrases
            )

        return {
            "mode": mode,
            "phrases": phrases,
        }

    except Exception as e:

        print(
            f"[MODE] Using defaults: {e}"
        )

        return default


# ============================================================
# CAMERA STREAM
# ============================================================

class CameraStream:

    """
    Raspberry Pi camera stream.

    Uses rpicam-vid and YUV420.

    IMPORTANT:
        stdout.read(N) is not guaranteed to return N bytes.

    Therefore a persistent byte buffer is used.
    """

    def __init__(
        self,
        width=640,
        height=480,
        fps=30,
    ):

        self.width = width

        self.height = height

        self.fps = fps

        self.frame_size = (
            width
            * height
            * 3
            // 2
        )

        cmd = [
            "rpicam-vid",

            "--nopreview",

            "--timeout",
            "0",

            "--width",
            str(width),

            "--height",
            str(height),

            "--framerate",
            str(fps),

            "--codec",
            "yuv420",

            "--output",
            "-",

            "--flush",

            "--denoise",
            "cdn_off",
        ]

        print()
        print("Camera command:")
        print(" ".join(cmd))

        print(
            f"Camera frame size: "
            f"{self.frame_size} bytes"
        )

        self.proc = None

        self.buffer = bytearray()

        self.frame_count = 0

        try:

            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                bufsize=0,
            )

        except Exception as e:

            print(
                "[CAMERA] Failed to start rpicam-vid:",
                e,
            )

            self.proc = None

    def isOpened(self):

        return (
            self.proc is not None
            and self.proc.poll() is None
        )

    def _read_complete_frame(self):

        if not self.isOpened():

            return None

        while not STOP_EVENT.is_set():

            if (
                len(self.buffer)
                >= self.frame_size
            ):

                frame = bytes(
                    self.buffer[
                        :self.frame_size
                    ]
                )

                del self.buffer[
                    :self.frame_size
                ]

                return frame

            try:

                chunk = self.proc.stdout.read(
                    max(
                        4096,
                        self.frame_size
                        - len(self.buffer),
                    )
                )

            except Exception as e:

                if not STOP_EVENT.is_set():

                    print(
                        "[CAMERA] Read error:",
                        e,
                    )

                return None

            if not chunk:

                if (
                    self.proc.poll()
                    is not None
                ):

                    return None

                time.sleep(
                    0.001
                )

                continue

            self.buffer.extend(
                chunk
            )

        return None

    def read(self):

        raw = (
            self._read_complete_frame()
        )

        if raw is None:

            return False, None

        try:

            yuv = np.frombuffer(
                raw,
                dtype=np.uint8,
            ).reshape(
                (
                    self.height * 3 // 2,
                    self.width,
                )
            )

            bgr = cv2.cvtColor(
                yuv,
                cv2.COLOR_YUV2BGR_I420,
            )

            self.frame_count += 1

            return True, bgr

        except Exception as e:

            print(
                "[CAMERA] Conversion error:",
                e,
            )

            return False, None

    def release(self):

        proc = self.proc

        self.proc = None

        if proc is None:

            return

        try:

            if proc.poll() is None:

                proc.terminate()

                try:

                    proc.wait(
                        timeout=2
                    )

                except subprocess.TimeoutExpired:

                    print(
                        "[CAMERA] Killing rpicam-vid..."
                    )

                    proc.kill()

                    try:

                        proc.wait(
                            timeout=1
                        )

                    except Exception:
                        pass

        except Exception:

            pass

        self.buffer.clear()


# ============================================================
# CAMERA WORKER
# ============================================================

class CameraWorker:

    def __init__(
        self,
        source,
        frame_buffer,
    ):

        self.source = source

        self.frame_buffer = (
            frame_buffer
        )

        self.thread = None

        self.cap = None

        self.frames = 0

        self.last_status = (
            time.monotonic()
        )
        self.last_status_frames = 0
        self.last_status_time = self.last_status

    def start(self):

        self.thread = threading.Thread(
            target=self.run,
            name="camera",
            daemon=True,
        )

        self.thread.start()

    def stop(self):

        if self.cap is None:
            return

        try:
            self.cap.release()
        except Exception:
            pass

        self.cap = None

    def run(self):

        try:

            if str(
                self.source
            ).isdigit():

                self.cap = CameraStream(
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT,
                    CAMERA_FPS,
                )

            else:

                self.cap = cv2.VideoCapture(
                    self.source
                )

                self.cap.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    CAMERA_WIDTH,
                )

                self.cap.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    CAMERA_HEIGHT,
                )

            if not self.cap.isOpened():

                print(
                    "ERROR: Camera failed to open."
                )

                STOP_EVENT.set()

                return

            print(
                "✓ Camera thread started"
            )

            while not STOP_EVENT.is_set():

                ok, frame = (
                    self.cap.read()
                )

                if not ok:

                    if not STOP_EVENT.is_set():

                        print(
                            "[CAMERA] Frame failure."
                        )

                        STOP_EVENT.set()

                    break

                if frame is None:

                    continue

                # Mirror camera.
                frame = cv2.flip(
                    frame,
                    1,
                )

                self.frame_buffer.put(
                    frame
                )

                self.frames += 1

                now = time.monotonic()

                if (
                    now
                    - self.last_status
                    >= 5.0
                ):

                    elapsed = now - self.last_status_time
                    frame_delta = (
                        self.frames
                        - self.last_status_frames
                    )
                    camera_fps = (
                        frame_delta / elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    self.last_status = now
                    self.last_status_time = now
                    self.last_status_frames = self.frames

                    print(
                        f"[CAMERA] fps={camera_fps:.1f} "
                        f"total={self.frames}"
                    )

        except Exception as e:

            print(
                "[CAMERA] Worker exception:",
                repr(e),
            )

            STOP_EVENT.set()

        finally:

            self.stop()


# ============================================================
# MEDIAPIPE WORKER
# ============================================================

class MediaPipeWorker:

    def __init__(
        self,
        model_path,
        result_buffer,
        debug=False,
    ):

        self.model_path = model_path

        self.result_buffer = (
            result_buffer
        )

        self.debug = debug

        self.landmarker = None

        self.mp = None

        self.last_submitted_timestamp = -1

        self.callback_count = 0

        self.error_count = 0

        self.last_callback_time = (
            time.monotonic()
        )
        self.last_callback_status = (
            time.monotonic()
        )
        self.last_callback_count = 0

        self.lock = threading.Lock()

    def build(self):

        import mediapipe as mp

        from mediapipe.tasks import python as mp_python

        from mediapipe.tasks.python import vision

        self.mp = mp

        if not os.path.exists(
            self.model_path
        ):

            raise FileNotFoundError(
                f"MediaPipe model not found: "
                f"{self.model_path}"
            )

        base_options = (
            mp_python.BaseOptions(
                model_asset_path=self.model_path
            )
        )

        options = (
            vision.HandLandmarkerOptions(
                base_options=base_options,

                running_mode=(
                    vision.RunningMode.LIVE_STREAM
                ),

                num_hands=MAX_HANDS,

                min_hand_detection_confidence=(
                    MP_DETECTION_CONFIDENCE
                ),

                min_hand_presence_confidence=(
                    MP_PRESENCE_CONFIDENCE
                ),

                min_tracking_confidence=(
                    MP_TRACKING_CONFIDENCE
                ),

                result_callback=self.callback,
            )
        )

        self.landmarker = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

        print(
            "✓ MediaPipe LIVE_STREAM ready"
        )

    def callback(
        self,
        result,
        output_image,
        timestamp_ms,
    ):

        try:

            try:
                hand_count = len(result.hand_landmarks)
            except Exception:
                hand_count = 0

            self.callback_count += 1

            now = time.monotonic()
            self.last_callback_time = now

            if now - self.last_callback_status >= 1.0:
                elapsed = (
                    now
                    - self.last_callback_status
                )
                callback_delta = (
                    self.callback_count
                    - self.last_callback_count
                )
                callback_fps = (
                    callback_delta / elapsed
                )

                print(
                    f"[MP] callback_fps={callback_fps:.1f} "
                    f"total={self.callback_count} "
                    f"hands={hand_count}"
                )

                self.last_callback_status = now
                self.last_callback_count = self.callback_count

            if self.debug and self.callback_count % 10 == 0:

                print(
                    f"[MP] callbacks={self.callback_count} "
                    f"| hands={hand_count}"
                )

                if self.debug and hand_count > 0:

                    try:
                        handedness = []

                        for categories in result.handedness:

                            if categories:
                                category = categories[0]
                                handedness.append(
                                    f"{category.category_name}="
                                    f"{category.score:.3f}"
                                )

                        print(
                            "[MP] handedness: "
                            + " | ".join(handedness)
                        )

                    except Exception as e:

                        print("[MP] handedness read error:", repr(e))

            packet = DetectionPacket(
                result=result,
                timestamp_ms=int(
                    timestamp_ms
                ),
                received_time=(
                    time.monotonic()
                ),
            )

            self.result_buffer.put(
                packet
            )

        except Exception as e:

            self.error_count += 1

            print(
                "[MEDIAPIPE CALLBACK] error:",
                repr(e),
            )

    def submit_frame(
        self,
        frame,
        timestamp_ms,
    ):

        if self.landmarker is None:

            return

        if STOP_EVENT.is_set():

            return

        timestamp_ms = int(
            timestamp_ms
        )

        if (
            timestamp_ms
            <=
            self.last_submitted_timestamp
        ):

            return

        self.last_submitted_timestamp = (
            timestamp_ms
        )

        try:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = self.mp.Image(
                image_format=(
                    self.mp.ImageFormat.SRGB
                ),
                data=rgb,
            )

            self.landmarker.detect_async(
                mp_image,
                timestamp_ms,
            )

        except Exception as e:

            if not STOP_EVENT.is_set():

                self.error_count += 1

                print(
                    "[MEDIAPIPE SUBMIT] error:",
                    repr(e),
                )

    def close(self):

        with self.lock:

            landmarker = (
                self.landmarker
            )

            self.landmarker = None

        if landmarker is None:

            return

        try:

            landmarker.close()

        except Exception as e:

            print(
                "[MEDIAPIPE] close error:",
                repr(e),
            )


# ============================================================
# LANDMARK CONFIGURATION
# ============================================================

NUM_LANDMARKS = 21
COORDS = 3
SLOT_SIZE = (
    NUM_LANDMARKS * COORDS
)

INPUT_SIZE = 126


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def landmarks_to_vector(result):

    x = np.zeros(
        INPUT_SIZE,
        dtype=np.float32,
    )

    confidence = 0.0

    if (
        result is None
        or not getattr(
            result,
            "hand_landmarks",
            None,
        )
    ):

        return (
            x,
            confidence,
            None,
        )

    confidences = []

    first_landmarks = None

    for hand_idx, hand_lms in enumerate(
        result.hand_landmarks
    ):

        if first_landmarks is None:

            first_landmarks = hand_lms

        try:

            label = (
                result
                .handedness[
                    hand_idx
                ][0]
                .category_name
            )

            score = (
                result
                .handedness[
                    hand_idx
                ][0]
                .score
            )

        except Exception:

            label = (
                "Left"
                if hand_idx == 0
                else "Right"
            )

            score = 1.0

        slot = (
            0
            if label.lower() == "left"
            else 1
        )

        offset = (
            slot
            * SLOT_SIZE
        )

        for i, lm in enumerate(
            hand_lms
        ):

            if i >= NUM_LANDMARKS:

                break

            base = offset + i * COORDS

            x[base] = float(lm.x)
            x[base + 1] = float(lm.y)
            x[base + 2] = float(lm.z)

        confidences.append(
            float(score)
        )

    if confidences:

        confidence = max(
            confidences
        )

    return (
        x,
        confidence,
        first_landmarks,
    )


# ============================================================
# FINGER COUNT
# ============================================================

def distance(a, b):

    return (
        (a.x - b.x) ** 2
        +
        (a.y - b.y) ** 2
    ) ** 0.5


def count_fingers(landmarks):

    if landmarks is None:

        return 0

    if len(landmarks) < 21:

        return 0

    wrist = landmarks[0]

    count = 0

    pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18),
    ]

    for tip, pip in pairs:

        if (
            distance(
                wrist,
                landmarks[tip],
            )
            >
            distance(
                wrist,
                landmarks[pip],
            )
            * 1.10
        ):

            count += 1

    if (
        distance(
            landmarks[4],
            landmarks[17],
        )
        >
        distance(
            landmarks[2],
            landmarks[17],
        )
        * 1.05
    ):

        count += 1

    return count


# ============================================================
# FINGER DETECTOR
# ============================================================

class FingerDetector:

    def __init__(
        self,
        stability=4,
        history_len=6,
        cooldown=1.0,
    ):

        self.history = deque(
            maxlen=history_len
        )

        self.stability = (
            stability
        )

        self.cooldown = (
            cooldown
        )

        self.last_fire = 0.0

        # A recognised gesture must be released before re-triggering.
        self.armed = True

    def reset(self):

        self.history.clear()

        self.last_fire = 0.0

        self.armed = True

    def update(
        self,
        landmarks,
    ):

        if landmarks is None:

            self.reset()

            return None

        if not self.armed:

            return None

        value = count_fingers(
            landmarks
        )

        self.history.append(
            value
        )

        if (
            len(self.history)
            <
            self.stability
        ):

            return None

        common, freq = (
            Counter(
                self.history
            ).most_common(1)[0]
        )

        if freq < self.stability:

            return None

        now = time.monotonic()

        if (
            now - self.last_fire
            <
            self.cooldown
        ):

            return None

        self.last_fire = now

        self.history.clear()

        self.armed = False

        return common


# ============================================================
# GESTURE DECISION ENGINE
# ============================================================

class GestureDecisionEngine:

    def __init__(
        self,
        idx_to_label,
        conf_threshold=CLASSIFIER_CONFIDENCE,
        margin_threshold=CLASSIFIER_MARGIN,
        history_len=HISTORY_LEN,
        cooldown=COOLDOWN_SECONDS,
        debug=False,
    ):

        self.idx_to_label = (
            idx_to_label
        )

        self.conf_threshold = (
            conf_threshold
        )

        self.margin_threshold = (
            margin_threshold
        )

        self.history_len = (
            history_len
        )

        self.cooldown = (
            cooldown
        )

        self.label_history = deque(
            maxlen=history_len
        )

        self.prob_ema = None

        self.last_fire = 0.0

        self.last_label = ""

        self.last_conf = 0.0

        self.debug = debug

    def reset(self):

        self.label_history.clear()

        self.prob_ema = None

        self.last_label = ""

        self.last_conf = 0.0

    def update(
        self,
        probs,
    ):

        probs = probs.detach()

        if probs.numel() == 0:

            self.reset()

            if self.debug:
                print("[FILTER] REJECT: empty probability vector")

            return None

        # ----------------------------------------------------
        # EMA
        # ----------------------------------------------------

        if self.prob_ema is None:

            self.prob_ema = probs.clone()

        else:

            self.prob_ema = (
                PROBABILITY_EMA
                * self.prob_ema
                +
                (1.0 - PROBABILITY_EMA)
                * probs
            )

        smooth = self.prob_ema

        # ----------------------------------------------------
        # Top 3 diagnostics
        # ----------------------------------------------------

        k = min(
            3,
            smooth.numel(),
        )

        top = torch.topk(
            smooth,
            k=k,
        )

        prediction_text = []

        for i in range(k):

            idx = int(top.indices[i])
            probability = float(top.values[i])
            label = self.idx_to_label.get(idx, f"class_{idx}")
            prediction_text.append(f"{label}={probability:.2f}")

        if self.debug:
            print("[PRED] " + " | ".join(prediction_text))

        idx1 = int(
            top.indices[0]
        )

        conf1 = float(
            top.values[0]
        )

        if k >= 2:

            conf2 = float(
                top.values[1]
            )

        else:

            conf2 = 0.0

        margin = (
            conf1 - conf2
        )

        if self.debug:

            label1 = self.idx_to_label.get(
                idx1,
                f"UNKNOWN_{idx1}",
            )

            print(
                f"[DEBUG][DECISION] "
                f"top1={label1} "
                f"conf={conf1:.3f} "
                f"top2={conf2:.3f} "
                f"margin={margin:.3f}"
            )

        label1 = self.idx_to_label.get(idx1, f"class_{idx1}")

        # ----------------------------------------------------
        # Adaptive stability
        # ----------------------------------------------------

        if (
            conf1 >= HIGH_CONFIDENCE
            and margin >= HIGH_MARGIN
        ):

            required = (
                FAST_STABILITY_FRAMES
            )

        elif (
            conf1 >= self.conf_threshold
            and margin >= self.margin_threshold
        ):

            required = (
                NORMAL_STABILITY_FRAMES
            )

        elif conf1 >= LOW_CONFIDENCE:

            required = (
                SLOW_STABILITY_FRAMES
            )

        else:

            if self.debug:
                print(
                    f"[DEBUG][REJECT] "
                    f"confidence too low: "
                    f"{conf1:.3f} < {LOW_CONFIDENCE:.3f}"
                )

            if self.debug:
                print(
                    f"[FILTER] REJECT: confidence "
                    f"{conf1:.2f} < {LOW_CONFIDENCE:.2f}"
                )

            self.label_history.clear()

            return None

        if (
            conf1 >= self.conf_threshold
            and margin < self.margin_threshold
        ):

            if self.debug:
                print(
                    f"[FILTER] REJECT: margin "
                    f"{margin:.2f} < {self.margin_threshold:.2f}"
                )

            # Ambiguous high-confidence result: do not let it
            # contaminate the stability history.
            self.label_history.clear()

            return None

        # ----------------------------------------------------
        # Add prediction
        # ----------------------------------------------------

        self.label_history.append(
            idx1
        )

        if self.debug:
            print(
                f"[FILTER] candidate={label1} "
                f"conf={conf1:.2f} margin={margin:.2f} "
                f"stability={len(self.label_history)}/{required}"
            )

        # We require the current candidate to dominate
        # the recent history.
        if (
            len(self.label_history)
            <
            required
        ):

            if self.debug:
                print(
                    f"[DEBUG][STABILITY] "
                    f"candidate={idx1} "
                    f"history={list(self.label_history)} "
                    f"required={required}"
                )

            return None

        recent = list(
            self.label_history
        )[-required:]

        common_idx, freq = (
            Counter(
                recent
            ).most_common(1)[0]
        )

        if freq < required:

            if self.debug:
                print(
                    f"[DEBUG][VOTE REJECT] "
                    f"history={recent} "
                    f"winner={common_idx} "
                    f"frequency={freq}/{required}"
                )

            if self.debug:
                print(f"[FILTER] REJECT: stability {freq}/{required}")

            return None

        # IMPORTANT:
        # The current prediction must also agree with
        # the voted class. This prevents an old class
        # from being accepted after the hand changes.
        if common_idx != idx1:

            if self.debug:
                print(
                    f"[DEBUG][CONFLICT] "
                    f"current={idx1} "
                    f"voted={common_idx} "
                    f"history={recent}"
                )

            if self.debug:
                print(
                    "[FILTER] REJECT: current prediction "
                    "does not match stability vote"
                )

            return None

        now = time.monotonic()

        if (
            now - self.last_fire
            <
            self.cooldown
        ):

            if self.debug:
                print(
                    f"[DEBUG][COOLDOWN] "
                    f"remaining="
                    f"{self.cooldown - (now - self.last_fire):.2f}s"
                )

            if self.debug:
                print(
                    f"[FILTER] REJECT: cooldown "
                    f"{self.cooldown - (now - self.last_fire):.2f}s"
                )

            return None

        accepted_label = (
            self.idx_to_label.get(
                common_idx,
                f"class_{common_idx}",
            )
        )

        self.last_fire = now

        self.last_label = (
            accepted_label
        )

        self.last_conf = (
            conf1
        )

        self.label_history.clear()

        print(
            f"[FILTER] ACCEPT: {accepted_label} "
            f"conf={conf1:.2f} margin={margin:.2f}"
        )

        return {
            "label": accepted_label,
            "confidence": conf1,
            "margin": margin,
        }


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(
        "Loading checkpoint:",
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    label_to_idx = checkpoint[
        "label_to_idx"
    ]

    idx_to_label = {
        int(v): k
        for k, v in label_to_idx.items()
    }

    hidden_size = checkpoint.get(
        "hidden_size",
        64,
    )

    embedding_size = checkpoint.get(
        "embedding_size",
        32,
    )

    num_classes = len(
        label_to_idx
    )

    print(
        "Classes:",
        num_classes,
    )

    print(
        "Hidden:",
        hidden_size,
    )

    print(
        "Embedding:",
        embedding_size,
    )

    model = GestureClassifier(
        num_classes=num_classes,
        hidden_size=hidden_size,
        embedding_size=embedding_size,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.eval()

    print(
        "✓ Model loaded successfully"
    )

    return (
        model,
        idx_to_label,
    )


# ============================================================
# SAFE SERIAL TTS WORKER
# ============================================================

class TTSWorker:

    def __init__(
        self,
        piper_bin,
        piper_model,
        audio_device,
        sample_rate,
    ):

        # IMPORTANT:
        # Only ONE pending speech request is allowed.
        # This prevents old gestures from building a queue.
        self.queue = queue.Queue(maxsize=1)

        self.piper_bin = piper_bin
        self.piper_model = piper_model
        self.audio_device = audio_device
        self.sample_rate = sample_rate

        self.thread = None

        # Only the TTS worker accesses these during normal operation.
        self.piper_proc = None
        self.aplay_proc = None

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.pregenerated_wav_dir = os.path.join(
            self.script_dir, "tts_audio"
        )
        self.cache_wav_dir = os.path.join(
            self.pregenerated_wav_dir, "cache"
        )

        try:
            os.makedirs(self.cache_wav_dir, exist_ok=True)
        except Exception as e:
            print("[TTS] WAV directory creation error:", repr(e))

        self.phrase_files = {
            self.normalize_text("I am hungry"): "hungry.wav",
            self.normalize_text("I need water"): "water.wav",
            self.normalize_text("I need some water"): "water.wav",
            self.normalize_text("Please help me"): "help.wav",
            self.normalize_text("I need my medicine"): "medicine.wav",
            self.normalize_text("Take me to hospital"): "hospital.wav",
            self.normalize_text(
                "Please take me to the hospital"
            ): "hospital.wav",
            self.normalize_text("No"): "no.wav",
            self.normalize_text(
                "With VoxBridge today we translate gestures. "
                "Tomorrow we transform lives"
            ): "tagline.wav",
        }

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self):

        self.thread = threading.Thread(
            target=self.run,
            name="tts",
            daemon=True,
        )

        self.thread.start()

    # --------------------------------------------------------
    # ADD SPEECH REQUEST
    # --------------------------------------------------------

    @staticmethod
    def normalize_text(text):

        return " ".join(
            str(text or "").strip().lower().split()
        ).rstrip(".!?")

    @staticmethod
    def safe_filename(text):

        chars = [
            ch if ch.isalnum() else "_"
            for ch in TTSWorker.normalize_text(text).lower()
            if ch.isalnum() or ch in (" ", "-", "_")
        ]
        filename = "".join(chars)

        while "__" in filename:
            filename = filename.replace("__", "_")

        return (filename.strip("_") or "speech")[:100]

    def find_pregenerated_wav(self, text):

        mapped = self.phrase_files.get(
            self.normalize_text(text)
        )

        if mapped:
            path = os.path.join(self.pregenerated_wav_dir, mapped)
            if os.path.isfile(path):
                return path

        return None

    def find_cached_wav(self, text):

        path = os.path.join(
            self.cache_wav_dir,
            self.safe_filename(text) + ".wav",
        )
        return path if os.path.isfile(path) else None

    def speak(self, text):

        text = self.normalize_text(text)

        if not text:
            return

        # Do not accept new work during shutdown.
        if STOP_EVENT.is_set():
            return

        try:

            self.queue.put_nowait(text)

        except queue.Full:

            # For VoxBridge demo:
            # If one phrase is already waiting, don't allow
            # another phrase to build up.
            print(
                "[TTS] Busy; dropping new speech request:",
                repr(text),
            )

    # --------------------------------------------------------
    # WORKER LOOP
    # --------------------------------------------------------

    def run(self):

        if not os.path.exists(self.piper_bin):

            print(
                "[TTS] WARNING: Piper not found:",
                self.piper_bin,
            )

            print("[TTS] WAV-only mode will be used.")

        if not os.path.exists(self.piper_model):

            print(
                "[TTS] WARNING: Piper model not found:",
                self.piper_model,
            )

            print("[TTS] WAV-only mode will be used.")

        print("✓ TTS worker started")

        while not STOP_EVENT.is_set():

            try:

                text = self.queue.get(
                    timeout=0.2
                )

            except queue.Empty:

                continue

            if not text:
                continue

            try:

                self.speak_once(text)

            except Exception as e:

                print(
                    "[TTS] Error:",
                    repr(e),
                )

    # --------------------------------------------------------
    # SPEAK ONE PHRASE
    # --------------------------------------------------------

    def speak_once_diagnostic(self, text):

        if STOP_EVENT.is_set():
            return

        piper = None
        aplay = None

        try:

            print(
                "[TTS] Speaking:",
                repr(text),
            )

            # ------------------------------------------------
            # TEMPORARY DIAGNOSTIC:
            # Generate a WAV file before playback rather than streaming
            # raw Piper output through an aplay pipe.
            # ------------------------------------------------

            start = time.monotonic()

            wav_file = "/tmp/voxbridge_tts.wav"

            piper = subprocess.Popen(
                [
                    self.piper_bin,
                    "--model",
                    self.piper_model,
                    "--output_file",
                    wav_file,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.piper_proc = piper

            # ------------------------------------------------
            # Send text to Piper
            # ------------------------------------------------

            try:

                piper.stdin.write(
                    text + "\n"
                )

                piper.stdin.close()

            except BrokenPipeError:

                print(
                    "[TTS] Piper pipe closed early."
                )

                return

            # ------------------------------------------------
            # Wait for Piper generation
            # ------------------------------------------------

            return_code = piper.wait(timeout=15)

            piper_ms = (
                time.monotonic() - start
            ) * 1000.0

            print(
                f"[TTS DEBUG] Piper finished: "
                f"{piper_ms:.0f} ms "
                f"returncode={return_code}"
            )

            if return_code != 0:

                try:
                    error = piper.stderr.read()
                except Exception:
                    error = ""

                print(
                    "[TTS] Piper error:",
                    error,
                )

                return

            if not os.path.exists(wav_file):

                print(
                    "[TTS] ERROR: WAV file was not created."
                )

                return

            wav_size = os.path.getsize(wav_file)

            print(
                f"[TTS DEBUG] WAV created: "
                f"{wav_size} bytes"
            )

            # ------------------------------------------------
            # Play the generated WAV file
            # ------------------------------------------------

            audio_start = time.monotonic()

            aplay = subprocess.Popen(
                [
                    "aplay",
                    "-D",
                    self.audio_device,
                    wav_file,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            self.aplay_proc = aplay

            return_code = aplay.wait(timeout=20)

            audio_ms = (
                time.monotonic() - audio_start
            ) * 1000.0

            print(
                f"[TTS DEBUG] aplay finished: "
                f"{audio_ms:.0f} ms "
                f"returncode={return_code}"
            )

            if return_code != 0:

                try:
                    error = aplay.stderr.read()
                except Exception:
                    error = ""

                print(
                    "[TTS] aplay error:",
                    error,
                )

            total_ms = (
                time.monotonic() - start
            ) * 1000.0

            print(
                f"[TTS DEBUG] TOTAL TTS TIME: "
                f"{total_ms:.0f} ms"
            )

        except subprocess.TimeoutExpired:

            print(
                "[TTS] Piper/aplay timeout."
            )

        except Exception as e:

            print(
                "[TTS] Error:",
                repr(e),
            )

        finally:

            # ------------------------------------------------
            # ALWAYS clean up both processes
            # ------------------------------------------------

            for proc, name in (
                (piper, "Piper"),
                (aplay, "aplay"),
            ):

                if proc is None:
                    continue

                try:

                    if proc.poll() is None:

                        proc.terminate()

                        try:

                            proc.wait(
                                timeout=1
                            )

                        except subprocess.TimeoutExpired:

                            proc.kill()

                except Exception as e:

                    print(
                        f"[TTS] {name} cleanup error:",
                        repr(e),
                    )

            # Close remaining pipes.

            try:

                if piper is not None and piper.stdin:

                    piper.stdin.close()

            except Exception:

                pass

            try:

                if piper is not None and piper.stderr:

                    piper.stderr.close()

            except Exception:

                pass

            try:

                if aplay is not None and aplay.stderr:

                    aplay.stderr.close()

            except Exception:

                pass

            self.piper_proc = None
            self.aplay_proc = None

            print("[TTS] Speech finished.")

    def generate_piper_wav(self, text):

        if not (
            os.path.isfile(self.piper_bin)
            and os.path.isfile(self.piper_model)
        ):
            print("[TTS] Piper unavailable.")
            return None

        final_path = os.path.join(
            self.cache_wav_dir,
            self.safe_filename(text) + ".wav",
        )
        temp_path = final_path + ".tmp.wav"

        if os.path.isfile(final_path):
            return final_path

        piper = None
        start = time.monotonic()

        try:
            print("[TTS] Piper generating:", repr(text))
            piper = subprocess.Popen(
                [
                    self.piper_bin,
                    "--model",
                    self.piper_model,
                    "--output_file",
                    temp_path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.piper_proc = piper
            piper.communicate(input=text + "\n", timeout=20)

            if piper.returncode != 0:
                print("[TTS] Piper failed:", piper.returncode)
                return None

            if not os.path.isfile(temp_path) or os.path.getsize(temp_path) <= 44:
                print("[TTS] Piper did not create a valid WAV.")
                return None

            os.replace(temp_path, final_path)
            print(
                f"[TTS] Piper finished: "
                f"{(time.monotonic() - start) * 1000.0:.0f} ms"
            )
            return final_path

        except subprocess.TimeoutExpired:
            print("[TTS] Piper timeout.")
            return None
        except Exception as e:
            print("[TTS] Piper generation error:", repr(e))
            return None
        finally:
            if piper is not None and piper.poll() is None:
                try:
                    piper.kill()
                    piper.wait(timeout=1)
                except Exception:
                    pass
            self.piper_proc = None
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    def play_wav(self, wav_path):

        if not os.path.isfile(wav_path) or STOP_EVENT.is_set():
            return

        aplay = None
        start = time.monotonic()

        try:
            print("[TTS] Playing:", os.path.basename(wav_path))
            aplay = subprocess.Popen(
                ["aplay", "-D", self.audio_device, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self.aplay_proc = aplay
            aplay.wait(timeout=30)

            if aplay.returncode != 0:
                print("[TTS] aplay failed:", aplay.returncode)
            else:
                print(
                    f"[TTS] Playback finished: "
                    f"{(time.monotonic() - start) * 1000.0:.0f} ms"
                )
        except subprocess.TimeoutExpired:
            print("[TTS] aplay timeout.")
        except Exception as e:
            print("[TTS] Playback error:", repr(e))
        finally:
            if aplay is not None and aplay.poll() is None:
                try:
                    aplay.kill()
                except Exception:
                    pass
            self.aplay_proc = None

    def speak_once(self, text):

        text = self.normalize_text(text)
        if not text or STOP_EVENT.is_set():
            return

        start = time.monotonic()
        wav_path = self.find_pregenerated_wav(text)

        if wav_path:
            print("[TTS] WAV:", wav_path)
        else:
            wav_path = self.find_cached_wav(text)
            if wav_path:
                print("[TTS] Cached WAV:", wav_path)
            else:
                wav_path = self.generate_piper_wav(text)

        if wav_path is None:
            print("[TTS] ERROR: No audio available for:", repr(text))
            return

        self.play_wav(wav_path)
        print(
            f"[TTS] Total time: "
            f"{(time.monotonic() - start) * 1000.0:.0f} ms"
        )

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop(self):

        print("[TTS] Stopping...")

        # STOP_EVENT should already be set by main shutdown.

        for proc in (
            self.piper_proc,
            self.aplay_proc,
        ):

            if proc is None:
                continue

            try:

                if proc.poll() is None:

                    proc.terminate()

            except Exception:

                pass

        if self.thread is not None:

            self.thread.join(
                timeout=2
            )

        # Final cleanup if something survived.

        for proc in (
            self.piper_proc,
            self.aplay_proc,
        ):

            if proc is None:
                continue

            try:

                if proc.poll() is None:

                    proc.kill()

            except Exception:

                pass

        self.piper_proc = None
        self.aplay_proc = None

# ============================================================
# FIREBASE WORKER
# ============================================================

class FirebaseWorker:

    def __init__(self):

        self.queue = queue.Queue(
            maxsize=20
        )

        self.thread = threading.Thread(
            target=self.run,
            name="firebase-queue",
            daemon=True,
        )

    def start(self):

        self.thread.start()

        print(
            "✓ Firebase worker started"
        )

    def send(self, sentence):

        if not sentence:

            return

        try:

            self.queue.put_nowait(
                sentence
            )

        except queue.Full:

            print(
                "[FIREBASE] Queue full."
            )

    def run(self):

        while not STOP_EVENT.is_set():

            try:

                sentence = (
                    self.queue.get(
                        timeout=0.2
                    )
                )

            except queue.Empty:

                continue

            try:

                tmp = (
                    QUEUE_FILE
                    + ".tmp"
                )

                with open(
                    tmp,
                    "w",
                ) as f:

                    json.dump(
                        {
                            "sentence":
                                sentence,
                        },
                        f,
                    )

                os.replace(
                    tmp,
                    QUEUE_FILE,
                )

            except Exception as e:

                print(
                    "[FIREBASE] Queue error:",
                    repr(e),
                )


class FirestoreWorkerProcess:
    """
    Runs the real firebase_worker.py as a child process.

    This worker watches Firestore settings and gesture phrases, updates
    mode_state.json, and uploads queued sentences.
    """

    def __init__(self):

        self.proc = None

    def start(self):

        script_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        python_bin = os.path.join(
            script_dir,
            ".venv_firebase",
            "bin",
            "python",
        )

        worker_script = os.path.join(
            script_dir,
            "firebase_worker.py",
        )

        if not os.path.exists(python_bin):
            raise FileNotFoundError(
                f"Firebase Python not found: {python_bin}"
            )

        if not os.path.exists(worker_script):
            raise FileNotFoundError(
                f"firebase_worker.py not found: {worker_script}"
            )

        print(
            "[FIREBASE] Starting Firestore worker..."
        )

        self.proc = subprocess.Popen(
            [
                python_bin,
                worker_script,
            ],
            cwd=script_dir,
        )

        print(
            f"✓ Firestore worker started "
            f"(PID {self.proc.pid})"
        )

    def stop(self):

        if self.proc is None:
            return

        if self.proc.poll() is not None:
            return

        print(
            "[FIREBASE] Stopping Firestore worker..."
        )

        try:
            self.proc.terminate()

            self.proc.wait(
                timeout=3
            )

        except subprocess.TimeoutExpired:

            print(
                "[FIREBASE] Worker did not stop; "
                "forcing termination."
            )

            try:
                self.proc.kill()
                self.proc.wait(
                    timeout=2
                )
            except Exception:
                pass

        except Exception as e:

            print(
                "[FIREBASE] Stop error:",
                repr(e),
            )


# ============================================================
# INFERENCE ENGINE
# ============================================================

class InferenceEngine:

    def __init__(
        self,
        model,
        idx_to_label,
        device,
        result_buffer,
        tts_worker,
        firebase_worker,
        debug=False,
    ):

        self.model = model

        self.idx_to_label = (
            idx_to_label
        )

        self.device = device

        self.result_buffer = (
            result_buffer
        )

        self.tts = tts_worker

        self.firebase = (
            firebase_worker
        )

        self.debug = debug
        self.debug_counter = 0
        self.debug_last_print = 0.0
        self.debug_interval = 0.25

        self.state = (
            self.model
            .mpgru
            .cell
            .init_state(
                batch_size=1,
                device=device,
            )
        )

        self.decision = (
            GestureDecisionEngine(
                idx_to_label,
                debug=self.debug,
            )
        )

        self.finger_detector = (
            FingerDetector(
                stability=(
                    FINGER_STABILITY_FRAMES
                ),
                history_len=(
                    FINGER_HISTORY_LEN
                ),
                cooldown=(
                    FINGER_COOLDOWN
                ),
            )
        )

        self.current_mode = "isl"

        self.mode_state = (
            load_mode_state()
        )

        self.sentence = []

        self.last_added_word = ""

        self.last_hand_seen = (
            time.monotonic()
        )

        self.last_result_timestamp = -1

        self.frame_counter = 0

        # Debug / diagnostic counters.
        self.hand_frames = 0
        self.no_hand_frames = 0
        self.prediction_frames = 0
        self.accepted_gestures = 0
        self.last_debug_print = time.monotonic()
        self.last_hand_state = None
        self.hand_loss_reset_done = False

        self.last_status = (
            time.monotonic()
        )

        self.last_mode_poll = 0
        self.last_mode_poll_time = time.monotonic()

        self.thread = None

        self.running = False

    def start(self):

        self.thread = threading.Thread(
            target=self.run,
            name="inference",
            daemon=True,
        )

        self.thread.start()

    def reset_state(self):

        try:

            self.state = (
                self.model
                .mpgru
                .cell
                .init_state(
                    batch_size=1,
                    device=self.device,
                )
            )

            print("[GRU] State reset.")

        except Exception as e:

            print(
                "[INFERENCE] State reset error:",
                repr(e),
            )

        self.decision.reset()

        self.finger_detector.reset()

        self.hand_loss_reset_done = True

    def switch_mode(
        self,
        new_mode,
    ):

        if new_mode == self.current_mode:

            return

        print(
            f"\n>>> MODE SWITCH: "
            f"{self.current_mode} "
            f"-> {new_mode}"
        )

        self.current_mode = (
            new_mode
        )

        self.reset_state()

        self.sentence.clear()

        self.last_added_word = ""

        self.last_hand_seen = (
            time.monotonic()
        )

    def process_isl(
        self,
        result,
    ):

        (
            x_np,
            confidence,
            landmarks,
        ) = landmarks_to_vector(
            result
        )

        hand_present = (
            landmarks is not None
        )

        now = time.monotonic()

        debug_due = (
            self.debug
            and now - self.debug_last_print >= self.debug_interval
        )

        if debug_due:
            self.debug_last_print = now
            self.debug_counter += 1

        if hand_present:

            self.hand_frames += 1

            if self.last_hand_state is not True:

                try:
                    hand_count = len(result.hand_landmarks)
                except Exception:
                    hand_count = 0

                print(
                    f"\n[HAND] DETECTED | hands={hand_count} "
                    f"| confidence={confidence:.3f}"
                )

            self.last_hand_state = True
            self.last_hand_seen = now
            self.hand_loss_reset_done = False

        else:

            self.no_hand_frames += 1

            if self.last_hand_state is not False:
                print("\n[HAND] LOST | hands=0")

            self.last_hand_state = False

        if hand_present:

            nonzero = np.count_nonzero(x_np)

            if self.debug:
                print(
                    f"[LANDMARKS] size={x_np.size} nonzero={nonzero} "
                    f"min={x_np.min():.4f} max={x_np.max():.4f}"
                )

            if x_np.size != INPUT_SIZE:
                print(
                    f"[LANDMARKS] ERROR: expected {INPUT_SIZE}, "
                    f"got {x_np.size}"
                )

            if nonzero == 0:
                print("[LANDMARKS] ERROR: vector is all zero")

        # ----------------------------------------------------
        # NO-HAND FAST PATH
        # ----------------------------------------------------
        # Never feed an all-zero landmark vector through the recurrent
        # model when no hand is present. This saves CPU and prevents
        # the GRU from being updated by meaningless zero frames.
        # Hand-loss reset and sentence flushing are still handled here.
        if not hand_present:
            hand_lost_for = now - self.last_hand_seen

            if (
                hand_lost_for > HAND_LOST_RESET_SECONDS
                and not self.hand_loss_reset_done
            ):
                print(f"\n[HAND] Lost for {hand_lost_for:.2f}s")
                print("[GRU] Performing one-time hand-loss reset.")
                self.decision.reset()

                try:
                    self.state = (
                        self.model
                        .mpgru
                        .cell
                        .init_state(
                            batch_size=1,
                            device=self.device,
                        )
                    )
                    print("[GRU] State reset successfully.")
                except Exception as e:
                    print("[GRU] State reset error:", repr(e))

                self.hand_loss_reset_done = True

            if self.sentence and hand_lost_for > END_OF_SIGN_DELAY:
                sentence = " ".join(self.sentence).strip()

                if sentence:
                    print()
                    print("=" * 60)
                    print("FINAL SENTENCE:", sentence)
                    print("=" * 60)
                    
                    self.tts.speak(sentence)
                    self.firebase.send(sentence)

                self.sentence.clear()
                self.last_added_word = ""
                self.decision.reset()

            return

        # INPUT
        # ----------------------------------------------------

        x_t = (
            torch.from_numpy(
                x_np
            )
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        c_t = torch.tensor(
            [confidence],
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # MP-GRU
        # ----------------------------------------------------

        gru_start = time.monotonic()

        with torch.inference_mode():

            (
                h_t,
                self.state,
                diag,
            ) = (
                self.model
                .mpgru
                .cell(
                    x_t,
                    c_t,
                    self.state,
                    step=self.frame_counter,
                )
            )

        gru_ms = (time.monotonic() - gru_start) * 1000.0

        # ----------------------------------------------------
        # NEUTRAL
        # ----------------------------------------------------

        try:

            nu = float(
                diag[
                    "nu_ema"
                ].item()
            )

        except Exception:

            nu = 0.0

        try:
            nu_raw = float(diag["nu"].item())
        except Exception:
            nu_raw = 0.0

        try:
            nu_base = float(diag["nu_base"].item())
        except Exception:
            nu_base = 0.0

        try:
            occlusion = float(diag["o"].item())
        except Exception:
            occlusion = -1.0

        try:
            visibility = float(diag["s"].item())
        except Exception:
            visibility = -1.0

        try:
            motion_trust = float(diag["p"].mean().item())
        except Exception:
            motion_trust = 0.0

        if debug_due:
            print(
                f"[DEBUG][MPGRU] "
                f"confidence={confidence:.3f} "
                f"nu={nu:.3f} "
                f"nu_base={nu_base:.3f} "
                f"occlusion={occlusion:.1f} "
                f"visibility={visibility:.0f} "
                f"motion_trust={motion_trust:.3f}"
            )

        if self.debug:
            print(
                f"[GRU] c={confidence:.3f} visible={visibility:.1f} "
                f"occlusion={occlusion:.0f} nu={nu:.3f} "
                f"nu_raw={nu_raw:.3f} time={gru_ms:.1f}ms"
            )

        # ----------------------------------------------------
        # HAND PRESENT
        # ----------------------------------------------------

        if hand_present and confidence < MP_DETECTION_CONFIDENCE:

            print(
                f"[FILTER] REJECT: hand confidence {confidence:.3f} "
                f"< {MP_DETECTION_CONFIDENCE:.3f}"
            )

        elif hand_present and nu >= NEUTRAL_THRESHOLD:

            print(
                f"[FILTER] REJECT: neutral blend nu={nu:.3f} "
                f">= {NEUTRAL_THRESHOLD:.3f}"
            )

        if (
            hand_present
            and confidence >= MP_DETECTION_CONFIDENCE
            and nu < NEUTRAL_THRESHOLD
        ):

            classifier_start = time.monotonic()

            with torch.inference_mode():

                z = F.relu(
                    self.model.fc1(
                        h_t
                    )
                )

                embedding = (
                    self.model.embedding(
                        z
                    )
                )

                logits = (
                    self.model.classifier(
                        embedding
                    )
                )

                probs = F.softmax(
                    logits,
                    dim=-1,
                ).squeeze(0)

            if debug_due:

                top_k = min(
                    3,
                    probs.numel(),
                )

                top = torch.topk(
                    probs,
                    k=top_k,
                )

                debug_items = []

                for i in range(top_k):

                    idx = int(top.indices[i])
                    p = float(top.values[i])
                    label = self.idx_to_label.get(
                        idx,
                        f"UNKNOWN_{idx}",
                    )
                    debug_items.append(
                        f"{label}={p:.3f}"
                    )

                print(
                    "[DEBUG][CLASSIFIER] "
                    + " | ".join(debug_items)
                )

            classifier_ms = (
                time.monotonic() - classifier_start
            ) * 1000.0

            self.prediction_frames += 1

            top_k = min(5, probs.numel())
            top = torch.topk(probs, k=top_k)
            raw_predictions = []

            for i in range(top_k):

                idx = int(top.indices[i])
                prob = float(top.values[i])
                label = self.idx_to_label.get(idx, f"class_{idx}")
                raw_predictions.append(f"{label}={prob:.3f}")

            if self.debug:
                print(
                    "[RAW PRED] " + " | ".join(raw_predictions)
                    + f" | {classifier_ms:.1f}ms"
                )

            decision = (
                self.decision.update(
                    probs
                )
            )

            if decision is not None:

                self.accepted_gestures += 1

                label = decision[
                    "label"
                ]

                conf = decision[
                    "confidence"
                ]

                margin = decision[
                    "margin"
                ]

                print(
                    f"\n✓ GESTURE: "
                    f"{label} "
                    f"conf={conf:.2f} "
                    f"margin={margin:.2f}"
                )

                # Prevent duplicate consecutive words.
                if (
                    label
                    !=
                    self.last_added_word
                ):

                    self.sentence.append(
                        label
                    )

                    self.last_added_word = (
                        label
                    )

        else:

            self.decision.reset()

        # ----------------------------------------------------
    def process_customisation(
        self,
        result,
    ):

        (
            _,
            _,
            landmarks,
        ) = landmarks_to_vector(
            result
        )

        stable = (
            self.finger_detector.update(
                landmarks
            )
        )

        if stable is None:

            return

        key = (
            "fist"
            if stable == 0
            else str(stable)
        )

        phrase = (
            self.mode_state[
                "phrases"
            ].get(key)
        )

        if not phrase:

            print(
                f"[CUSTOM] No phrase for "
                f"finger count={stable}"
            )

            return

        print(
            f"\n✓ FINGER: "
            f"{stable} "
            f"-> {phrase}"
        )

        self.tts.speak(
            phrase
        )

        self.firebase.send(
            phrase
        )

    def run(self):

        print(
            "✓ Inference thread started"
        )

        self.running = True

        while not STOP_EVENT.is_set():

            try:

                packet, version = (
                    self.result_buffer.get()
                )

                if packet is None:

                    time.sleep(
                        0.002
                    )

                    continue

                if (
                    packet.timestamp_ms
                    <=
                    self.last_result_timestamp
                ):

                    time.sleep(
                        0.001
                    )

                    continue

                self.last_result_timestamp = (
                    packet.timestamp_ms
                )

                # ------------------------------------------------
                # MODE POLLING
                # ------------------------------------------------

                now = time.monotonic()

                # Poll by wall-clock time instead of inference-frame count.
                # This keeps mode changes responsive even when MediaPipe FPS
                # changes.
                if (
                    now
                    - self.last_mode_poll_time
                    >= MODE_POLL_INTERVAL
                ):

                    self.last_mode_poll_time = now

                    new_state = (
                        load_mode_state()
                    )

                    self.mode_state = (
                        new_state
                    )

                    self.switch_mode(
                        new_state[
                            "mode"
                        ]
                    )

                # ------------------------------------------------
                # PROCESS
                # ------------------------------------------------

                if (
                    self.current_mode
                    ==
                    "customisation"
                ):

                    self.process_customisation(
                        packet.result
                    )

                else:

                    self.process_isl(
                        packet.result
                    )

                self.frame_counter += 1

                # ------------------------------------------------
                # STATUS
                # ------------------------------------------------

                now = time.monotonic()

                if (
                    now
                    - self.last_status
                    >= 1.0
                ):

                    self.last_status = now

                    print(
                        f"[ML] mode="
                        f"{self.current_mode} "
                        f"results="
                        f"{self.frame_counter} "
                        f"hands={self.hand_frames} "
                        f"no_hands={self.no_hand_frames} "
                        f"predictions={self.prediction_frames} "
                        f"accepted={self.accepted_gestures}"
                    )

            except Exception as e:

                print(
                    "[INFERENCE] Worker error:",
                    repr(e),
                )

                # Do not kill the entire product because
                # one inference packet failed.
                self.decision.reset()

                time.sleep(
                    0.01
                )

        self.running = False


# ============================================================
# MEDIAPIPE SUBMISSION WORKER
# ============================================================

class MediaPipeSubmitWorker:

    def __init__(
        self,
        frame_buffer,
        mp_worker,
    ):

        self.frame_buffer = (
            frame_buffer
        )

        self.mp_worker = (
            mp_worker
        )

        self.thread = threading.Thread(
            target=self.run,
            name="mediapipe-submit",
            daemon=True,
        )

        self.start_time = (
            time.monotonic()
        )

        self.last_version = -1

        self.last_submit_time = 0.0

        self.submitted = 0
        self.last_status = time.monotonic()
        self.last_status_submitted = 0

    def start(self):

        self.thread.start()

    def run(self):

        print(
            "✓ MediaPipe submit thread started"
        )

        while not STOP_EVENT.is_set():

            try:

                frame, version = (
                    self.frame_buffer.get()
                )

                if frame is None:

                    time.sleep(
                        0.002
                    )

                    continue

                # CRITICAL FIX:
                #
                # Only submit when the camera produced
                # a NEW frame.
                if (
                    version
                    ==
                    self.last_version
                ):

                    time.sleep(
                        0.001
                    )

                    continue

                now = time.monotonic()

                # Keep the newest frame available, but do not overload
                # MediaPipe with more requests than the inference pipeline
                # can process.
                if (
                    now
                    - self.last_submit_time
                    < MIN_MP_INTERVAL
                ):

                    # Avoid a busy-spin while waiting for the next
                    # MediaPipe submission slot.
                    time.sleep(0.002)

                    continue

                self.last_version = version

                self.last_submit_time = now

                timestamp_ms = int(
                    (
                        now
                        - self.start_time
                    )
                    * 1000
                )

                self.mp_worker.submit_frame(
                    frame,
                    timestamp_ms,
                )

                self.submitted += 1

                now = time.monotonic()
                if now - self.last_status >= 1.0:
                    elapsed = now - self.last_status
                    submitted_delta = (
                        self.submitted
                        - self.last_status_submitted
                    )
                    submit_fps = (
                        submitted_delta / elapsed
                    )

                    print(
                        f"[MP] submit_fps={submit_fps:.1f} "
                        f"total={self.submitted}"
                    )

                    self.last_status = now
                    self.last_status_submitted = self.submitted

            except Exception as e:

                if not STOP_EVENT.is_set():

                    print(
                        "[MP SUBMIT] Worker error:",
                        repr(e),
                    )

                time.sleep(
                    0.01
                )


# ============================================================
# SHUTDOWN
# ============================================================

def safe_stop_thread(
    thread,
    name,
    timeout=2.0,
):

    if thread is None:

        return

    if (
        not thread.is_alive()
    ):

        return

    print(
        f"[SHUTDOWN] Waiting for {name}..."
    )

    thread.join(
        timeout=timeout
    )

    if thread.is_alive():

        print(
            f"[SHUTDOWN] {name} did not "
            f"finish within timeout."
        )


def shutdown(
    camera=None,
    mp_worker=None,
    tts=None,
    firebase=None,
    firestore_worker=None,
    submit_worker=None,
    inference=None,
):

    global _SHUTDOWN_STARTED

    with _SHUTDOWN_LOCK:

        if _SHUTDOWN_STARTED:

            # Signal was already received.
            # Continue cleanup anyway.
            pass

        _SHUTDOWN_STARTED = True

    STOP_EVENT.set()

    if firestore_worker is not None:

        try:
            firestore_worker.stop()
        except Exception as e:
            print(
                "[FIREBASE] Shutdown error:",
                repr(e),
            )

    print()
    print(
        "=" * 60
    )
    print(
        "VOXBRIDGE SHUTDOWN"
    )
    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Stop camera first.
    # This is important because camera read can block.
    # --------------------------------------------------------

    if camera is not None:

        try:
            camera.stop()
        except Exception as e:
            print(
                "[SHUTDOWN] Camera:",
                repr(e),
            )

    # --------------------------------------------------------
    # IMPORTANT: stop/join the MediaPipe submitter BEFORE closing
    # MediaPipe itself. The submit thread may be inside detect_async().
    # Closing the landmarker first can race with an in-flight submit.

    if submit_worker is not None:
        safe_stop_thread(
            submit_worker.thread,
            "MediaPipe submit",
        )

    # Close MediaPipe only after submissions have stopped.

    if mp_worker is not None:
        try:
            mp_worker.close()
        except Exception as e:
            print(
                "[SHUTDOWN] MediaPipe:",
                repr(e),
            )

    # Stop TTS.

    if tts is not None:
        try:
            tts.stop()
        except Exception as e:
            print(
                "[SHUTDOWN] TTS:",
                repr(e),
            )

    # Join remaining workers.

    if inference is not None:
        safe_stop_thread(
            inference.thread,
            "Inference",
        )

    if camera is not None:
        safe_stop_thread(
            camera.thread,
            "Camera",
        )

    print(
        "✓ VoxBridge stopped cleanly."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global MIN_MP_INTERVAL

    parser = argparse.ArgumentParser(
        description=(
            "VoxBridge Production Live Detection"
        )
    )

    parser.add_argument(
        "--source",
        default="0",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed gesture debugging",
    )

    parser.add_argument(
        "--mp-interval",
        type=float,
        default=MIN_MP_INTERVAL,
        help=(
            "Minimum seconds between MediaPipe submissions. "
            "Use 0 for maximum throughput."
        ),
    )

    parser.add_argument(
        "--piper-bin",
        default=os.path.join(
            SCRIPT_DIR,
            ".venv_piper",
            "bin",
            "piper",
        ),
    )

    parser.add_argument(
        "--piper-model",
        default=os.path.join(
            SCRIPT_DIR,
            "en_US-lessac-medium.onnx",
        ),
    )

    parser.add_argument(
        "--audio-device",
        default="plughw:3,0",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
    )

    args = parser.parse_args()

    # Apply the runtime MediaPipe submission interval.
    MIN_MP_INTERVAL = max(0.0, args.mp_interval)

    # ========================================================
    # DEVICE
    # ========================================================

    if args.device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            args.device
        )

    print(
        "=" * 60
    )

    print(
        "VOXBRIDGE PRODUCT LIVE DETECTION"
    )

    print(
        "=" * 60
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Raspberry Pi CPU tuning
    # --------------------------------------------------------

    if device.type == "cpu":

        try:

            torch.set_num_threads(
                min(
                    2,
                    os.cpu_count()
                    or 1,
                )
            )

            torch.set_num_interop_threads(
                1
            )

        except RuntimeError:

            pass

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, idx_to_label = (
        load_model(
            args.checkpoint,
            device,
        )
    )

    # ========================================================
    # BUFFERS
    # ========================================================

    frame_buffer = (
        LatestValue()
    )

    result_buffer = (
        LatestValue()
    )

    # ========================================================
    # WORKERS
    # ========================================================

    camera = CameraWorker(
        args.source,
        frame_buffer,
    )

    mp_worker = MediaPipeWorker(
        args.model,
        result_buffer,
        debug=args.debug,
    )

    submit_worker = (
        MediaPipeSubmitWorker(
            frame_buffer,
            mp_worker,
        )
    )

    tts = TTSWorker(
        args.piper_bin,
        args.piper_model,
        args.audio_device,
        args.sample_rate,
    )

    firebase = FirebaseWorker()

    firestore_worker = FirestoreWorkerProcess()

    inference = InferenceEngine(
        model,
        idx_to_label,
        device,
        result_buffer,
        tts,
        firebase,
        debug=args.debug,
    )

    # ========================================================
    # START
    # ========================================================

    try:

        mp_worker.build()

        firebase.start()

        firestore_worker.start()

        tts.start()

        camera.start()

        # Give camera a short startup period.
        time.sleep(
            0.5
        )

        if STOP_EVENT.is_set():

            return

        submit_worker.start()

        inference.start()

        print()
        print(
            "=" * 60
        )

        print(
            "VOXBRIDGE RUNNING"
        )

        print(
            "=" * 60
        )

        print(
            "Camera: latest NEW-frame mode"
        )

        print(
            f"MediaPipe: LIVE_STREAM | submit interval="
            f"{MIN_MP_INTERVAL:.3f}s"
        )

        print(
            "Inference: separate thread"
        )

        print(
            "TTS: separate worker"
        )

        print(
            "Firebase: separate worker"
        )

        print()
        print(
            "Press Ctrl+C to stop."
        )

        print()

        # ----------------------------------------------------
        # Main watchdog loop.
        #
        # Do not process camera or inference here.
        # ----------------------------------------------------

        while not STOP_EVENT.is_set():

            time.sleep(
                0.5
            )

            # Detect unexpected camera termination.
            if (
                camera.thread is not None
                and not camera.thread.is_alive()
                and not STOP_EVENT.is_set()
            ):

                print(
                    "[WATCHDOG] Camera thread stopped."
                )

                STOP_EVENT.set()

                break

            # Detect unexpected inference termination.
            if (
                inference.thread is not None
                and not inference.thread.is_alive()
                and not STOP_EVENT.is_set()
            ):

                print(
                    "[WATCHDOG] Inference thread stopped."
                )

                STOP_EVENT.set()

                break

    except KeyboardInterrupt:

        # Usually handled by signal handler,
        # but keep this for safety.
        request_shutdown(
            signal.SIGINT,
            None,
        )

    except Exception as e:

        print()
        print(
            "[MAIN] Fatal error:",
            repr(e),
        )

        STOP_EVENT.set()

    finally:

        shutdown(
            camera=camera,
            mp_worker=mp_worker,
            tts=tts,
            firebase=firebase,
            firestore_worker=firestore_worker,
            submit_worker=submit_worker,
            inference=inference,
        )


# ============================================================
# ATEXIT SAFETY
# ============================================================

atexit.register(
    lambda: STOP_EVENT.set()
)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
    
"""Compare values
Setting	MediaPipe max rate	Pi load
0.20	5 FPS	Very low
0.12	8.3 FPS	Good/stable
0.10	10 FPS	Higher
0.08	12.5 FPS	Higher
0.05	20 FPS	High
0.00	~camera rate	Very high / risky"""
