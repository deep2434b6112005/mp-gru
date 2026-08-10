"""
VoxBridge Dataset Collector - Raspberry Pi (headless SSH version)
Gesture -> Person -> S (start) -> E (end) -> saved MP4
No GUI / cv2.imshow required - works entirely over SSH terminal.

Run: python3 record_dataset.py
"""

import cv2
import os
import sys
import time
import termios
import tty
import threading
import queue

DATASET_DIR = "/home/voxbridge/dataset"
FPS = 30
FRAME_SIZE = (640, 480)


# ---------- Non-blocking single-key reader (works over SSH) ----------
def key_listener(key_queue, stop_event):
    """Reads single keypresses without needing Enter. Runs in its own thread."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            ch = sys.stdin.read(1)
            if ch:
                key_queue.put(ch.lower())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def drain_queue(key_queue):
    """Discard any leftover/stale keypresses sitting in the queue."""
    while not key_queue.empty():
        key_queue.get()


def get_next_index(folder, person):
    existing = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".mp4")
    ]

    nums = []
    for f in existing:
        name = os.path.splitext(f)[0]
        if name.startswith(person + "_"):
            number = name[len(person) + 1:]
            if number.isdigit():
                nums.append(int(number))

    return max(nums, default=0) + 1


def record_one_clip(cam, save_path, key_queue):
    """Records frames until 'e' is received in key_queue."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, FPS, FRAME_SIZE)

    if not out.isOpened():
        print(f"❌ Could not create video file: {save_path}")
        return 0

    frame_count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Camera read failed")
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        out.write(frame)
        frame_count += 1

        # check for 'e' without blocking
        if not key_queue.empty():
            key = key_queue.get()
            if key == 'e':
                break

    out.release()
    if frame_count < FPS:
        print("⚠️ Video is less than 1 second. Consider recording again.")
    return frame_count


def main():
    gesture = input("Enter gesture label: ").strip()
    person = input("Enter person name: ").strip()

    folder = os.path.join(DATASET_DIR, gesture)
    os.makedirs(folder, exist_ok=True)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    if not cam.isOpened():
        print("Could not open camera")
        return

    key_queue = queue.Queue()
    stop_event = threading.Event()
    listener = threading.Thread(target=key_listener, args=(key_queue, stop_event), daemon=True)
    listener.start()

    print(f"\nCamera ready for '{gesture}' / '{person}'")
    print("Press S = START recording | E = END recording | Q = QUIT\n")

    try:
        while True:
            drain_queue(key_queue)  # clear any leftover keys (e.g. stray 'e' from last clip)
            print("Waiting for [S] to start...")
            action = None
            while action not in ('s', 'q'):
                if not key_queue.empty():
                    action = key_queue.get()

            if action == 'q':
                print("Quitting.")
                break

            # short countdown so the gesture isn't cut off at the start
            drain_queue(key_queue)  # ignore any keys pressed during countdown
            print("Get ready...")
            print("2")
            time.sleep(1)
            print("1")
            time.sleep(1)
            drain_queue(key_queue)  # clear any accidental presses during countdown

            idx = get_next_index(folder, person)
            save_path = os.path.join(folder, f"{person}_{idx:03d}.mp4")

            print("RECORDING... press [E] to stop")
            frames = record_one_clip(cam, save_path, key_queue)
            duration = frames / FPS
            print(f"Saved: {save_path}  ({frames} frames, ~{duration:.1f}s)\n")

    finally:
        stop_event.set()
        cam.release()
        print(f"\nSession ended. Files saved under: {folder}")


if __name__ == "__main__":
    main()