"""
Step 3a: Preprocess a labeled video dataset into cached landmark sequences.

Expects:
    dataset/<label_name>/video1.mp4
    dataset/<label_name>/video2.mp4
    ...
    dataset/<another_label>/video1.mp4
    ...

Produces (mirrors the input structure):
    cache/<label_name>/video1.npz   (contains x: (T,126) float32, c: (T,) float32)
    cache/<label_name>/video2.npz
    ...
    cache/labels.json               (sorted list of label names -> index mapping)

This only needs to run once per dataset (or whenever you add new videos).
Re-running skips videos that already have a cached .npz, unless --overwrite
is passed.

Usage:
    python3 preprocess_dataset.py --dataset dataset --cache cache \
        --model hand_landmarker.task
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from validate_real_landmarks import build_hand_landmarker, landmarks_to_vector


def process_video(mp, model_path: str, video_path: Path):
    """
    Creates a fresh HandLandmarker for this video only. VIDEO running mode
    requires strictly increasing timestamps for the *lifetime of the
    landmarker object* - reusing one landmarker across multiple videos causes
    "Input timestamp must be monotonically increasing" as soon as the second
    video's timestamps restart from 0. A new landmarker per video sidesteps
    this cleanly.
    """
    import cv2

    landmarker, _ = build_hand_landmarker(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError(f"Could not open video: {video_path}")

    xs, cs = [], []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ms_per_frame = 1000.0 / fps

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * ms_per_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            x_np, c_val = landmarks_to_vector(result)
            xs.append(x_np)
            cs.append(c_val)
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()

    if len(xs) == 0:
        return None, None
    return np.stack(xs, axis=0), np.array(cs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="Root folder: dataset/<label>/*.mp4")
    parser.add_argument("--cache", default="cache", help="Output folder for cached .npz files")
    parser.add_argument("--model", default="hand_landmarker.task")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-ext", default=".mp4,.mov,.avi",
                         help="Comma-separated list of video extensions to include")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    cache_root = Path(args.cache)
    cache_root.mkdir(parents=True, exist_ok=True)
    exts = tuple(e.strip().lower() for e in args.video_ext.split(","))

    label_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if not label_dirs:
        raise RuntimeError(f"No label subfolders found under {dataset_root}")

    labels = [p.name for p in label_dirs]
    label_to_idx = {name: i for i, name in enumerate(labels)}
    with open(cache_root / "labels.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)
    print(f"Found {len(labels)} labels: {labels}")

    landmarker, mp = build_hand_landmarker(args.model)
    landmarker.close()  # was only used to validate the model loads; real work uses per-video instances

    total, skipped, failed = 0, 0, 0
    start = time.time()

    for label_dir in label_dirs:
        out_dir = cache_root / label_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        videos = sorted([p for p in label_dir.iterdir() if p.suffix.lower() in exts])

        for video_path in videos:
            out_path = out_dir / (video_path.stem + ".npz")
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            try:
                x, c = process_video(mp, args.model, video_path)
                if x is None:
                    print(f"  [WARN] no frames read from {video_path}, skipping")
                    failed += 1
                    continue
                np.savez_compressed(out_path, x=x, c=c, label=label_dir.name)
                total += 1
                print(f"  processed {video_path} -> {out_path}  ({x.shape[0]} frames)")
            except Exception as e:
                print(f"  [ERROR] {video_path}: {e}")
                failed += 1

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Processed: {total}, skipped (cached): {skipped}, failed: {failed}")


if __name__ == "__main__":
    main()