"""
confusion_matrix.py

Loads gesture_classifier.pt, reproduces the same train/val split used during
training, runs the val set through the model, and prints:
  - a full confusion matrix
  - per-class accuracy
  - a "confused with" breakdown for each class (what it gets misclassified as)

USAGE
    Place this file in the same folder as train_classifier.py
    (C:\\Users\\mharr\\Desktop\\model) and run:

        py confusion_matrix.py --model gesture_classifier.pt --cache cache

    It imports the model class, dataset class, and data-loading/splitting
    functions directly from train_classifier.py so the preprocessing here is
    guaranteed to match what the model was trained on.

ASSUMPTIONS (adjust the "ADAPT HERE" block below if these don't match)
    - train_classifier.py exposes something like:
        * a model class (e.g. GestureClassifier) that can be constructed
          from the args stored in the checkpoint and loaded with
          load_state_dict
        * a dataset class / build_dataset() function that takes --cache and
          returns (dataset, classes_dict) or similar
        * a function (e.g. split_dataset / train_val_split) that reproduces
          the same seeded train/val split used in training, OR the split
          indices/seed are simple enough to redo here (seed=42, val_split=0.2)
    - gesture_classifier.pt is a checkpoint dict containing at least
      'model_state_dict' (or the raw state_dict itself) and ideally the
      'classes' mapping and the args used to build the model. If your
      checkpoint format differs, adjust the loading block below.

If any of the imports below fail, open train_classifier.py and swap in the
actual class/function names you find there -- the goal is to reuse your
existing code, not guess new logic.
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# ------------------------------------------------------------------
# ADAPT HERE: import your own model / dataset / split code so this
# script uses EXACTLY the same preprocessing as training.
# ------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from train_classifier import (
        GestureClassifier,     # your model class -- rename if different
        build_dataset,         # function that builds the dataset from --cache
        split_dataset,         # function that reproduces the train/val split
    )
except ImportError as e:
    print(
        "Could not import from train_classifier.py:\n"
        f"  {e}\n"
        "Open train_classifier.py, find the actual names of your model class,\n"
        "dataset-building function, and split function, and edit the import\n"
        "block near the top of this script to match.\n"
    )
    raise


def load_checkpoint(model_path: str, device: str):
    ckpt = torch.load(model_path, map_location=device)

    # Handle both "raw state_dict" and "wrapped dict with metadata" checkpoints.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        classes = ckpt.get("classes")
        args = ckpt.get("args", {})
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        classes = ckpt.get("classes")
        args = ckpt.get("args", {})
    else:
        # Assume it's a raw state_dict; classes/args must come from elsewhere.
        state_dict = ckpt
        classes = None
        args = {}

    return state_dict, classes, args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gesture_classifier.pt")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    args_cli = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    state_dict, classes, ckpt_args = load_checkpoint(args_cli.model, device)

    # ------------------------------------------------------------------
    # ADAPT HERE: build the dataset + reproduce the split the same way
    # train_classifier.py did (same seed, same val_split).
    # ------------------------------------------------------------------
    dataset, classes_from_data = build_dataset(cache=args_cli.cache)
    if classes is None:
        classes = classes_from_data

    train_set, val_set = split_dataset(
        dataset,
        val_split=ckpt_args.get("val_split", args_cli.val_split),
        seed=ckpt_args.get("seed", args_cli.seed),
    )
    print(f"Val samples: {len(val_set)}")

    idx_to_class = {v: k for k, v in classes.items()}
    num_classes = len(classes)

    # ------------------------------------------------------------------
    # ADAPT HERE: construct the model the same way train_classifier.py does,
    # using the hyperparameters stored in the checkpoint if available.
    # ------------------------------------------------------------------
    model = GestureClassifier(
        num_classes=num_classes,
        hidden_size=ckpt_args.get("hidden_size", 64),
        embedding_size=ckpt_args.get("embedding_size", 32),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args_cli.batch_size, shuffle=False
    )

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch in val_loader:
            # ADAPT HERE if your dataset returns something other than (x, y)
            # or (x, y, lengths) -- match whatever train_classifier.py's
            # training loop unpacks.
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)

            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]  # some models return (logits, aux)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---- confusion matrix (rows = true label, cols = predicted label) ----
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    print("\nConfusion matrix (rows = true, cols = predicted)")
    header = "".join(f"{i:>6}" for i in range(num_classes))
    print(f"{'':25}{header}")
    for i in range(num_classes):
        row = "".join(f"{cm[i, j]:>6}" for j in range(num_classes))
        label = idx_to_class[i][:24]
        print(f"{label:<25}{row}")
    print("\nColumn index -> class:")
    for i in range(num_classes):
        print(f"  {i}: {idx_to_class[i]}")

    # ---- per-class accuracy ----
    print("\nPer-class accuracy:")
    for i in range(num_classes):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total if total else float("nan")
        print(f"  {idx_to_class[i]:<50} {acc:.3f}  ({correct}/{total})")

    # ---- "confused with" breakdown ----
    print("\nWhat each class gets misclassified as (excluding correct preds):")
    for i in range(num_classes):
        total = cm[i].sum()
        if total == 0:
            continue
        row = cm[i].copy()
        correct = row[i]
        row[i] = 0
        if row.sum() == 0:
            continue
        print(f"\n  True class: {idx_to_class[i]}  ({correct}/{total} correct)")
        order = np.argsort(-row)
        for j in order:
            if row[j] > 0:
                print(f"      -> predicted '{idx_to_class[j]}': {row[j]} times")

    # ---- optional: save a heatmap image ----
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels([idx_to_class[i][:15] for i in range(num_classes)], rotation=90)
        ax.set_yticklabels([idx_to_class[i][:15] for i in range(num_classes)])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("\nSaved heatmap to confusion_matrix.png")
    except ImportError:
        print("\n(matplotlib not installed -- skipping heatmap image; "
              "`pip install matplotlib` if you want confusion_matrix.png)")


if __name__ == "__main__":
    main()