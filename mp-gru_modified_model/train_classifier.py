"""
Step 3c: Gesture classifier built on top of MP-GRU, plus a training loop.

Architecture:
    (x, c) padded batch
        -> MPGRU (per-timestep hidden states for the whole batch)
        -> gather each sample's hidden state at its true final frame
           (using `lengths`, not the padded end - see gesture_dataset.py)
        -> Linear -> ReLU -> Dropout -> Linear -> L2-normalize
        -> embedding, scored against class prototypes (cosine logits)
        -> optional auxiliary linear classifier head on the same features
           (helps stabilize training early on / with small datasets)

Usage:
    python3 train_classifier.py --cache cache --epochs 60 --batch-size 8

Changes from the previous version (aimed at lifting val_acc):
  1. Vectorized the per-sample Python loop for LOO-prototype + cosine
     logits into batched tensor ops (same math, faster, fewer places for
     a subtle indexing bug to hide).
  2. Added an auxiliary standard cross-entropy classifier head trained
     jointly with the prototype loss. Pure metric/prototype learning
     from scratch tends to be noisy on small datasets; blending in a
     normal softmax classifier head usually stabilizes and speeds up
     convergence a lot.
  3. Added light on-the-fly augmentation for landmark sequences
     (per-sample rotation/scale jitter + additive noise + random
     temporal masking), applied only when --augment is on and only
     during training. This is usually the single biggest lever for
     small gesture datasets.
  4. Added an LR scheduler (ReduceLROnPlateau on val loss), gradient
     clipping, label smoothing, and early stopping with patience.
  5. Made temperature and label smoothing real CLI args instead of a
     hardcoded value that silently overrode the function default.
  6. Added a per-class accuracy printout at the end so you can see
     which gestures are dragging down val_acc instead of just one
     aggregate number.

Changes in this version (reproducibility / observability, no effect on
the MP-GRU architecture or inference logic):
  7. Deterministic random seeding (random / numpy / torch / cuda) so
     repeated runs with the same args are comparable, with an optional
     --deterministic-cudnn flag (off by default -- it costs throughput
     for full determinism, and day-to-day comparisons don't need it).
  8. File + console logging via the `logging` module -- every epoch's
     numbers land in a timestamped log file under --log-dir, so a
     crashed or killed run still leaves a full history.
  9. tqdm progress bars on both the train and val batch loops, plus an
     outer epoch progress bar.
  10. Per-head accuracy logged separately every epoch: proto-only,
      aux-only, and the combined (blended) accuracy already used for
      early stopping / checkpointing. This directly answers the "is the
      auxiliary head doing most of the work" question raised for the
      prototype+auxiliary setup -- log all three and you can see it
      epoch by epoch instead of only ever seeing the blended number.

NOTE: I don't have access to gesture_dataset.py, mp_gru.py, or the
actual cached data, so I can't run this and confirm it clears 70%
myself -- these are the highest-value, well-established changes for
this kind of setup (small dataset, sequence encoder + metric learning
head). Treat the numbers below as a starting point and iterate based
on what the per-class breakdown shows you.
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import GestureClassifier
from gesture_dataset import make_dataloaders


# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed: int, deterministic_cudnn: bool = False):
    """
    Seed random / numpy / torch (+ CUDA) so repeated runs with the same
    args are comparable. deterministic_cudnn is off by default: it makes
    cudnn pick only deterministic kernels, which can meaningfully slow
    down training (sometimes 10-20%) for a level of determinism that
    isn't usually needed for day-to-day hyperparameter comparisons on a
    dataset this small. Turn it on only if you need bit-for-bit repeat
    runs (e.g. debugging a suspected non-determinism bug).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ==========================================================
# Logging
# ==========================================================

def setup_logging(log_dir: str, run_name: str = None):
    """
    Configure logging to write to both console and a timestamped file
    under log_dir. Returns the log file path so it can be printed /
    referenced later (e.g. attached to a run's saved checkpoint).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or "train"
    log_path = Path(log_dir) / f"{name}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Clear any handlers from a previous call in the same process
    # (e.g. if main() is invoked more than once, as in a notebook).
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    return str(log_path)


def compute_class_weights(loader, num_classes, device):
    """
    Inverse-frequency class weights from the training set, to counteract
    imbalance (here, up to 6.7x between the largest and smallest class).
    weight_c = total_samples / (num_classes * count_c)
    """
    counts = torch.zeros(num_classes)
    for _, _, labels, _ in loader:
        for label in labels:
            counts[label] += 1
    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    return weights.to(device)


# ==========================================================
# Augmentation (landmark sequences: assumed (B, T, 126) ==
# 2 hands x 21 landmarks x (x, y, z))
# ==========================================================

def augment_landmarks(x, c, lengths, noise_std=0.01, scale_range=0.1,
                       rot_range_deg=10.0, occlusion_prob=0.3,
                       occlusion_len_range=(2, 8), occlusion_c_value=0.05):
    """
    Cheap, in-place-safe augmentation applied per training batch.
    Operates on clones of x and c so the originals are untouched.

    - random per-sample uniform scale jitter
    - random per-sample small rotation about the z axis (in the xy plane)
    - additive Gaussian noise
    - synthetic occlusion: with probability `occlusion_prob`, blanks out a
      short *contiguous* span of frames within the sample's valid length,
      zeroing the landmarks AND dropping c_t below vis_low for that same
      span.

    That last part matters a lot here: MPGRUCell's whole occlusion
    pipeline (visibility hysteresis -> occlusion counter -> motion-
    confidence decay -> neutral blend) is driven off c_t, not off whether
    the landmarks look like zeros. A frame with blanked landmarks but
    untouched (high) confidence is a state that never occurs in real
    detector output, and it fights the mechanism the cell was designed
    around instead of exercising it. A contiguous span (rather than
    scattered single frames) also matches the real failure mode this
    architecture targets -- a hand leaving frame for a stretch, not
    flickering frame-by-frame -- so it gives the occlusion counter /
    motion prediction / neutral-blend logic something realistic to learn
    from.

    x : (B, T, 126) float tensor, last dim = concatenated (x,y,z) triples
    c : (B, T) float tensor, MediaPipe detection confidence in [0, 1]
    lengths : (B,) valid lengths per sample

    Returns (x_aug, c_aug).
    """
    B, T, D = x.shape
    assert D % 3 == 0, "expected landmark features in (x,y,z) triples"
    device = x.device
    x_aug = x.clone()
    c_aug = c.clone()

    # per-sample scale in [1-scale_range, 1+scale_range]
    scale = 1.0 + (torch.rand(B, 1, 1, device=device) * 2 - 1) * scale_range
    x_aug = x_aug * scale

    # per-sample rotation about z, applied to every (x, y) pair
    theta = (torch.rand(B, device=device) * 2 - 1) * (rot_range_deg * 3.14159265 / 180.0)
    cos_t = torch.cos(theta).view(B, 1)
    sin_t = torch.sin(theta).view(B, 1)

    x_view = x_aug.view(B, T, D // 3, 3)
    xs = x_view[..., 0]
    ys = x_view[..., 1]
    new_xs = xs * cos_t.unsqueeze(-1) - ys * sin_t.unsqueeze(-1)
    new_ys = xs * sin_t.unsqueeze(-1) + ys * cos_t.unsqueeze(-1)
    x_view[..., 0] = new_xs
    x_view[..., 1] = new_ys
    x_aug = x_view.view(B, T, D)

    # additive noise
    x_aug = x_aug + torch.randn_like(x_aug) * noise_std

    # synthetic contiguous occlusion span, consistent in both x and c
    if occlusion_prob > 0:
        for i in range(B):
            L = int(lengths[i].item())
            if L <= occlusion_len_range[0]:
                continue
            if torch.rand(1).item() > occlusion_prob:
                continue
            max_len = min(occlusion_len_range[1], L - 1)
            span_len = torch.randint(occlusion_len_range[0], max_len + 1, (1,)).item()
            start = torch.randint(0, L - span_len + 1, (1,)).item()
            x_aug[i, start:start + span_len, :] = 0.0
            c_aug[i, start:start + span_len] = occlusion_c_value

    return x_aug, c_aug


def temporal_warp(x, c, lengths, warp_range=(0.85, 1.15)):
    """
    Random per-sample speed perturbation: resample each sequence's valid
    (unpadded) span to a new length in [warp_range[0], warp_range[1]] x its
    original length, via linear interpolation along the time axis, then
    re-pad. Simulates the same gesture signed a bit faster or slower.

    This is a better fit here than mixup/cutmix: those blend two different
    *samples* together, and for a sequence of joint coordinates that means
    averaging landmark positions from two different gestures frame-by-frame
    -- which produces hand poses that don't correspond to any real hand
    configuration. Time-warping only touches the timing of a single real
    sequence, so every resulting frame is still a pose that actually
    occurred.

    x : (B, T, 126)
    c : (B, T)
    lengths : (B,)

    Returns (x_out, c_out, new_lengths). x_out/c_out are zero-padded past
    each sample's new_lengths[i], same as the original collate_fn output.
    """
    B, T, D = x.shape
    device = x.device
    x_out = torch.zeros_like(x)
    c_out = torch.zeros_like(c)
    new_lengths = lengths.clone()

    for i in range(B):
        L = int(lengths[i].item())
        if L <= 2:
            new_lengths[i] = L
            x_out[i, :L, :] = x[i, :L, :]
            c_out[i, :L] = c[i, :L]
            continue

        rate = float(torch.empty(1).uniform_(*warp_range).item())
        new_L = max(2, min(T, int(round(L * rate))))

        seq = x[i, :L, :].unsqueeze(0).transpose(1, 2)          # (1, D, L)
        seq_resampled = F.interpolate(seq, size=new_L, mode='linear', align_corners=True)
        seq_resampled = seq_resampled.transpose(1, 2).squeeze(0)  # (new_L, D)

        cseq = c[i, :L].view(1, 1, L)
        cseq_resampled = F.interpolate(cseq, size=new_L, mode='linear', align_corners=True)
        cseq_resampled = cseq_resampled.view(new_L)

        x_out[i, :new_L, :] = seq_resampled
        c_out[i, :new_L] = cseq_resampled
        new_lengths[i] = new_L

    return x_out, c_out, new_lengths


# ==========================================================
# Prototype Helper Functions
# ==========================================================

def compute_prototypes(embeddings, labels, num_classes):
    device = embeddings.device
    dim = embeddings.size(1)

    prototypes = torch.zeros(num_classes, dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            prototypes[c] = embeddings[mask].mean(dim=0)
            prototypes[c] = F.normalize(prototypes[c], dim=0)
            counts[c] = mask.sum()

    return prototypes, counts


def batched_loo_logits(embeddings, labels, prototypes, counts, temperature):
    """
    Vectorized leave-one-out cosine logits for a whole batch at once
    (replaces the old per-sample Python loop -- same math).

    embeddings : (B, D)  L2-normalized
    labels     : (B,)
    prototypes : (C, D)  L2-normalized, frozen (no grad)
    counts     : (C,)    frozen sample counts per class

    Returns
    -------
    logits : (B, C)
    """
    B = embeddings.size(0)
    C = prototypes.size(0)

    n = counts[labels].clamp(min=1).unsqueeze(1)          # (B, 1)
    own_proto = prototypes[labels]                        # (B, D)

    loo = (own_proto * n - embeddings.detach()) / (n - 1).clamp(min=1)
    loo = F.normalize(loo, dim=1)
    # where n == 1 (only sample of its class), fall back to the
    # non-LOO prototype since there's nothing to leave out
    single = (counts[labels] <= 1).unsqueeze(1)
    loo = torch.where(single, own_proto, loo)

    # (B, C, D): broadcast prototypes per sample, then swap in each
    # sample's own LOO row at its own label position
    proto_per_sample = prototypes.unsqueeze(0).expand(B, C, -1).clone()
    proto_per_sample[torch.arange(B), labels] = loo

    logits = torch.einsum('bd,bcd->bc', embeddings, proto_per_sample) * temperature
    return logits


def cosine_logits(embeddings, prototypes, temperature=15.0):
    embeddings = F.normalize(embeddings, dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    logits = embeddings @ prototypes.t()
    return logits * temperature


def compute_frozen_prototypes(model, loader, device, num_classes):
    """
    Compute prototypes from a loader with the model in eval mode
    (no dropout noise) and no grad tracking. Always uses train_loader
    data -- this is the frozen reference table for the epoch.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for x, c, labels, lengths in loader:
            x = x.to(device)
            c = c.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            _, emb, _ = model(x, c, lengths)

            all_embeddings.append(emb)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return compute_prototypes(all_embeddings, all_labels, num_classes)


def ema_update_prototypes(ema_prototypes, fresh_prototypes, counts, momentum):
    """
    Smooth the prototype table across steps instead of hard-replacing it.

    On a dataset this small, the embedding space can shift a fair amount
    between one prototype computation and the next, which means a hard
    replace every time makes the classifier chase a moving target rather
    than converging against a stable reference. Blending the freshly
    computed prototypes into a running EMA (only for classes actually seen
    -- `counts > 0` -- so an unseen class's entry isn't dragged toward
    zero) keeps the reference table stable while still tracking real
    drift in the embedding space over the course of training.

    ema_prototypes : (C, D) running EMA table, or None on the first call
    fresh_prototypes : (C, D) prototypes computed from the current model
    counts : (C,) per-class sample counts from the same computation
    momentum : float in [0, 1); higher = slower-moving / more stable
    """
    if ema_prototypes is None:
        return fresh_prototypes.clone()

    seen = counts > 0
    updated = ema_prototypes.clone()
    updated[seen] = momentum * ema_prototypes[seen] + (1 - momentum) * fresh_prototypes[seen]
    updated = F.normalize(updated, dim=1)
    return updated


def make_balanced_sampler(train_loader):
    """
    Build a WeightedRandomSampler that inverse-weights samples by class
    frequency, as an ALTERNATIVE to loss-level class weighting (not in
    addition to it -- stacking both would double-compensate for the
    imbalance and risks over-correcting against the majority classes,
    same concern as using a very large inverse-frequency loss weight on
    its own).

    Returns a new DataLoader with the same dataset/batch_size/collate_fn
    as train_loader but sampler-based balancing instead of plain shuffle.
    """
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler

    subset = train_loader.dataset  # torch.utils.data.Subset
    base_dataset = subset.dataset
    labels = [base_dataset.samples[i][1] for i in subset.indices]
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return DataLoader_like(train_loader, sampler=sampler)


def DataLoader_like(loader, sampler):
    from torch.utils.data import DataLoader
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        collate_fn=loader.collate_fn,
        num_workers=getattr(loader, "num_workers", 0),
    )


def run_epoch(
        model,
        loader,
        optimizer,
        device,
        train: bool,
        num_classes,
        prototypes,
        counts,
        class_weights=None,
        temperature=30.0,
        label_smoothing=0.0,
        aux_loss_weight=0.5,
        eval_proto_weight=0.7,
        augment=False,
        temporal_warp_aug=False,
        occlusion_prob=0.15,
        grad_clip=1.0,
        progress_desc=None):
    """
    prototypes/counts are FROZEN, computed once per epoch from train_loader
    only (see compute_frozen_prototypes) -- this function never builds its
    own prototype table from whatever loader it's given, which was the
    earlier source of validation leakage.

    Total loss = prototype cosine-logit CE + aux_loss_weight * auxiliary
    classifier-head CE. The auxiliary head shares the same underlying
    features and gives the model a cleaner, lower-variance gradient
    signal than pure prototype learning alone, especially early in
    training / on small datasets.

    eval_proto_weight controls how the two heads are combined for the
    reported prediction: `eval_proto_weight * softmax(proto_logits) +
    (1 - eval_proto_weight) * softmax(class_logits)`. A flat 50/50 average
    assumes both heads are equally reliable, which usually isn't true --
    one typically ends up noticeably stronger than the other, so this is
    exposed as a tunable rather than hardcoded equal weighting.

    Returns a dict with loss and three separate accuracy numbers, so the
    combined (blended) accuracy can be attributed back to its two heads
    instead of only ever being reported as one aggregate figure:
      - "acc"       : accuracy of the blended prediction (what's used
                      for checkpointing / early stopping)
      - "proto_acc" : accuracy using ONLY the prototype head's argmax
      - "aux_acc"   : accuracy using ONLY the auxiliary head's argmax
      - "per_class_acc" : per-class accuracy of the blended prediction
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_proto_correct = 0
    total_aux_correct = 0
    total_count = 0

    num_classes_seen_correct = torch.zeros(num_classes)
    num_classes_seen_total = torch.zeros(num_classes)

    context = torch.enable_grad() if train else torch.no_grad()

    iterable = tqdm(loader, desc=progress_desc, leave=False, unit="batch")

    with context:
        for x, c, labels, lengths in iterable:
            x = x.to(device)
            c = c.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            if train and temporal_warp_aug:
                x, c, lengths = temporal_warp(x, c, lengths)
            if train and augment:
                x, c = augment_landmarks(x, c, lengths, occlusion_prob=occlusion_prob)

            _, embeddings, class_logits = model(x, c, lengths)

            if train:
                proto_logits = batched_loo_logits(
                    embeddings, labels, prototypes, counts, temperature
                )
            else:
                proto_logits = cosine_logits(embeddings, prototypes, temperature)

            proto_loss = F.cross_entropy(
                proto_logits, labels,
                weight=class_weights, label_smoothing=label_smoothing
            )
            aux_loss = F.cross_entropy(
                class_logits, labels,
                weight=class_weights, label_smoothing=label_smoothing
            )
            loss = proto_loss + aux_loss_weight * aux_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * x.size(0)

            proto_probs = F.softmax(proto_logits, dim=1)
            aux_probs = F.softmax(class_logits, dim=1)
            combined = eval_proto_weight * proto_probs + (1 - eval_proto_weight) * aux_probs

            prediction = combined.argmax(dim=1)
            proto_prediction = proto_probs.argmax(dim=1)
            aux_prediction = aux_probs.argmax(dim=1)

            batch_correct = (prediction == labels).sum().item()
            total_correct += batch_correct
            total_proto_correct += (proto_prediction == labels).sum().item()
            total_aux_correct += (aux_prediction == labels).sum().item()
            total_count += x.size(0)

            for cls in range(num_classes):
                mask = labels == cls
                if mask.any():
                    num_classes_seen_total[cls] += mask.sum().item()
                    num_classes_seen_correct[cls] += (prediction[mask] == cls).sum().item()

            iterable.set_postfix(
                loss=f"{loss.item():.3f}",
                acc=f"{batch_correct / x.size(0):.2f}"
            )

    per_class_acc = (num_classes_seen_correct / num_classes_seen_total.clamp(min=1)).tolist()

    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
        "proto_acc": total_proto_correct / total_count,
        "aux_acc": total_aux_correct / total_count,
        "per_class_acc": per_class_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--out", default="gesture_classifier.pt")
    parser.add_argument("--no-class-weights", action="store_true",
                         help="Disable inverse-frequency class weighting (on by default)")
    parser.add_argument("--balanced-sampler", action="store_true",
                         help="Use a WeightedRandomSampler for train batches instead of "
                              "loss-level class weights. Don't combine with class weights -- "
                              "pick one imbalance-correction method, not both.")
    parser.add_argument("--temperature", type=float, default=15.0,
                         help="Cosine logit temperature. Worth sweeping 10-25; no single "
                              "value is clearly best without running it on your data.")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--aux-loss-weight", type=float, default=0.5,
                         help="Weight of the auxiliary classifier-head loss during training")
    parser.add_argument("--eval-proto-weight", type=float, default=0.7,
                         help="Weight given to the prototype head vs. the auxiliary head "
                              "(1 - this) when combining predictions at eval time")
    parser.add_argument("--proto-momentum", type=float, default=0.8,
                         help="EMA momentum for the prototype table across steps; higher "
                              "= more stable / slower to track embedding drift")
    parser.add_argument("--augment", action="store_true", default=True,
                         help="Apply landmark augmentation during training (on by default)")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--temporal-warp", action="store_true", default=True,
                         help="Apply random speed-warping augmentation (on by default)")
    parser.add_argument("--no-temporal-warp", dest="temporal_warp", action="store_false")
    parser.add_argument("--occlusion-prob", type=float, default=0.15,
                         help="Per-sample probability of injecting a synthetic occlusion "
                              "span during training augmentation")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler-patience", type=int, default=8,
                         help="Epochs of no val_loss improvement before LR is reduced")
    parser.add_argument("--patience", type=int, default=18,
                         help="Early-stopping patience, in epochs without val_acc improvement. "
                              "Kept comfortably above --scheduler-patience so the LR gets a "
                              "chance to actually help before training gives up.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for random/numpy/torch/cuda")
    parser.add_argument("--deterministic-cudnn", action="store_true",
                         help="Force fully deterministic cudnn kernels (slower). Off by "
                              "default -- seeding alone is enough for day-to-day comparisons.")
    parser.add_argument("--log-dir", default="logs",
                         help="Directory to write the timestamped training log file to")
    parser.add_argument("--run-name", default=None,
                         help="Optional label used in the log filename, e.g. 'proto_only'")
    args = parser.parse_args()

    log_path = setup_logging(args.log_dir, args.run_name)
    logging.info(f"Logging to {log_path}")
    logging.info(f"Args: {vars(args)}")

    set_seed(args.seed, deterministic_cudnn=args.deterministic_cudnn)
    logging.info(f"Seed set to {args.seed} (deterministic_cudnn={args.deterministic_cudnn})")

    if args.balanced_sampler and not args.no_class_weights:
        logging.info("Note: --balanced-sampler already corrects for class imbalance at the "
                      "sampling level, so loss-level class weights are being disabled to avoid "
                      "stacking both corrections. Pass --no-class-weights explicitly to silence "
                      "this note.")
        args.no_class_weights = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader, val_loader, dataset = make_dataloaders(
        args.cache, batch_size=args.batch_size, val_split=args.val_split)
    num_classes = len(dataset.label_to_idx)
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    logging.info(f"Classes ({num_classes}): {dataset.label_to_idx}")
    logging.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    if args.balanced_sampler:
        train_loader = make_balanced_sampler(train_loader)
        logging.info("Using WeightedRandomSampler for train batches.")

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_loader, num_classes, device)
        logging.info(f"Class weights (inverse frequency): {class_weights.cpu().tolist()}")

    model = GestureClassifier(
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience
    )

    best_val_acc = 0.0
    epochs_without_improvement = 0
    ema_prototypes = None

    # Start timer
    start_time = time.time()

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        epoch_start = time.time()

        fresh_prototypes, counts = compute_frozen_prototypes(
            model, train_loader, device, num_classes
        )
        ema_prototypes = ema_update_prototypes(
            ema_prototypes, fresh_prototypes, counts, momentum=args.proto_momentum
        )

        train_metrics = run_epoch(
            model, train_loader, optimizer, device, train=True,
            num_classes=num_classes, prototypes=ema_prototypes, counts=counts,
            class_weights=class_weights, temperature=args.temperature,
            label_smoothing=args.label_smoothing,
            aux_loss_weight=args.aux_loss_weight,
            eval_proto_weight=args.eval_proto_weight,
            augment=args.augment, temporal_warp_aug=args.temporal_warp,
            occlusion_prob=args.occlusion_prob, grad_clip=args.grad_clip,
            progress_desc=f"Epoch {epoch} [train]"
        )

        # Re-freeze (and re-blend into the EMA) after the train step's
        # weight updates, so validation is scored against the model's
        # current state -- still built only from train_loader, never
        # val_loader.
        fresh_prototypes, counts = compute_frozen_prototypes(
            model, train_loader, device, num_classes
        )
        ema_prototypes = ema_update_prototypes(
            ema_prototypes, fresh_prototypes, counts, momentum=args.proto_momentum
        )

        val_metrics = run_epoch(
            model, val_loader, optimizer, device, train=False,
            num_classes=num_classes, prototypes=ema_prototypes, counts=counts,
            class_weights=class_weights, temperature=args.temperature,
            label_smoothing=args.label_smoothing,
            aux_loss_weight=args.aux_loss_weight,
            eval_proto_weight=args.eval_proto_weight,
            augment=False, temporal_warp_aug=False, grad_clip=None,
            progress_desc=f"Epoch {epoch} [val]"
        )

        train_loss, train_acc = train_metrics["loss"], train_metrics["acc"]
        val_loss, val_acc = val_metrics["loss"], val_metrics["acc"]
        per_class_acc = val_metrics["per_class_acc"]

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        epoch_bar.set_postfix(
            train_acc=f"{train_acc:.3f}",
            val_acc=f"{val_acc:.3f}",
            lr=f"{current_lr:.1e}"
        )

        # Per-head accuracy, logged separately every epoch -- this is
        # the data needed to tell whether the auxiliary head or the
        # prototype head is actually driving the combined accuracy.
        logging.info(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"(proto={train_metrics['proto_acc']:.3f} aux={train_metrics['aux_acc']:.3f}) "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"(proto={val_metrics['proto_acc']:.3f} aux={val_metrics['aux_acc']:.3f}) "
            f"| lr={current_lr:.2e} | time={epoch_time:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "label_to_idx": dataset.label_to_idx,
                "hidden_size": args.hidden_size,
                "embedding_size": args.embedding_size,
            }, args.out)
            logging.info(f"  -> saved new best model (val_acc={val_acc:.3f}) to {args.out}")
            logging.info("  per-class val acc: " + ", ".join(
                f"{idx_to_label[i]}={a:.2f}" for i, a in enumerate(per_class_acc)
            ))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logging.info(f"No val_acc improvement for {args.patience} epochs, stopping early.")
                break

    logging.info(f"Best val_acc: {best_val_acc:.3f}")

    total_time = time.time() - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    logging.info(f"Total training time: {hours}h {minutes}m {seconds:.2f}s")
    logging.info(f"Full log saved to {log_path}")


if __name__ == "__main__":
    main()
