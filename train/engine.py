"""The training and validation loops for one epoch each.

What this file does:
    train_one_epoch()  shows the model every training image once, updating the
                       weights after every batch
    validate()         measures the model on the validation images, with no
                       weight updates, and returns its predictions

Vim paper correspondence (arXiv 2401.09417):
    None of the architecture. This is the standard supervised training loop
    behind every result in the paper's Section 4.

Shapes:
    each batch: images (B, 3, 128, 128), labels (B,) -> logits (B, 5) -> loss ()

Deliberate differences from the paper:
    None here; the recipe (optimiser, schedule, epochs) is set in train/train.py.

THE CANONICAL FIVE-STEP PYTORCH TRAINING LOOP (once per batch):
    1. optimizer.zero_grad()   wipe the gradients left over from the last batch
    2. logits = model(images)  forward pass: compute the predictions
    3. loss = criterion(...)   one number saying how wrong the predictions are
    4. loss.backward()         backward pass: compute d(loss)/d(weight) for every
                               weight (the gradient), by the chain rule
    5. optimizer.step()        nudge every weight a small step DOWNHILL, i.e. in
                               the direction that makes the loss smaller

    Why step 1 is needed at all: PyTorch ADDS each new gradient to whatever is
    already stored in a weight's `.grad` field, instead of overwriting it. That
    is deliberate (it lets you sum gradients over several small batches when a
    big one does not fit in memory), but it means that without zero_grad the
    gradient of batch 2 would be mixed with that of batch 1, batch 3 with both,
    and so on, and the steps would point in increasingly wrong directions.
"""

import numpy as np  # collecting predictions into arrays
import torch  # tensors, autocast, no_grad
from tqdm import tqdm  # a progress bar around any loop

from .utils import AverageMeter  # running averages (train/utils.py)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    scaler, use_amp=False, grad_clip=None):
    """Train for one pass over `loader`. Returns {"loss": ..., "accuracy": ...}.

    Args:
        model: the VisionMamba network.
        loader: the training DataLoader (shuffled, augmented).
        criterion: the loss function (weighted cross-entropy).
        optimizer: AdamW.
        scheduler: learning-rate schedule, stepped once per BATCH.
        device: torch.device("cuda") or torch.device("cpu").
        scaler: a torch.amp.GradScaler (does nothing when AMP is disabled).
        use_amp: run the forward pass in mixed precision (GPU only).
        grad_clip: maximum gradient norm, or None for no clipping.
    """
    # model.train() puts every layer in TRAINING mode. Layers such as Dropout and
    # BatchNorm behave differently when training and when evaluating. Our model
    # has neither, so this changes nothing today, but calling it is the correct
    # habit: forget it after a validate() call and such layers would misbehave.
    model.train()

    # Running mean of the loss over the epoch, and a count of correct answers
    loss_meter = AverageMeter()
    correct = 0
    seen = 0

    # tqdm(loader) wraps the loader in a progress bar; the loop itself is unchanged.
    # leave=False removes the bar when the epoch ends, keeping the output tidy.
    for images, labels in tqdm(loader, desc="train", leave=False):
        # Move the batch to the GPU (if we have one). non_blocking=True lets the copy
        # overlap with computation when the batch is in pinned memory.
        # images: (B, 3, 128, 128), labels: (B,)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ---- Step 1: zero_grad ----
        # set_to_none=True frees the old gradient tensors instead of filling them
        # with zeros; same effect for the maths, slightly less memory traffic.
        optimizer.zero_grad(set_to_none=True)

        # ---- Steps 2-3: forward pass and loss, inside autocast ----
        # AUTOMATIC MIXED PRECISION (AMP): inside this block, PyTorch runs the
        # operations that are safe in 16-bit floats (matrix multiplies) in float16,
        # which is roughly twice as fast on a T4 and halves their memory, and keeps
        # precision-sensitive operations (exp, softplus, sums, the loss) in float32.
        # It is a SPEED optimisation only: the model and the maths are unchanged.
        # With enabled=False (e.g. on CPU) the block does nothing at all.
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            # Step 2: forward. (B, 3, 128, 128) -> (B, 5)
            logits = model(images)
            # Step 3: loss. Cross-entropy compares the 5 scores with the true grade:
            # softmax turns the scores into probabilities, and the loss is
            # -log(probability given to the correct grade), averaged over the batch
            # (with class weights, see train/train.py). (B, 5), (B,) -> () a scalar
            loss = criterion(logits, labels)

        # ---- Step 4: backward ----
        # float16 has a small range: tiny gradients can round to zero ("underflow").
        # The GradScaler multiplies the loss by a large factor first, so the
        # gradients are large enough to survive in float16; it divides them back
        # down before the weights are updated. Disabled, scale() returns the loss
        # unchanged, so this one line serves both the GPU and the CPU case.
        scaler.scale(loss).backward()

        # Gradient clipping: if the length (norm) of the whole gradient vector,
        # across all 6.9M weights, exceeds grad_clip, shrink it to that length.
        # The direction is kept; only an abnormally large step is prevented. The
        # gradients must be un-scaled first, or we would clip the scaled values.
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # ---- Step 5: step ----
        # AdamW updates every weight: new = old - lr * (adaptive gradient step),
        # plus a small pull towards zero (weight decay). Through the scaler: if the
        # scaled gradients overflowed to inf/NaN, the step is skipped and the scale
        # factor reduced. Then the scaler adjusts its factor for the next batch.
        scaler.step(optimizer)
        scaler.update()

        # Move the learning rate one step along its warmup + cosine schedule
        scheduler.step()

        # Bookkeeping for the epoch summary. .item() turns a one-number tensor into
        # a Python float (and waits for the GPU to finish this batch).
        batch_size = labels.shape[0]
        loss_meter.update(loss.item(), n=batch_size)
        # argmax(dim=1): the index of the highest of the 5 scores = predicted grade
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        seen += batch_size

    # max(seen, 1) avoids dividing by zero for an (unexpected) empty loader
    return {"loss": loss_meter.avg, "accuracy": correct / max(seen, 1)}


# PYTHON CONCEPT: a decorator again (see vim/config.py). @torch.no_grad() wraps the
# whole function so that everything inside runs with gradient tracking switched
# OFF, exactly as if the body were inside `with torch.no_grad():`.
# Why: during the forward pass PyTorch normally records every operation (the
# "computation graph") so that backward() can run later. Validation never calls
# backward(), so recording would only waste memory (for our scan: every hidden
# state of every step) and time.
@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=False):
    """Evaluate on `loader` without updating any weights.

    Returns a dict with the mean loss, and the true and predicted grades as
    NumPy arrays (eval/metrics.py turns those into accuracy, QWK, ...).
    """
    # model.eval(): EVALUATION mode, the opposite of model.train(). Dropout would
    # be switched off and BatchNorm would use its stored statistics. (Again a
    # no-op for our model, but always required before evaluating.)
    model.eval()

    loss_meter = AverageMeter()
    # Python lists collecting one small array per batch
    all_labels = []
    all_predictions = []

    for images, labels in tqdm(loader, desc="val", leave=False):
        # Move the batch to the device: (B, 3, 128, 128) and (B,)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass only (no zero_grad, backward or step): (B, 3, 128, 128) -> (B, 5)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # Mean loss over the whole set, weighting each batch by its size
        loss_meter.update(loss.item(), n=labels.shape[0])

        # Predicted grade = index of the highest score; (B, 5) -> (B,)
        predictions = logits.argmax(dim=1)

        # .cpu().numpy(): copy back from the GPU and convert to a NumPy array
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    # Glue the per-batch arrays into one long array each: (n_images,)
    return {
        "loss": loss_meter.avg,
        "labels": np.concatenate(all_labels),
        "predictions": np.concatenate(all_predictions),
    }
