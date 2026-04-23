import time
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from pathlib import Path

from src.opts import parse_opts_offline
from src.model import generate_model
from src.mean import get_mean, get_std
from src.transforms import *
from src.transforms.target_transforms import ClassLabel, VideoID
from src.transforms.target_transforms import Compose as TargetCompose
from src.dataset import get_training_set, get_validation_set, get_test_set, get_online_data
from src.utils import Logger
from src.train import train_epoch
from src.validation import val_epoch
import src.test as test
from src.utils import AverageMeter, calculate_precision, calculate_recall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plot_cm(cm: np.ndarray, classes: list, normalize: bool = True) -> None:
    """Plot a confusion matrix using seaborn."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    ax = plt.subplot()
    sns.heatmap(cm, annot=False, ax=ax, xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    plt.xticks(rotation="vertical")
    plt.yticks(rotation="horizontal")
    plt.tight_layout()


def calculate_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, topk: tuple = (1,)
) -> list[float]:
    """Return top-k accuracy scores for each k in *topk*."""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].float().sum().item()
        results.append(correct_k / batch_size)
    return results


def aggregate_clip_outputs(out_queue: list[np.ndarray], clf_threshold: float) -> torch.Tensor:
    """
    Average per-clip softmax scores for a single video instance.
    Appends a 'null/other' class score when max confidence is below the threshold.
    Returns a CPU (1, C) float tensor.
    """
    output = np.mean(out_queue, axis=0)
    if clf_threshold > 0.1 and output.max() < clf_threshold:
        output = np.append(output, 1.0)
    return torch.from_numpy(output).float().unsqueeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    opt = parse_opts_offline()

    if opt.dataset == 'jester':
        root = Path(opt.jester_root_path)
        opt.video_path      = str(root / opt.jester_video_path)
        opt.annotation_path = str(root / opt.jester_annotation_path)
    elif opt.dataset == 'ipn':
        root = Path(opt.ipn_root_path)
        opt.video_path      = str(root / opt.ipn_video_path)
        opt.annotation_path = str(root / opt.ipn_annotation_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    opt.arch       = f"{opt.model}-{opt.model_depth}"
    opt.store_name = f"{opt.store_name}_{opt.arch}"
    opt.mean       = get_mean(opt.norm_value)
    opt.std        = get_std(opt.norm_value)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Normalisation
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    # Transforms
    spatial_transform = Compose([
        Scale(112),
        CenterCrop(112),
        ToTensor(opt.norm_value),
        norm_method,
    ])

    if opt.true_valid:
        test_batch      = 1
        opt.batch_size  = test_batch
        temporal_transform = Compose([TemporalBeginCrop(opt.sample_duration)])
    else:
        test_batch         = opt.batch_size
        temporal_transform = Compose([TemporalCenterCrop(opt.sample_duration)])

    target_transform = ClassLabel()

    # Dataset / loader
    test_data = get_test_set(opt, spatial_transform, temporal_transform, target_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_batch,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
    )

    # Logging
    result_dir  = Path(opt.result_path)
    test_logger = Logger(
        result_dir / f"test-{opt.test_subset}_{opt.store_name}",
        ["top1", "top5", "precision", "recall", "time", "cm", "class_names", "y_true", "y_pred"],
    )

    print(opt)
    with open(result_dir / f"optsTest_{opt.store_name}.json", "w") as opt_file:
        json.dump(vars(opt), opt_file)
    print(model)
    sys.stdout.flush()
    print(f"Total number of trainable parameters: {pytorch_total_params}")

    # Checkpoint
    if opt.resume_path:
        print(f"Loading checkpoint: {opt.resume_path}")
        checkpoint = torch.load(opt.resume_path, weights_only=True)
        assert opt.arch == checkpoint["arch"], (
            f"Architecture mismatch: expected {opt.arch}, got {checkpoint['arch']}"
        )
        opt.begin_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])

    # ---------------------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------------------

    n_classes = len(test_data.class_names)
    use_top5  = n_classes > 4
    dataset_len = len(test_loader.dataset.data)  # type: ignore

    print(f"\nRunning evaluation on {dataset_len} samples")
    sys.stdout.flush()

    model.eval()

    batch_time = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()
    precisions = AverageMeter()
    recalls    = AverageMeter()

    y_true: list[int] = []
    y_pred: list[int] = []

    # Per-video clip accumulation buffer (only meaningful in true_valid mode)
    out_queue: list[np.ndarray] = []

    for i, (inputs, targets) in enumerate(test_loader):
        if not opt.no_cuda:
            inputs  = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        # --- Forward pass (timed) -------------------------------------------
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(inputs)
            if not opt.no_softmax_in_test:
                outputs = F.softmax(outputs, dim=1)
        batch_time.update(time.perf_counter() - t_start)

        # Accumulate clip scores for per-video aggregation
        out_queue.append(outputs.cpu().numpy()[0].reshape(-1))

        # --- Determine whether this sample closes a video instance ----------
        if opt.true_valid:
            if i + 1 < dataset_len:
                instance_complete = (
                    test_data.data[i]["vid_name"] != test_data.data[i + 1]["vid_name"]
                )
            else:
                instance_complete = True  # final sample always closes its instance
        else:
            instance_complete = True  # every batch is its own evaluation unit

        if not instance_complete:
            continue

        # --- Aggregate clips → single video prediction ----------------------
        if opt.true_valid:
            agg_outputs = aggregate_clip_outputs(out_queue, opt.clf_threshold)
            agg_targets = targets.cpu()
            out_queue   = []
        else:
            agg_outputs = outputs.cpu()
            agg_targets = targets.cpu()

        # --- Compute metrics ------------------------------------------------
        topk      = (1, 5) if use_top5 else (1,)
        prec_vals = calculate_accuracy(agg_outputs, agg_targets, topk=topk)
        prec1     = prec_vals[0]
        prec5     = prec_vals[1] if use_top5 else 0.0

        precision = calculate_precision(agg_outputs, agg_targets)
        recall    = calculate_recall(agg_outputs, agg_targets)

        y_true.extend(agg_targets.numpy().tolist())
        y_pred.extend(agg_outputs.argmax(dim=1).numpy().tolist())

        top1.update(prec1, inputs.size(0))
        precisions.update(precision, inputs.size(0))
        recalls.update(recall, inputs.size(0))
        if use_top5:
            top5.update(prec5, inputs.size(0))

        # --- Per-iteration logging ------------------------------------------
        prefix = f"[{i + 1}/{len(test_loader)}]\tTime {batch_time.val:.5f} ({batch_time.avg:.5f})\t"
        acc_str = f"acc@1 {top1.avg:.5f}" + (f"  acc@5 {top5.avg:.5f}" if use_top5 else "")

        if opt.true_valid:
            vid_info           = test_data.data[i]
            seg_start, seg_end = vid_info["segment"]
            true_label         = agg_targets[0].item()
            pred_label         = agg_outputs.argmax(dim=1)[0].item()
            conf_score         = agg_outputs.max(dim=1)[0].item()
            print(
                f"{prefix}{acc_str}\t"
                f"{vid_info['vid_name']}  true: {true_label}  pred: {pred_label}  "
                f"score: {conf_score:.2f}  start: {seg_start}  end: {seg_end}"
            )
        else:
            pr_str = (
                f"precision {precisions.val:.5f} ({precisions.avg:.5f})\t"
                f"recall {recalls.val:.5f} ({recalls.avg:.5f})"
            )
            print(f"{prefix}{acc_str}\t{pr_str}")

        sys.stdout.flush()

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------

    print("\n----- Evaluation finished -----")
    summary = f"Overall Acc@1 {top1.avg * 100:.3f}%"
    if use_top5:
        summary += f" | Acc@5 {top5.avg * 100:.3f}%"
    summary += f" | Avg Inference Time {batch_time.avg * 1000:.2f} ms"
    print(summary)

    class_names = list(test_data.class_names.values())
    cm = confusion_matrix(y_true, y_pred)
    print("\nClass Names:", class_names)
    print("Confusion Matrix:\n", cm)

    test_logger.log({
        "top1":        top1.avg,
        "top5":        top5.avg if use_top5 else None,
        "precision":   precisions.avg,
        "recall":      recalls.avg,
        "time":        batch_time.avg,
        "cm":          cm.tolist(),
        "class_names": class_names,
        "y_true":      y_true,
        "y_pred":      y_pred,
    })


if __name__ == '__main__':
    main()
