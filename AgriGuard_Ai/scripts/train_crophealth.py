"""Train the crop-health MobileNetV3-Small classifier.

    python scripts/train_crophealth.py --epochs 6

Input dataset (created by scripts/prepare_crop_dataset.py):

    data/crop_health/healthy/*.jpg
    data/crop_health/possible_disease/*.jpg
    data/crop_health/splits.csv        reproducible 70/15/15 split

Recipe - plain, well-known transfer learning, nothing exotic:

* MobileNetV3-Small pretrained on ImageNet, 2-class head.
* Stage 1: freeze the backbone, train only the new classification head.
* Stage 2: unfreeze the last two backbone blocks and fine-tune at a low LR.
* Resize 224x224, ImageNet normalisation, modest augmentation.
* Early stopping is NOT used - the epoch count is a CLI flag so the run is
  reproducible.

Evaluation happens on the test split only, and the test split is never touched
during training or model selection.

    ##########################################################################
    # Accuracy below is measured on PlantVillage images taken on a plain      #
    # laboratory background. It is NOT an estimate of performance on ESP32-CAM #
    # field photos. See docs/limitations.md.                                  #
    ##########################################################################
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import (  # noqa: E402
    CROP_HEALTH_DIR,
    CROP_HEALTH_META_PATH,
    CROP_HEALTH_WEIGHTS_PATH,
)
from agriguard.cropdata import API_LABELS, CLASS_NAMES  # noqa: E402
from agriguard.imageutils import (  # noqa: E402
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    InvalidImageError,
    build_transforms,
    load_image_rgb,
)
from agriguard.version import CROP_HEALTH_MODEL_VERSION  # noqa: E402

from torch.utils.data import DataLoader, Dataset  # noqa: E402

METHOD = "MobileNetV3-Small"

#: The input contract, recorded in the model JSON so an integrator can tell
#: what the network actually consumes without reading imageutils.py. The values
#: are pulled from the shared module, so they cannot drift away from the
#: transform the model was trained with.
INPUT_FEATURES: Dict[str, object] = {
    "input_type": "image",
    "colour_space": "RGB",
    "image_size": [IMAGE_SIZE, IMAGE_SIZE],
    "scale": "0..1",
    "normalisation_mean": list(IMAGENET_MEAN),
    "normalisation_std": list(IMAGENET_STD),
    "learned_features": "32x downsampled convolutional embedding learned by "
    "training; not a fixed, named feature vector",
    "classifier_input": "pooled convolutional embedding -> linear(2)",
}
SPLITS = ("train", "validation", "test")


class CropHealthDataset(Dataset):
    """Reads the split manifest produced by prepare_crop_dataset.py."""

    def __init__(self, root: Path, records, transform):
        self.root = root
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.root / record["key"]
        image = load_image_rgb(path)
        return self.transform(image), CLASS_NAMES.index(record["category"])


def read_manifest(root: Path):
    manifest = root / "splits.csv"
    if not manifest.exists():
        raise SystemExit(
            f"{manifest} not found. Run: python scripts/prepare_crop_dataset.py"
        )
    with open(manifest, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_loader(dataset, batch_size, shuffle, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = correct = 0
    losses = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, targets)
        losses.append(float(loss.item()) * targets.size(0))
        correct += int((logits.argmax(1) == targets).sum().item())
        total += int(targets.size(0))
    return total, correct, sum(losses) / max(total, 1)


def build_model(num_classes=2):
    net = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
    return net


def set_backbone_trainable(net, trainable: bool):
    for parameter in net.features.parameters():
        parameter.requires_grad = trainable


def unfreeze_last_blocks(net, blocks=2):
    set_backbone_trainable(net, False)
    for block in list(net.features)[-blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def predict_all(model, loader, device):
    model.eval()
    probabilities, targets = [], []
    with torch.no_grad():
        for images, batch_targets in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(batch_targets.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def compute_metrics(y_true, y_pred, probabilities=None):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "test_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision_per_class": {
            API_LABELS[CLASS_NAMES[i]]: round(float(precision[i]), 4) for i in range(2)
        },
        "recall_per_class": {
            API_LABELS[CLASS_NAMES[i]]: round(float(recall[i]), 4) for i in range(2)
        },
        "f1_per_class": {
            API_LABELS[CLASS_NAMES[i]]: round(float(f1[i]), 4) for i in range(2)
        },
        "support_per_class": {
            API_LABELS[CLASS_NAMES[i]]: int(support[i]) for i in range(2)
        },
        "macro_precision": round(float(precision.mean()), 4),
        "macro_recall": round(float(recall.mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "confusion_matrix": {
            "labels": [API_LABELS[name] for name in CLASS_NAMES],
            "rows": "true class",
            "values": matrix.tolist(),
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
    }
    if probabilities is not None:
        metrics["mean_confidence"] = round(float(probabilities.max(axis=1).mean()), 4)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CROP_HEALTH_DIR)
    parser.add_argument("--out", type=Path, default=CROP_HEALTH_WEIGHTS_PATH)
    parser.add_argument("--meta", type=Path, default=CROP_HEALTH_META_PATH)
    parser.add_argument("--epochs", type=int, default=6, help="total epochs")
    parser.add_argument(
        "--head-epochs", type=int, default=2, help="epochs with a frozen backbone"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="stage 2 learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print the stored training report from the metadata file and exit "
        "without training anything (safe for documentation/evidence runs)",
    )
    args = parser.parse_args()

    if args.report_only:
        if not args.meta.exists():
            raise SystemExit(f"no stored report at {args.meta}; train the model first")
        print(f"stored training report from {args.meta}")
        print(json.dumps(json.loads(args.meta.read_text(encoding="utf-8")), indent=2))
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device      : {device}  (torch {torch.__version__})")
    print(f"seed        : {args.seed}")

    records = read_manifest(args.data)
    by_split = {
        name: [r for r in records if r["split"] == name] for name in SPLITS
    }
    for name in SPLITS:
        if not by_split[name]:
            raise SystemExit(f"split '{name}' is empty; re-run prepare_crop_dataset.py")
        counts = {
            category: sum(1 for r in by_split[name] if r["category"] == category)
            for category in CLASS_NAMES
        }
        print(f"{name:<11}: {len(by_split[name]):5d} images  {counts}")

    train_tf = build_transforms(train=True)
    eval_tf = build_transforms(train=False)

    datasets = {
        name: CropHealthDataset(args.data, rows, train_tf if name == "train" else eval_tf)
        for name, rows in by_split.items()
    }
    train_loader = make_loader(datasets["train"], args.batch_size, True, args.seed)
    val_loader = make_loader(datasets["validation"], args.batch_size, False)
    test_loader = make_loader(datasets["test"], args.batch_size, False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()

    head_params = list(model.classifier.parameters())
    set_backbone_trainable(model, False)
    optimizer = optim.Adam(head_params, lr=1e-3)
    print(f"image size  : {IMAGE_SIZE}x{IMAGE_SIZE}   batch size: {args.batch_size}")
    print(f"stage 1     : frozen backbone, {args.head_epochs} epochs, lr 1e-3")

    history = []
    best_val = -1.0
    best_state = None
    for epoch in range(1, args.head_epochs + 1):
        started = time.time()
        train_n, train_ok, train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_n, val_ok, val_loss = run_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "stage": "frozen-backbone",
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_ok / train_n, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_ok / val_n, 4),
                "seconds": round(time.time() - started, 1),
            }
        )
        print(
            f"epoch {epoch}: train loss {train_loss:.4f} acc {train_ok / train_n:.4f} | "
            f"val loss {val_loss:.4f} acc {val_ok / val_n:.4f} "
            f"({history[-1]['seconds']}s)"
        )
        val_acc = val_ok / val_n
        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if args.epochs > args.head_epochs:
        unfreeze_last_blocks(model, blocks=2)
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr
        )
        print(
            f"stage 2     : last 2 backbone blocks unfrozen, "
            f"{args.epochs - args.head_epochs} epochs, lr {args.lr}"
        )
        for epoch in range(args.head_epochs + 1, args.epochs + 1):
            started = time.time()
            train_n, train_ok, train_loss = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_n, val_ok, val_loss = run_epoch(model, val_loader, criterion, device)
            history.append(
                {
                    "epoch": epoch,
                    "stage": "fine-tune",
                    "train_loss": round(train_loss, 4),
                    "train_accuracy": round(train_ok / train_n, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_ok / val_n, 4),
                    "seconds": round(time.time() - started, 1),
                }
            )
            print(
                f"epoch {epoch}: train loss {train_loss:.4f} acc {train_ok / train_n:.4f} | "
                f"val loss {val_loss:.4f} acc {val_ok / val_n:.4f} "
                f"({history[-1]['seconds']}s)"
            )
            val_acc = val_ok / val_n
            if val_acc > best_val:
                best_val = val_acc
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    # ---- restore the best-validation weights before touching the test set ----
    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"\nrestored best-validation weights (val accuracy {best_val:.4f})"
            " - the last epoch is not necessarily the best one"
        )

    # ---- TEST EVALUATION, once, on unseen images only ----------------------
    probabilities, y_true = predict_all(model, test_loader, device)
    y_pred = probabilities.argmax(axis=1)
    metrics = compute_metrics(y_true, y_pred, probabilities)

    print("\n--- TEST SET (unseen images) ---")
    print(f"Test samples : {metrics['test_samples']}")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"Balanced acc : {metrics['balanced_accuracy']:.4f}")
    for label in metrics["precision_per_class"]:
        print(
            f"{label:<18} precision {metrics['precision_per_class'][label]:.4f} "
            f"recall {metrics['recall_per_class'][label]:.4f} "
            f"F1 {metrics['f1_per_class'][label]:.4f} "
            f"support {metrics['support_per_class'][label]}"
        )
    cm = metrics["confusion_matrix"]
    print(f"Confusion matrix (rows = true, cols = predicted {cm['labels']}):")
    for label, row in zip(cm["labels"], cm["values"]):
        print(f"  {label:<18} {row}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)

    meta = {
        "model": "crop-health",
        "method": METHOD,
        "model_version": CROP_HEALTH_MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_dataset": "PlantVillage color images (public, see docs/crop_health_dataset.md)",
        "classes": list(CLASS_NAMES),
        "class_labels": dict(API_LABELS),
        "features": INPUT_FEATURES,
        "image_size": IMAGE_SIZE,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "head_epochs_frozen_backbone": args.head_epochs,
        "learning_rate_stage2": args.lr,
        "seed": args.seed,
        "parameters_total": int(sum(p.numel() for p in model.parameters())),
        "parameters_trainable_last_stage": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "dataset_counts": {
            name: {
                "total": len(by_split[name]),
                **{
                    category: sum(
                        1 for r in by_split[name] if r["category"] == category
                    )
                    for category in CLASS_NAMES
                },
            }
            for name in SPLITS
        },
        "history": history,
        "best_validation_accuracy": round(float(best_val), 4),
        "test_metrics": metrics,
        "limitations": [
            "Two-class screening model: Healthy vs Possible Disease. It never "
            "identifies which disease is present and is not a diagnostic system.",
            "PlantVillage images are single leaves on plain laboratory "
            "backgrounds. ESP32-CAM field photos with soil, shadows and mixed "
            "lighting are a different domain and will score lower.",
            "Class balance is 31% Healthy / 69% Possible Disease, so the raw "
            "confidence is not a calibrated probability.",
            "38 original PlantVillage classes were collapsed into 2 categories; "
            "many different diseases share the 'Possible Disease' label.",
            "A high score here must not be read as real-world field accuracy.",
        ],
    }
    with open(args.meta, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"\nsaved weights -> {args.out}")
    print(f"saved metadata -> {args.meta}")


if __name__ == "__main__":
    main()
