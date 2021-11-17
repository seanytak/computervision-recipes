"""
"""
import argparse
import copy
import logging
import multiprocessing
import time
import uuid
from os.path import join
from pathlib import Path
from typing import Dict, Tuple
import warnings

import albumentations as A
import mlflow
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from config.augmentation import preprocessing, augmentation
from src.datasets.semantic_segmentation import (
    SemanticSegmentationPyTorchDataset,
    SemanticSegmentationStochasticPatchingDataset,
)
from src.losses.loss import semantic_segmentation_class_balancer
from src.metrics.metrics import get_semantic_segmentation_metrics, log_metrics
from src.models.deeplabv3 import DeepLabV3
from src.models.fcn_resnet50 import FCNResNet50


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def log_metrics(results: Dict[str, torch.Tensor], classes: List[str], split: str):
    """Log metrics to stdout and AML

    Parameters
    ----------
    results : Dict
        Key is the name of the metric, value is a metric tensor
        If the metric is a mean, it is a 0-dim tensor
        If the metric is per class, it is a C-dim tensor (C for number of classes)
    split : {"train", "val", "test"}
        Split that the metrics are for
    """
    # Import does not appear to work on some non-AML environments
    from azureml.core.run import Run, _SubmittedRun

    # Get script logger
    log = logging.getLogger(__name__)

    # Get AML context
    run = Run.get_context()

    split = split.capitalize()

    for metric_name, result in results.items():
        log_name = f"[{split}] {metric_name}"
        if "mean" in metric_name:
            result = float(result)

            if isinstance(run, _SubmittedRun):
                run.parent.log(log_name, result)
            run.log(log_name, result)
        elif "per_class" in metric_name:
            result = {c: float(r) for c, r in zip(classes, result)}

            # Steps are children the experiment they belong to so the parent
            # also needs a log
            if isinstance(run, _SubmittedRun):
                run.parent.log_row(log_name, **result)
            run.log_row(log_name, **result)

        log.info(f"{log_name}: {result}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="deeplab")
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--loss", type=str, default="balanced_cross_entropy")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--aux-loss-weight", type=float, default=0.4)
    parser.add_argument(
        "--patch-strategy",
        type=str,
        default="deterministic_center_crop",
    )
    parser.add_argument(
        "--val-patch-strategy",
        type=str,
        default="",
    )
    parser.add_argument("--patch-dim", type=str, default="512, 512")
    parser.add_argument("--resize-dim", type=str, default="3632, 5456")
    parser.add_argument("--iou-thresholds", type=str, default="0.5, 0.3")
    parser.add_argument("--cache-strategy", type=str, default="none")
    args = parser.parse_args()

    fh = logging.FileHandler(str(args.log_file))
    log.addHandler(fh)

    train_dir = str(args.train_dir)
    val_dir = str(args.val_dir)

    model_dir = join("outputs", "models")
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    if args.cache_dir is not None:
        cache_dir = str(args.cache_dir)
    else:
        cache_dir = join("/tmp", str(uuid.uuid4()))
    cache_strategy = str(args.cache_strategy)

    # Model and Loss
    model_name = str(args.model_name).lower()
    pretrained = bool(args.pretrained)
    loss = str(args.loss).lower()

    # Hyperparameters
    n_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    aux_loss_weight = float(args.aux_loss_weight)
    patch_strategy = str(args.patch_strategy).lower()
    val_patch_strategy = str(args.val_patch_strategy).lower()

    batch_validation_perc = float(args.batch_validation_perc)

    patch_dim: Tuple[int, int] = tuple([int(x) for x in args.patch_dim.split(",")])
    resize_dim: Tuple[int, int] = tuple([int(x) for x in args.resize_dim.split(",")])
    iou_thresholds = [float(x) for x in args.iou_thresholds.split(",")]

    # Train on GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    log.info(f"Model Name: {model_name}")
    log.info(f"Epochs: {n_epochs}")
    log.info(f"Learning Rate: {learning_rate}")
    log.info(f"Auxiliary Loss Weight: {aux_loss_weight}")
    log.info(f"Batch Size: {batch_size}")
    log.info(f"Patch Strategy: {patch_strategy}")
    log.info(f"GPU: {torch.cuda.is_available()}")
    log.info(f"Patch Dimension: {patch_dim}")
    log.info(f"Resize Dimension: {resize_dim}")
    log.info(f"Pretrained: {pretrained}")

    train_labels_filepath = join(train_dir, "train.json")
    val_labels_filepath = join(val_dir, "val.json")

    Dataset = SemanticSegmentationPyTorchDataset

    # Validation patch strategy may differ from train patch strategy
    if val_patch_strategy == "":
        if patch_strategy == "resize":
            val_patch_strategy = "resize"
        else:
            val_patch_strategy = "crop_all"

    if patch_strategy == "stochastic":
        dataset = SemanticSegmentationStochasticPatchingDataset(
            f"{train_dir}/patch/*.png",
            f"{train_dir}/mask",
            augmentation=preprocessing,
        )
        dataset_val = SemanticSegmentationStochasticPatchingDataset(
            f"{val_dir}/patch/*.png",
            f"{val_dir}/mask",
            augmentation=preprocessing,
        )

        dataset = Dataset(
            labels_filepath=train_labels_filepath,
            classes=classes,
            annotation_format="coco",
            root_dir=train_dir,
            cache_dir=join(cache_dir, "train"),
            cache_strategy=cache_strategy,
            # preprocessing=get_preprocessing(),
            augmentation=augmentation,
            patch_strategy=patch_strategy,
            patch_dim=patch_dim,
            resize_dim=resize_dim,
        )
        dataset_val = Dataset(
            labels_filepath=val_labels_filepath,
            classes=classes,
            annotation_format="coco",
            root_dir=val_dir,
            cache_dir=join(cache_dir, "val"),
            cache_strategy=cache_strategy,
            # Specified as augmentation because it's not guaranteed to target
            # the correct instances
            augmentation=get_validation_preprocessing(),
            patch_strategy=val_patch_strategy,
            patch_dim=patch_dim,
            resize_dim=resize_dim,
        )

    dataset_len = len(dataset)
    dataset_val_len = len(dataset_val)
    tot_training_batches = dataset_len // batch_size
    tot_validation_batches = dataset_val_len // batch_size

    print(
        f"Train dataset number of images: {dataset_len} | Batch size: {batch_size} | Expected number of batches: {tot_training_batches}"
    )
    print(
        f"Validation dataset number of images: {dataset_val_len} | Batch size: {batch_size} | Expected number of batches: {tot_validation_batches}"
    )

    num_classes: int = len(classes) + 1  # Plus 1 for background

    # define training and validation data loaders
    # drop_last True to avoid single instances which throw an error on batch norm layers

    num_workers = min(
        # Preferably use 2/3's of total cpus. If the cpu count is 1, it will be set to 0 which will result
        # in dataloader using the main thread
        int(round(multiprocessing.cpu_count() * 2 / 3)),
        # Maxing the num_workers at 8 due to shared memory limitations
        8,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # get the model using our helper function
    if model_name == "fcn" or model_name == "fcn-resnet50":
        model = FCNResNet50(num_classes, pretrained=pretrained)
    elif model_name == "deeplab" or model_name == "deeplabv3":
        model = DeepLabV3(
            num_classes,
            pretrained=pretrained,
            is_feature_extracting=pretrained,
        )
    else:
        raise ValueError(f'Provided model name "{model_name}" is not supported.')

    model = torch.nn.DataParallel(model)
    # move model to the right device
    model.to(device)

    # Create balanced cross entropy loss
    if loss == "balanced_cross_entropy":
        weights = semantic_segmentation_class_balancer(dataset)
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        warnings.warn(
            f"The loss function {loss} is not available. Using the default CrossEntropyLoss instead."
        )
        criterion = nn.CrossEntropyLoss()

    criterion = criterion.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=0.9, weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    metrics = get_semantic_segmentation_metrics(num_classes, thresholds=iou_thresholds)
    metrics = metrics.to(device)
    best_mean_iou = 0

    best_model_wts: Dict = copy.deepcopy(model.state_dict())

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(n_epochs):
        start = time.time()
        train_loss = 0
        val_loss = 0

        # Switch to train mode for training
        model.train()

        for batch_num, (images, targets) in enumerate(dataloader, 0):
            batch_time = time.time()
            images: torch.Tensor = images.to(device).float()
            targets: torch.Tensor = targets.to(device).long()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(images)

                outputs = out["out"]
                loss = criterion(outputs, targets)

                # FCN Model w/o pre-training does not have an auxiliary loss component
                # so we avoid this calculation
                if not (not pretrained and model_name == "fcn"):
                    aux_outputs = out["aux"]
                    aux_loss = criterion(aux_outputs, targets)
                    loss = loss + aux_loss_weight * aux_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)

            train_loss += loss.item() * images.size(0)

            metrics(preds, targets)

            print(
                f"Train Epoch: {epoch} | Batch: {batch_num} | Batch Loss: {loss.item()} | Batch Time: {time.time() - batch_time}"
            )

        train_loss /= len(dataloader.dataset)
        print(f"Epoch: {epoch} | Train Loss: {train_loss}")

        # Compute and log training metrics
        results = metrics.compute()
        results["loss"] = train_loss
        log_metrics(results, classes, split="train")
        metrics.reset()

        # Switch to eval mode for validation
        model.eval()

        if epoch < n_epochs - 1 and batch_validation_perc < 1.0:
            max_batch_num = int(tot_validation_batches * batch_validation_perc)
        else:
            max_batch_num = -1

        with torch.no_grad():
            for batch_num, (images, targets) in enumerate(dataloader_val, 0):
                if max_batch_num == -1 or batch_num < max_batch_num:
                    images: torch.Tensor = images.to(device).float()
                    targets: torch.Tensor = targets.to(device).long()

                    with torch.cuda.amp.autocast():
                        outputs = model(images)["out"]
                        loss = criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)

                    val_loss += loss.item() * images.size(0)
                    metrics(preds, targets)

                    print(
                        f"Validation Epoch: {epoch} | Batch {batch_num} | Batch Loss: {loss.item()}"
                    )

        val_loss /= len(dataloader_val.dataset)
        print(f"Epoch: {epoch} | Val Loss: {val_loss}")

        # Compute and log validation metrics
        results = metrics.compute()
        results["loss"] = val_loss
        log_metrics(results, classes, split="val")
        metrics.reset()

        mean_iou = float(results["mean_iou_0_5"])
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(
                model.state_dict(),
                join(model_dir, f"{model_name}_checkpoint_{epoch}.pth"),
            )

    # Log best model
    model_filename = f"{model_name}_final.pth"
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), join(model_dir, model_filename))
    mlflow.log_params(hyperparameters)
    mlflow.log_artifact(model_filename)

    # Log augmentations
    albumentations_filename = "augmentation.json"
    A.save(augmentation, albumentations_filename)
    mlflow.log_artifact(albumentations_filename)
