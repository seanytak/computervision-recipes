import logging
from typing import Dict, List

import torch
from torchmetrics import F1, Accuracy, IoU, MetricCollection, Precision, Recall


def semantic_segmentation_metrics(
    num_classes: int, thresholds: List[float] = [0.5, 0.3]
) -> MetricCollection:
    """Construct MetricCollection of Segmentation Metrics

    Parameters
    ----------
    num_classes : int
        Number of classes
    thresholds : List[float]
        List of thresholds for different IOU computing

    Returns
    -------
    metrics : torchmetrics.MetricCollection
        Collection of Segmentation metrics
    """
    metrics = {
        "mean_accuracy": Accuracy(num_classes=num_classes, mdmc_average="global"),
        "per_class_accuracy": Accuracy(
            num_classes=num_classes, average="none", mdmc_average="global"
        ),
        "mean_precision": Precision(num_classes=num_classes, mdmc_average="global"),
        "per_class_precision": Precision(
            num_classes=num_classes, average="none", mdmc_average="global"
        ),
        "mean_recall": Recall(num_classes=num_classes, mdmc_average="global"),
        "per_class_recall": Recall(
            num_classes=num_classes, average="none", mdmc_average="global"
        ),
        "mean_f1": F1(num_classes=num_classes, mdmc_average="global"),
        "per_class_f1": F1(
            num_classes=num_classes, average="none", mdmc_average="global"
        ),
    }

    for threshold in thresholds:
        threshold_string = str(threshold).replace(".", "_")
        metrics[f"mean_iou_{threshold_string}"] = IoU(
            num_classes=num_classes, reduction="elementwise_mean", threshold=threshold
        )
        metrics[f"per_class_iou_{threshold_string}"] = IoU(
            num_classes=num_classes, reduction="none", threshold=threshold
        )

    return MetricCollection(metrics)
