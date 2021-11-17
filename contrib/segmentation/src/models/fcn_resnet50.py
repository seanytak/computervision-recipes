import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation.fcn import FCNHead


class FCNResNet50(nn.Module):
    def __init__(self, n_classes, pretrained=True, is_feature_extracting: bool = False):
        """Load Fully Convolutional Network with ResNet-50 backbone

        Parameters
        ----------
        n_classes : int
            Number of classes
        pretrained : bool
            True if model should use pre-trained weights from COCO
        is_feature_extracting : bool
            True if the convolutional layers should be set to non-trainable retaining their original
            parameters
        """
        self.model = torchvision.self.models.segmentation.fcn_resnet50(
            pretrained=pretrained
        )

        if is_feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = FCNHead(2048, n_classes)
        self.model.aux_classifier = FCNHead(1024, n_classes)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            return self.model.forward(x)
