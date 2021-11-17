import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3(nn.Module):
    """Wrapper class around torchvision DeepLabV3 to ensure it works with
    Mixed Precision Training
    """

    def __init__(self, n_classes: int, pretrained: bool, is_feature_extracting: bool):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=pretrained, aux_loss=True
        )

        if is_feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = DeepLabHead(2048, n_classes)
        self.model.aux_classifier = DeepLabHead(1024, n_classes)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            return self.model.forward(x)
