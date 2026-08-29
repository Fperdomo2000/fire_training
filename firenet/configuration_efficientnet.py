"""Config for the torchvision EfficientNetV2-S wrapper."""

from transformers import PretrainedConfig

# ImageNet stats the torchvision pretrained weights were trained with.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_ID2LABEL = {0: "no_fire", 1: "fire"}


class TorchvisionEfficientNetConfig(PretrainedConfig):
    """Config for `TorchvisionEfficientNetForClassification`.

    Wraps `torchvision.models.efficientnet_v2_s`. `num_channels` controls the
    stem: 3 for RGB (PNG dataset, stock pretrained stem) or 6 for the RGB+IR
    TIFF dataset (stem replaced, see `firenet.weight_init`).
    """

    model_type = "torchvision_efficientnet_v2_s"

    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 384,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        id2label: dict | None = None,
        label2id: dict | None = None,
        classification_threshold: float = 0.5,
        **kwargs,
    ):
        self.num_channels = num_channels
        self.image_size = image_size
        extra_channels = max(num_channels - 3, 0)
        self.image_mean = image_mean or IMAGENET_MEAN + [0.0] * extra_channels
        self.image_std = image_std or IMAGENET_STD + [1.0] * extra_channels
        # P(fire) >= classification_threshold is classified as "fire" at inference
        # time. Set by train.py after sweeping thresholds on the validation set.
        self.classification_threshold = classification_threshold

        if id2label is None:
            id2label = DEFAULT_ID2LABEL
        if label2id is None:
            label2id = {name: idx for idx, name in id2label.items()}

        super().__init__(id2label=id2label, label2id=label2id, **kwargs)
