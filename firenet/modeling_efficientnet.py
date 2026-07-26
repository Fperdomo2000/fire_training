"""HF-style wrapper around torchvision's EfficientNetV2-S for classification."""

from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

from .configuration_efficientnet import TorchvisionEfficientNetConfig
from .weight_init import build_stem_conv


class TorchvisionEfficientNetForClassification(PreTrainedModel):
    """`torchvision/efficientnet_v2_s` wrapped as a Hugging Face model.

    - `config.num_channels == 3`: stock pretrained RGB stem, untouched.
    - `config.num_channels == 6`: stem replaced with a 6-input Conv2d; see
      `firenet.weight_init` for how the extra (infrared) channels are seeded.
    """

    config_class = TorchvisionEfficientNetConfig
    base_model_prefix = "efficientnet"
    main_input_name = "pixel_values"

    def __init__(self, config: TorchvisionEfficientNetConfig):
        super().__init__(config)

        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, config.num_labels)

        if config.num_channels != 3:
            old_stem = backbone.features[0][0]
            backbone.features[0][0] = build_stem_conv(old_stem, config.num_channels)

        self.efficientnet = backbone
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        # No-op on purpose: all weights already come from the pretrained
        # torchvision backbone (plus firenet.weight_init for the stem).
        # `post_init()` still requires this method to exist -- if it did
        # anything here it would clobber the pretrained weights we just set.
        pass

    def forward(self, pixel_values=None, labels=None, **kwargs) -> ImageClassifierOutput:
        logits = self.efficientnet(pixel_values)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return ImageClassifierOutput(loss=loss, logits=logits)
