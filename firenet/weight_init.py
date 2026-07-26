"""Initialize the stem Conv2d for 6-channel input.

The RGB channels (0-2) use pretrained EfficientNetV2-S weights.
The infrared channels (3-5) are initialized with zero-mean, unit-variance
random weights, matching the standardized input distribution.
"""

import torch
from torch import nn


def build_stem_conv(pretrained_conv: nn.Conv2d, num_channels: int) -> nn.Conv2d:
    """Build a new `num_channels`-input Conv2d with pretrained RGB and
    standardized-normal initialization for the extra (infrared) channels.

    Args:
        pretrained_conv: The original 3-input Conv2d with trained weights.
        num_channels: The new number of input channels (typically 6).

    Returns:
        A Conv2d with shape matching pretrained_conv but with `num_channels`
        inputs. RGB channels keep the pretrained weights; extra channels
        are initialized with N(0, 1) since they receive standardized inputs.
    """
    new_conv = nn.Conv2d(
        in_channels=num_channels,
        out_channels=pretrained_conv.out_channels,
        kernel_size=pretrained_conv.kernel_size,
        stride=pretrained_conv.stride,
        padding=pretrained_conv.padding,
        bias=pretrained_conv.bias is not None,
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = pretrained_conv.weight.clone()
        nn.init.normal_(new_conv.weight[:, 3:], mean=0.0, std=1.0)

        if pretrained_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(pretrained_conv.bias)

    return new_conv
