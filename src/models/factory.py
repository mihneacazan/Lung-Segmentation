"""
Model factory for the segmentation experiments.

Maps a --model_type string onto a network, so that every architecture goes
through the same training loop, the same evaluation, and the same checkpoint
selection rule. Comparing architectures is only meaningful if nothing else
differs between the runs.

Usage:
    from src.models.factory import build_model
    model = build_model("attention_unet", in_channels=1)
"""

from src.models.unet_2d import build_unet_2d
from src.models.attention_unet import build_attention_unet
from src.models.segresnet import build_segresnet


MODEL_TYPES = ("unet", "attention_unet", "segresnet")


def build_model(model_type: str, in_channels: int = 1, out_channels: int = 1):
    """
    Builds a segmentation network by name.

    Args:
        model_type (str): One of 'unet', 'attention_unet', 'segresnet'.
        in_channels (int): Input channels. 1 for plain 2D, or n_adjacent for a
            2.5D model that stacks consecutive slices as channels.
        out_channels (int): Output channels (1 for binary segmentation).

    Returns:
        torch.nn.Module: The requested model.

    Raises:
        ValueError: If model_type is not recognised.
    """
    if model_type == "unet":
        return build_unet_2d(in_channels=in_channels, out_channels=out_channels)
    if model_type == "attention_unet":
        return build_attention_unet(in_channels=in_channels, out_channels=out_channels)
    if model_type == "segresnet":
        return build_segresnet(in_channels=in_channels, out_channels=out_channels)

    raise ValueError(
        f"Unknown model_type: {model_type!r}. Choose from: {', '.join(MODEL_TYPES)}"
    )
