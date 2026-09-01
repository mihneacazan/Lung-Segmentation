"""
SegResNet Model for Lung Tumor Segmentation.

Uses MONAI's SegResNet architecture — a ResNet-based encoder-decoder
designed specifically for medical image segmentation. Features residual
connections throughout the encoder and decoder, enabling deeper networks
with better gradient flow.

Usage:
    from src.models.segresnet import build_segresnet
"""

import torch
from monai.networks.nets import SegResNet


def build_segresnet(in_channels=1, out_channels=1, spatial_dims=2,
                    channels=None):
    """
    Builds a 2D SegResNet model using MONAI.
    
    SegResNet uses residual blocks with group normalization and VAE-style
    regularization options. It is deeper than standard U-Net and better
    at capturing fine-grained segmentation boundaries.
    
    Args:
        in_channels (int): Number of input channels (1 for 2D, 3/5 for 2.5D).
        out_channels (int): Number of output channels (1 for binary segmentation).
        spatial_dims (int): 2 for 2D segmentation.
        channels (tuple|None): Width, given in the same form as for the U-Nets
            so one flag drives every architecture. SegResNet is parameterised by
            a single `init_filters` and doubles it per stage internally, so only
            the first entry is used and the rest are ignored.
        
    Returns:
        MONAI SegResNet model.
    """
    init_filters = int(channels[0]) if channels else 16
    model = SegResNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.1
    )
    return model


if __name__ == "__main__":
    print("=== Testing SegResNet Architecture ===")
    model = build_segresnet(in_channels=1, out_channels=1)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    test_input = torch.randn(2, 1, 192, 192)
    test_output = model(test_input)
    
    print(f"  Input shape:  {test_input.shape}")
    print(f"  Output shape: {test_output.shape}")
    print(f"  Parameters:   {trainable_params:,} trainable / {total_params:,} total")
