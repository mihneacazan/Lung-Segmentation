"""
Attention U-Net Model for Lung Tumor Segmentation.

Uses MONAI's AttentionUnet architecture which adds attention gates at each
decoder level, allowing the network to focus on relevant regions (tumor)
while suppressing irrelevant features (healthy tissue, air).

Usage:
    from src.models.attention_unet import build_attention_unet
"""

import torch
from monai.networks.nets import AttentionUnet


def build_attention_unet(in_channels=1, out_channels=1, spatial_dims=2,
                         channels=None):
    """
    Builds a 2D Attention U-Net model using MONAI.
    
    The attention gates learn to weight feature maps based on the skip
    connections, enabling the decoder to focus on tumor-relevant features.
    
    Args:
        in_channels (int): Number of input channels (1 for 2D, 3/5 for 2.5D).
        out_channels (int): Number of output channels (1 for binary segmentation).
        spatial_dims (int): 2 for 2D segmentation.
        channels (tuple|None): Filters per encoder level. Defaults to the
            benchmark width; `strides` follows its length.
        
    Returns:
        MONAI AttentionUnet model.
    """
    channels = tuple(channels) if channels else (16, 32, 64, 128, 256)
    model = AttentionUnet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=(2,) * (len(channels) - 1),
        dropout=0.1
    )
    return model


if __name__ == "__main__":
    print("=== Testing Attention U-Net Architecture ===")
    model = build_attention_unet(in_channels=1, out_channels=1)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    test_input = torch.randn(2, 1, 192, 192)
    test_output = model(test_input)
    
    print(f"  Input shape:  {test_input.shape}")
    print(f"  Output shape: {test_output.shape}")
    print(f"  Parameters:   {trainable_params:,} trainable / {total_params:,} total")
