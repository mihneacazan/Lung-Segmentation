import torch
from monai.networks.nets import UNet

def build_unet_2d(in_channels=1, out_channels=1):
    """
    Builds a 2D U-Net model using MONAI for binary medical image segmentation.
    """
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1
    )
    return model

if __name__ == "__main__":
    print("=== Testing 2D U-Net Model Architecture ===")
    model = build_unet_2d()
    test_input = torch.randn(2, 1, 192, 192)
    test_output = model(test_input)
    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
