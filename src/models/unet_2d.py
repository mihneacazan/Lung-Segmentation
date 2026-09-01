import torch
from monai.networks.nets import UNet

DEFAULT_CHANNELS = (16, 32, 64, 128, 256)


def build_unet_2d(in_channels=1, out_channels=1, channels=None):
    """
    Builds a 2D U-Net model using MONAI for binary medical image segmentation.

    Args:
        in_channels (int): 1 for 2D, or n_adjacent for a 2.5D model.
        out_channels (int): 1 for binary segmentation.
        channels (tuple|None): Filters per encoder level. The default is
            1.62M parameters, which is small for a segmentation network and was
            never varied across the benchmark; doubling it to
            (32, 64, 128, 256, 512) gives 6.49M. `strides` is derived from its
            length, so a shorter tuple is a shallower network.
    """
    channels = tuple(channels) if channels else DEFAULT_CHANNELS
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=(2,) * (len(channels) - 1),
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
