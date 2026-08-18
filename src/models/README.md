# Model architectures

Three segmentation networks, all built on MONAI, all reachable through one
factory so that `--model_type` selects between them without anything else in the
run changing.

| File | Builder | `--model_type` |
|---|---|---|
| `unet_2d.py` | `build_unet_2d` | `unet` |
| `attention_unet.py` | `build_attention_unet` | `attention_unet` |
| `segresnet.py` | `build_segresnet` | `segresnet` |
| `factory.py` | `build_model` | — dispatches to the above |

---

## Why a factory

Comparing architectures is only meaningful if nothing else differs between the
runs. Routing all three through `build_model()` means every architecture gets the
same training loop, the same loss, the same sampling, the same evaluation code,
and the same checkpoint selection rule.

The indirection also makes the selection testable. An architecture reachable only
by editing `train.py` is an architecture that can silently go unexercised, so
`tests/test_training_smoke.py` trains all three end to end on synthetic data,
which keeps every builder on a path that runs in CI rather than only when someone
remembers to try it.

---

## Shared configuration

All three are built for **2D axial slices at 192 × 192**, binary output
(1 channel, logits — the sigmoid is applied in the evaluation code, not in the
model), and dropout 0.1.

`in_channels` is not fixed at 1: it takes the value of `--n_adjacent`, so the
same builders serve a plain 2D model (1 channel) and a 2.5D model that stacks 3
or 5 consecutive slices as channels.

---

## The three networks

**U-Net** (`unet_2d.py`) — the baseline. Channels (16, 32, 64, 128, 256) across
5 resolution levels, `num_res_units=2`. About 1.6 M trainable parameters. Skip
connections carry high-resolution encoder features to the decoder, which is what
keeps tumour boundaries sharp after four downsampling steps.

**Attention U-Net** (`attention_unet.py`) — same channel progression, plus
attention gates on each skip connection. The gates learn to weight the encoder
features before they are concatenated, which is intended to suppress healthy
tissue and air in favour of the lesion. Relevant here because the target occupies
well under 1% of the volume.

**SegResNet** (`segresnet.py`) — residual encoder-decoder with group
normalization, `blocks_down=(1, 2, 2, 4)`, `blocks_up=(1, 1, 1)`,
`init_filters=16`. Deeper than the U-Net, with residual connections for gradient
flow.

---

## Running a quick shape check

Each file has a `__main__` block that builds the model, pushes a
`(2, 1, 192, 192)` tensor through it, and prints the output shape and parameter
count:

```bash
python -m src.models.unet_2d
python -m src.models.attention_unet
python -m src.models.segresnet
```
