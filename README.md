# Stable_Diffusion

A PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) with a U-Net backbone. This project follows the DDPM training and sampling procedure, inspired by the Hugging Face DDPM example.

**Repository:** [https://github.com/wrafique007/Stable_Diffusion](https://github.com/wrafique007/Stable_Diffusion)

---

## Table of Contents

- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Sampling](#sampling)
  - [Customization](#customization)
- [Model Architecture](#model-architecture)
- [References](#references)
- [License](#license)

---

## Features

- **U-Net architecture** with residual blocks, attention, and sinusoidal time embeddings.
- **Beta schedules**: linear, cosine, quadratic, and sigmoid noise schedules.
- **Forward diffusion** (`q_sample`) and **reverse sampling** (`p_sample`) implementations.
- **Training loop** with Huber loss and periodic image sampling.
- **Dataset integration** via Hugging Face `datasets` (e.g., Fashion-MNIST).

---

## File Structure

```
Stable_Diffusion/
├── helpers.py          # Utility functions and modules (Residual, Upsample, Downsample, etc.)
├── unet.py             # U-Net model, diffusion schedules, training & sampling loops
└── requirements.txt    # Python dependencies
```

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Dependencies include:

- `torch`
- `torchvision`
- `matplotlib`
- `tqdm`
- `einops`
- `requests`
- `datasets`

---

## Usage

### Training

The training script is embedded in `unet.py`. It loads the Fashion-MNIST dataset, builds a `Unet` model, and optimizes it using Adam with a Huber loss. To start training, run:

```bash
python unet.py
```

Key training parameters (modify inside `unet.py`):

| Parameter | Value |
|-----------|-------|
| Image size | 28 |
| Channels | 1 |
| Batch size | 128 |
| Timesteps | 300 |
| Epochs | 6 |
| Learning rate | 1e-3 |
| Loss type | Huber |

During training, generated samples are saved to `./results/` every 1000 steps.

### Sampling

The `sample` function generates images from pure noise. After training, you can call:

```python
from unet import sample, Unet

model = Unet(dim=28, channels=1, dim_mults=(1, 2, 4))
images = sample(model, image_size=28, batch_size=16, channels=1)
```

### Customization

- **Beta schedule**: Change `betas` in `unet.py` to use `cosine_beta_schedule`, `quadratic_beta_schedule`, or `sigmoid_beta_schedule`.
- **Model dimensions**: Adjust `dim`, `dim_mults`, `channels`, and `resnet_block_groups` when instantiating `Unet`.
- **Dataset**: Replace the `load_dataset` call with your own dataset and update the transform pipeline.

---

## Model Architecture

The `Unet` class implements a standard diffusion U-Net:

- **Initial convolution** with 1×1 kernel.
- **Downsampling path** with ResNet blocks, linear attention, and downsampling layers.
- **Middle block** with ResNet blocks and full attention.
- **Upsampling path** with ResNet blocks, linear attention, and upsampling layers.
- **Skip connections** between corresponding down/up resolutions.
- **Time embedding** via sinusoidal position embeddings and MLP.

For details, see `unet.py` (lines 58–95).

---

## References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)
- [Hugging Face DDPM Example](https://github.com/huggingface/diffusers/tree/main/examples/ddpm)

---

## License

No license is specified in the repository. Please contact the repository owner for licensing information.
