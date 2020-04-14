from time import time_ns
from typing import Any

import numpy as np
import torch
from pytorch_pretrained_biggan import BigGAN
from pytorch_pretrained_biggan import one_hot_from_int
from pytorch_pretrained_biggan import truncated_noise_sample
from torch import nn

from hippo7_app.hippo7_backend import get_asset_folder


class BeatTimeMessage:
    def __init__(
        self,
        counter=0,
        isBeat=False,
        time=0.0,
        delta_time=0.0,
        local_time=0.0,
        bpm=120,
        speed=1,
    ):
        self.counter = counter
        self.isBeat = isBeat
        self.time = time
        self.delta_time = delta_time
        self.local_time = local_time
        self.bpm = bpm
        self.speed = speed

    def __repr__(self):
        return (
            f"t:{self.time:03.4f} - dt: {self.delta_time:03.4f} - "
            f"lt:{self.local_time:03.4f} - c:{self.counter} - "
            f"isBeat: {self.isBeat} - bpm:{self.bpm} - speed:{self.speed}"
        )


def toTensor(self, value):
    return torch.tensor([value], device=self.device)


def toNumpy(self, value):
    return np.array(value, dtype=float)


def create_noise_vector(
    batch_size: int = 1,
    dim_z: int = 128,
    truncation: float = 1.0,
    device: str = "cpu",
    seed: Any = None,
):
    # 643 - weird faces
    noise_vector = truncated_noise_sample(batch_size, dim_z, truncation, seed)
    noise_vector = torch.from_numpy(noise_vector)
    return noise_vector.to(device)


def create_class_vector(x, device, batch_size: int = 1):
    class_vector = one_hot_from_int(x, batch_size)
    return torch.from_numpy(class_vector).to(device)


def time_generator():
    while True:
        yield time_ns() / (10 ** 9)


def fixed_time_generator(framelength=1.0 / 30):
    """
    Using a fixed length for each frame, enables a easy solution for video recording or similar tasks.

    Args:
        framelength: Length of a single frame, where 1.0/30 translates to 30 frames per second.
    """
    time = 0
    while True:
        time += framelength
        yield time


def generate_model_file(model_type, device):
    model: nn.Module = BigGAN.from_pretrained(model_type)
    model.to(device)
    model.eval()
    truncation = torch.tensor([1.0]).to(device)

    for p in model.parameters():
        p.require_grads = False
    # Remove the spectral norm from all layers, we can do this because we only do inference
    for module in model.modules():
        try:
            torch.nn.utils.remove_spectral_norm(module)
        except (AttributeError, ValueError):
            pass
    # Do a JIT precompute for additional speedup
    model = torch.jit.trace(
        model,
        (
            create_noise_vector(device=device),
            create_class_vector(643, device=device),
            truncation,
        ),
    )

    torch.jit.save(
        model,
        (get_asset_folder() / "networks" / (model_type + "-" + device)).__str__(),
    )
