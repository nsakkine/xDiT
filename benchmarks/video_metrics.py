"""Video comparison metrics shared by the accuracy benchmarks.

Lifted out of wan_vsa_video_validation.py so a second benchmark cannot end up with a second,
subtly different SSIM. Numbers produced here are comparable across every benchmark that imports
it, which is the point: a beta sweep is only useful if its quality bar means the same thing as the
one the VSA runs were judged against.
"""

import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


def tensor_video(video) -> torch.Tensor:
    """A video, however the pipeline handed it back, as float (frames, channels, height, width)."""
    frames = video[0] if isinstance(video, list) and len(video) == 1 else video
    array = np.stack([np.asarray(frame) for frame in frames])
    result = torch.from_numpy(array).float()
    if result.max() > 1.0:
        result.div_(255.0)
    return result.permute(0, 3, 1, 2).contiguous()


def load_video(path: str) -> torch.Tensor:
    """Decode an mp4 off disk into the same layout tensor_video produces.

    Lets a report score runs that were produced by separate processes, which is how the sweep is
    actually driven: one generation per launch, compared afterwards.
    """
    import av

    container = av.open(path)
    try:
        frames = [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(video=0)
        ]
    finally:
        container.close()
    if not frames:
        raise ValueError(f"{path} decoded to no frames")
    array = torch.from_numpy(np.stack(frames)).float().div_(255.0)
    return array.permute(0, 3, 1, 2).contiguous()


def find_video(directory: str) -> str | None:
    """The single mp4 a run directory is expected to hold, or None if it has not produced one."""
    matches = sorted(glob.glob(os.path.join(directory, "*.mp4")))
    return matches[0] if matches else None


def ssim(reference: torch.Tensor, actual: torch.Tensor) -> float:
    channels = reference.shape[1]
    coords = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coords * coords) / (2 * 1.5 * 1.5))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    window = window.expand(channels, 1, 11, 11)
    mu_x = F.conv2d(reference, window, padding=5, groups=channels)
    mu_y = F.conv2d(actual, window, padding=5, groups=channels)
    sigma_x = F.conv2d(reference * reference, window, padding=5, groups=channels)
    sigma_y = F.conv2d(actual * actual, window, padding=5, groups=channels)
    sigma_xy = F.conv2d(reference * actual, window, padding=5, groups=channels)
    sigma_x -= mu_x.square()
    sigma_y -= mu_y.square()
    sigma_xy -= mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean())


def metrics(reference, actual, lpips_metric=None) -> dict:
    """Compare two videos given either as pipeline output or as already-loaded tensors."""
    if not isinstance(reference, torch.Tensor):
        reference = tensor_video(reference)
    if not isinstance(actual, torch.Tensor):
        actual = tensor_video(actual)
    mse = float((reference - actual).square().mean())
    result = {
        "mae": float((reference - actual).abs().mean()),
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "ssim": ssim(reference, actual),
    }
    if lpips_metric is not None:
        lpips_metric.reset()
        sampled_reference = F.interpolate(
            reference[::8], size=(224, 224), mode="bilinear",
            align_corners=False,
        )
        sampled_actual = F.interpolate(
            actual[::8], size=(224, 224), mode="bilinear",
            align_corners=False,
        )
        for start in range(0, len(sampled_reference), 2):
            lpips_metric.update(
                sampled_reference[start:start + 2],
                sampled_actual[start:start + 2],
            )
        result["lpips_alex_frame_stride_8"] = float(lpips_metric.compute())
    return result


def measured_timings(timings: list[float]) -> list[float]:
    """Discard the first iteration as warmup when repeats are available.

    The first call in a process pays MIOpen's convolution solver search and kernel compilation,
    which is a fixed cost of a few seconds and nothing to do with the attention backend under test.
    """
    return timings[1:] if len(timings) > 1 else timings
