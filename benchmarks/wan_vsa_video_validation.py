"""Run paired Dense/VSA Wan validations without reloading model weights."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xfuser import xFuserArgs
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.runner import xFuserModelRunner

from video_metrics import measured_timings, metrics


DEFAULT_PROMPTS = (
    "Two anthropomorphic cats in boxing gear fight on a spotlighted stage.",
    "A red sailboat crosses a stormy ocean at sunset, cinematic camera.",
    "A robot chef chops vegetables in a bright modern kitchen.",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))

    config = xFuserArgs(
        model="Wan2.1-T2V", attention_backend="AITER_VSA",
        cross_attention_backend="AITER", dit_parallel_size=4,
        ulysses_degree=4, ring_degree=1, height=args.height,
        width=args.width, num_frames=args.frames,
        num_inference_steps=args.steps, prompt=args.prompts[0],
        seed=args.seeds[0], guidance_scale=6.0, flow_shift=8.0,
        output_directory="/tmp/wan-vsa-validation", warmup_steps=0,
        num_iterations=args.iterations,
        vsa_drop_rates=[0.25, 0.40], vsa_prob_threshold=0.9,
        input_images=[],
    )
    runner = xFuserModelRunner(vars(config))
    runner.model.settings.model_name = args.model_path
    first_input = runner.preprocess_args(vars(config))
    runner.initialize(first_input)
    runtime = get_runtime_state()
    rows = []
    lpips_metric = None
    if rank == 0:
        try:
            from torchmetrics.image.lpip import (
                LearnedPerceptualImagePatchSimilarity,
            )
            lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True, sync_on_compute=False
            )
        except (ImportError, RuntimeError):
            pass

    for prompt in args.prompts:
        for seed in args.seeds:
            run_args = dict(first_input, prompt=prompt, seed=seed)
            runtime.attention_backend = AttentionBackendType.AITER
            dense, dense_time = runner.run(run_args)
            runtime.attention_backend = AttentionBackendType.AITER_VSA
            sparse, sparse_time = runner.run(run_args)
            if rank == 0:
                dense_measured = measured_timings(dense_time)
                sparse_measured = measured_timings(sparse_time)
                dense_seconds = float(np.mean(dense_measured))
                sparse_seconds = float(np.mean(sparse_measured))
                rows.append({
                    "prompt": prompt,
                    "seed": seed,
                    "metrics": metrics(
                        dense.videos, sparse.videos, lpips_metric
                    ),
                    "dense_timings": dense_measured,
                    "vsa_timings": sparse_measured,
                    "dense_seconds": dense_seconds,
                    "vsa_seconds": sparse_seconds,
                    "speedup": dense_seconds / sparse_seconds,
                })

    if rank == 0:
        summary = {
            "mean_psnr": float(np.mean([r["metrics"]["psnr"] for r in rows])),
            "mean_ssim": float(np.mean([r["metrics"]["ssim"] for r in rows])),
            "mean_speedup": float(np.mean([r["speedup"] for r in rows])),
        }
        if lpips_metric is not None:
            summary["mean_lpips_alex_frame_stride_8"] = float(np.mean([
                r["metrics"]["lpips_alex_frame_stride_8"] for r in rows
            ]))
        with open(args.output, "w") as handle:
            json.dump({"summary": summary, "rows": rows}, handle, indent=2)
    runner.cleanup()


if __name__ == "__main__":
    main()
