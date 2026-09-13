"""Pick a Sol-Attn beta from a directory of finished runs.

Answers one question: which beta is at least as good as Sparge while being faster? Quality is
measured against a dense run of the same quantization recipe, because that is the ceiling a routed
backend is trying to reach -- comparing against bf16 dense would fold in quantization error that no
beta can remove. Sparge is the bar rather than the reference, since the goal is to match it, not to
resemble it.

Reads runs produced by separate launches rather than driving them, which is how the sweep is
actually done: one generation per launch, into a directory whose name records the recipe and beta.

  python3 benchmarks/sol_beta_report.py --outputs /outputs --prefix wan2_2.quantgemm_

Directory names are expected to look like <prefix><recipe><beta><suffix>, e.g.
wan2_2.quantgemm_fp8solattn-025.gfx942, where -025 is beta=-0.25. A run with no beta in its name is
a dense or Sparge reference.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_metrics import find_video, load_video, measured_timings, metrics

# quantgemm_fp8solattn-025 -> ("quantgemm_fp8", -0.25). Anything at all may precede "solattn", so
# that a looser --prefix still parses: an unquantized run is unlikely to be named like a quantized
# one, and the prefix has to be loose enough to match both to compare them.
#
# The digits are written without a decimal point: the first is units and the rest are the fraction,
# so 05 is 0.5, 025 is 0.25 and 10 is 1.0.
_RUN = re.compile(r"^(?P<recipe>.*?)solattn(?P<sign>[+-])(?P<digits>\d+)$")


def _parse_beta(sign: str, digits: str) -> float:
    whole = int(digits[0])
    fraction = int(digits[1:]) / 10 ** len(digits[1:]) if len(digits) > 1 else 0.0
    return (whole + fraction) * (-1.0 if sign == "-" else 1.0)


def _classify(name: str, prefix: str, suffix: str):
    """(kind, recipe, beta) for a run directory name, or None when it is not one of ours."""
    if not name.startswith(prefix):
        return None
    stem = name[len(prefix):]
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    match = _RUN.match(stem)
    if match:
        return (
            "sol",
            match.group("recipe"),
            _parse_beta(match.group("sign"), match.group("digits")),
        )
    if "sparge" in stem:
        return ("sparge", stem, None)
    if "solattn" not in stem:
        return ("dense", stem, None)
    return None


def _seconds(directory: str):
    """(mean seconds, iterations kept) for a run, or (None, 0).

    The count is reported because a single-iteration run has no warmup to discard, so its time
    carries MIOpen's solver search and the allocator's first-touch growth. Those are worth several
    seconds, which is larger than the differences this report is asked to rank.
    """
    path = os.path.join(directory, "timings.json")
    if not os.path.exists(path):
        return None, 0
    with open(path) as handle:
        timings = json.load(handle)
    if not timings:
        return None, 0
    kept = measured_timings(timings)
    return sum(kept) / len(kept), len(timings)


def _collect(root: str, prefix: str, suffix: str):
    runs = []
    for name in sorted(os.listdir(root)):
        directory = os.path.join(root, name)
        if not os.path.isdir(directory):
            continue
        classified = _classify(name, prefix, suffix)
        if classified is None:
            continue
        kind, recipe, beta = classified
        seconds, iterations = _seconds(directory)
        runs.append({
            "name": name,
            "kind": kind,
            "recipe": recipe,
            "beta": beta,
            "video": find_video(directory),
            "seconds": seconds,
            "iterations": iterations,
        })
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="/outputs")
    parser.add_argument("--prefix", default="wan2_2.quantgemm_")
    parser.add_argument("--suffix", default=".gfx942")
    parser.add_argument(
        "--reference",
        help="Run directory to measure quality against. Defaults to the dense run whose recipe "
             "matches the Sol runs being scored.",
    )
    parser.add_argument(
        "--bar",
        help="Run directory whose quality and latency must be matched. Defaults to the Sparge run.",
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Also score perceptual similarity, and use it for the recommendation. Slower, and "
             "downloads AlexNet weights on first use, but it is the metric worth deciding on: "
             "PSNR against a dense reference measures how closely a run tracked that particular "
             "trajectory, which is not the same as how good it looks.",
    )
    parser.add_argument("--json", help="Write the full table here.")
    args = parser.parse_args()

    runs = _collect(args.outputs, args.prefix, args.suffix)
    if not runs:
        raise SystemExit(f"no runs matching {args.prefix}*{args.suffix} under {args.outputs}")

    def _by_name(name, what):
        """Resolve a run by directory name or path, failing loudly rather than scoring the wrong one."""
        wanted = os.path.basename(name.rstrip("/"))
        found = next((r for r in runs if r["name"] == wanted), None)
        if found is None:
            raise SystemExit(
                f"--{what} {name!r} matched no run. Available:\n  "
                + "\n  ".join(r["name"] for r in runs)
            )
        return found

    if args.reference:
        reference = _by_name(args.reference, "reference")
    else:
        candidates = [r for r in runs if r["kind"] == "dense"]
        if len(candidates) > 1:
            # Two dense runs answer different questions -- one isolates routing error, the other
            # gives the whole error budget -- so guessing would quietly pick a different meaning.
            raise SystemExit(
                "Several dense runs could serve as the reference, so pass --reference to say "
                "which:\n  " + "\n  ".join(r["name"] for r in candidates)
            )
        reference = candidates[0] if candidates else None

    bar = _by_name(args.bar, "bar") if args.bar else next(
        (r for r in runs if r["kind"] == "sparge"), None
    )

    print(f"found {len(runs)} runs under {args.outputs}")
    for run in runs:
        state = "ok" if run["video"] else "NO VIDEO YET"
        beta = "-" if run["beta"] is None else f"{run['beta']:+.3f}"
        secs = "-" if run["seconds"] is None else f"{run['seconds']:.2f}s"
        print(f"  {run['name']:52s} {run['kind']:7s} beta={beta:>7s} {secs:>9s}  {state}")
    print()

    if reference is None or reference["video"] is None:
        print(
            "No dense reference with a video yet, so quality cannot be scored. Produce the dense "
            "run of the same recipe (a directory without 'solattn' in its name) and re-run.\n"
            "Latency above is still usable: compare each beta against the Sparge run."
        )
        if bar is not None and bar["seconds"] is not None:
            faster = [
                r for r in runs
                if r["kind"] == "sol" and r["seconds"] and r["seconds"] < bar["seconds"]
            ]
            print(f"\nSparge is {bar['seconds']:.2f}s. Betas already faster than it:")
            for run in sorted(faster, key=lambda r: r["seconds"]):
                gain = bar["seconds"] / run["seconds"]
                print(
                    f"  {run['recipe']:10s} beta={run['beta']:+.3f}  "
                    f"{run['seconds']:.2f}s  {gain:.3f}x vs Sparge"
                )
        return

    print(f"reference (quality ceiling): {reference['name']}")
    if bar is not None:
        print(f"bar to beat:                {bar['name']}")

    single = [r["name"] for r in runs if r["iterations"] == 1]
    if single:
        print(
            f"\nWARNING: {len(single)} of {len(runs)} runs timed a single iteration, so no warmup "
            "was discarded.\n"
            "         Those times include MIOpen's solver search and first-touch allocation, worth\n"
            "         seconds. Treat latency gaps below ~10% as noise until these are re-run with\n"
            "         --num_iterations 2 or more, or with --warmup_calls 1. Quality is unaffected."
        )
    print()

    lpips_metric = None
    if args.lpips:
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True, sync_on_compute=False
            )
        except (ImportError, RuntimeError) as error:
            print(f"perceptual scoring unavailable ({error}); continuing without it\n")

    reference_video = load_video(reference["video"])
    scored = []
    for run in runs:
        if run["video"] is None or run["name"] == reference["name"]:
            continue
        measured = metrics(reference_video, load_video(run["video"]), lpips_metric)
        scored.append({**run, **measured})

    lpips_key = "lpips_alex_frame_stride_8"
    has_lpips = all(lpips_key in row for row in scored) and bool(scored)

    bar_row = next((r for r in scored if r["name"] == (bar or {}).get("name")), None)

    lpips_header = f" {'lpips':>7}" if has_lpips else ""
    print(
        f"{'run':>10} {'beta':>7} {'psnr':>7} {'ssim':>7}{lpips_header} "
        f"{'mae':>7} {'sec':>7} {'vs bar':>8}"
    )
    for row in sorted(
        scored, key=lambda r: (r["kind"] != "sparge", r["beta"] if r["beta"] is not None else 0)
    ):
        beta = "-" if row["beta"] is None else f"{row['beta']:+.3f}"
        secs = "-" if row["seconds"] is None else f"{row['seconds']:.2f}"
        speed = "-"
        if bar_row and bar_row["seconds"] and row["seconds"]:
            speed = f"{bar_row['seconds'] / row['seconds']:.3f}x"
        lpips_cell = f" {row[lpips_key]:7.4f}" if has_lpips else ""
        print(
            f"{row['recipe']:>10} {beta:>7} {row['psnr']:7.2f} {row['ssim']:7.4f}"
            f"{lpips_cell} {row['mae']:7.4f} {secs:>7} {speed:>8}"
        )

    if bar_row is None:
        print("\nNo Sparge run to compare against, so no recommendation.")
    else:
        bar_lpips = f", lpips {bar_row[lpips_key]:.4f}" if has_lpips else ""
        print(
            f"\nbar: psnr {bar_row['psnr']:.2f}, ssim {bar_row['ssim']:.4f}{bar_lpips}, "
            f"{bar_row['seconds']:.2f}s"
        )
        def _at_least_as_good(row):
            """Perceptual similarity decides when we have it, since it is what the eye reports."""
            if has_lpips:
                return row[lpips_key] <= bar_row[lpips_key]
            return row["psnr"] >= bar_row["psnr"] and row["ssim"] >= bar_row["ssim"]

        qualifying = [
            r for r in scored
            if r["kind"] == "sol"
            and _at_least_as_good(r)
            and r["seconds"] and bar_row["seconds"]
            and r["seconds"] < bar_row["seconds"]
        ]
        if not qualifying:
            print(
                "No beta is both at least as good as Sparge and faster than it. Ranked by the "
                "criterion above, best first:"
            )
            order = (
                (lambda r: r[lpips_key]) if has_lpips else (lambda r: -r["psnr"])
            )
            for row in sorted((r for r in scored if r["kind"] == "sol"), key=order)[:3]:
                cell = f" lpips {row[lpips_key]:.4f}" if has_lpips else ""
                verdict = (
                    "clears quality, too slow"
                    if _at_least_as_good(row)
                    else "below quality bar"
                )
                print(
                    f"  beta={row['beta']:+.3f} psnr {row['psnr']:.2f} "
                    f"ssim {row['ssim']:.4f}{cell} {row['seconds']:.2f}s  ({verdict})"
                )
        else:
            best = min(qualifying, key=lambda r: r["seconds"])
            print(
                f"recommended: {best['recipe']} beta={best['beta']:+.3f} -- psnr "
                f"{best['psnr']:.2f} (bar {bar_row['psnr']:.2f}), ssim {best['ssim']:.4f} "
                f"(bar {bar_row['ssim']:.4f}), {best['seconds']:.2f}s "
                f"({bar_row['seconds'] / best['seconds']:.3f}x Sparge)"
            )
            if len(qualifying) > 1:
                print("other betas that clear the bar:")
                for row in sorted(qualifying, key=lambda r: r["seconds"]):
                    if row is not best:
                        print(
                            f"  beta={row['beta']:+.3f} psnr {row['psnr']:.2f} "
                            f"ssim {row['ssim']:.4f} {row['seconds']:.2f}s"
                        )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "reference": reference["name"],
                    "bar": (bar or {}).get("name"),
                    "rows": scored,
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
