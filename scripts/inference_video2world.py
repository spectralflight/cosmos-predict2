#!/usr/bin/env -S uv run --script
# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate",
#   "cosmos-guardrail",
#   "diffusers>=0.34.0",
#   "rich",
#   "pyyaml",
#   "transformers",
# ]
# [tool.uv]
# exclude-newer = "2025-08-15T00:00:00Z"
# override-dependencies = ["peft>=0.15.0"]
# ///

import torch
import diffusers
import argparse
from rich import print
import pathlib
import yaml

ROOT = pathlib.Path(__file__).parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Output directory")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt")
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--video", type=str, help="Path to video")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Predict2-2B-Video2World",
        help="Model name or path (Cosmos-Predict2: https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959)",
    )
    parser.add_argument("--revision", type=str, help="Model revision (branch name, tag name, or commit id)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # TODO: Move to config
    fps = 16
    seed = 42

    prompt = args.prompt
    if args.negative_prompt is not None:
        negative_prompt = args.negative_prompt
    else:
        negative_prompt = yaml.safe_load(open(f"{ROOT}/prompts/default.yaml", "rb"))["negative_prompt"]
    if args.image is not None:
        image = diffusers.utils.load_image(args.image)
    else:
        image = None
    if args.video is not None:
        video = diffusers.utils.load_video(args.video)
    else:
        video = None

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = diffusers.Cosmos2VideoToWorldPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        revision=args.revision,
    )
    pipe.to("cuda")

    output = pipe(
        image=image,
        video=video,
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=torch.Generator("cuda").manual_seed(seed),
        fps=fps,
    ).frames[0]
    diffusers.utils.export_to_video(output, "output.mp4", fps=fps)


if __name__ == "__main__":
    main()
