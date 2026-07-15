from __future__ import annotations

import argparse
from pathlib import Path

from visionentropy.datasets.registry import dataset_status, load_dataset_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visionentropy")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an experiment configuration.")
    run_parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")

    serve_parser = subparsers.add_parser("serve", help="Start the local dashboard API.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    dataset_parser = subparsers.add_parser("dataset", help="Inspect configured datasets.")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")

    dataset_subparsers.add_parser("list", help="List configured datasets and readiness.")

    status_parser = dataset_subparsers.add_parser("status", help="Show one dataset status.")
    status_parser.add_argument("name", help="Dataset name from configs/datasets.yaml.")

    inspect_parser = dataset_subparsers.add_parser("inspect", help="Inspect a sample from a dataset.")
    inspect_parser.add_argument("--name", required=True, choices=["synthetic_shapes", "skimage_examples"])
    inspect_parser.add_argument("--sample-index", type=int, default=0)
    inspect_parser.add_argument("--resize", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    inspect_parser.add_argument("--normalize", choices=["none", "zero_one", "standard"], default="none")
    inspect_parser.add_argument(
        "--representation",
        choices=["rgb", "grayscale", "lab", "red", "green", "blue"],
        default=None,
    )

    subparsers.add_parser("version", help="Show package version.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "version":
        from visionentropy import __package__

        print(__package__ or "visionentropy")
        return

    if args.command == "run":
        from visionentropy.pipeline import run_vertical_slice_from_config

        result = run_vertical_slice_from_config(args.config)
        print(f"Run complete: {result.metadata['experiment']}")
        print(f"Sample: {result.sample_id}")
        print(f"Output summary: {result.artifacts['summary']}")
        if result.metrics:
            print("Metrics:")
            for key, value in result.metrics.items():
                print(f"  {key}: {value:.4f}")
        return

    if args.command == "serve":
        from visionentropy.api.server import serve

        serve(host=args.host, port=args.port)
        return

    if args.command == "dataset":
        handle_dataset_command(args)
        return

    parser.print_help()


def handle_dataset_command(args: argparse.Namespace) -> None:
    specs = load_dataset_specs()

    if args.dataset_command == "list":
        for spec in specs.values():
            status = dataset_status(spec, project_root=Path.cwd())
            marker = "ready" if status.ready else "missing"
            root = status.root if status.root is not None else "-"
            print(f"{status.name:18} {marker:8} {status.mode:13} {root}")
        return

    if args.dataset_command == "status":
        spec = specs[args.name]
        status = dataset_status(spec, project_root=Path.cwd())
        print(f"Dataset: {status.title}")
        print(f"Name: {status.name}")
        print(f"Mode: {status.mode}")
        print(f"Ready: {status.ready}")
        print(f"Root: {status.root or '-'}")
        print(f"Message: {status.message}")
        if status.missing_paths:
            print("Missing paths:")
            for path in status.missing_paths:
                print(f"  - {path}")
        return

    if args.dataset_command == "inspect":
        sample = load_preview_sample(args.name, args.sample_index)
        sample = apply_preview_preprocessing(
            sample,
            resize_to=tuple(args.resize) if args.resize else None,
            normalization=args.normalize,
        )
        mask_shape = sample.mask.shape if sample.mask is not None else None
        print(f"Sample: {sample.sample_id}")
        print(f"Image shape: {sample.image.shape}")
        print(f"Mask shape: {mask_shape}")
        print(f"Label: {sample.label}")
        print(f"Metadata: {sample.metadata}")
        if args.representation:
            from visionentropy.representations import build_representation

            representation = build_representation(args.representation).transform(sample.image)
            print(f"Representation: {representation.name}")
            print(f"Representation shape: {representation.data.shape}")
            print(f"Representation channels: {representation.channels}")
        return

    raise SystemExit("Choose a dataset command: list, status, or inspect.")


def load_preview_sample(name: str, index: int):
    if name == "synthetic_shapes":
        from visionentropy.datasets.synthetic_shapes import SyntheticShapesDataset

        return SyntheticShapesDataset()[index]

    if name == "skimage_examples":
        from visionentropy.datasets.skimage_examples import SkimageExamplesDataset

        return SkimageExamplesDataset()[index]

    raise ValueError(f"Preview inspection is not supported for {name}.")


def apply_preview_preprocessing(sample, *, resize_to: tuple[int, int] | None, normalization: str):
    from visionentropy.preprocessing import ComposeTransforms, NormalizeImage, ResizeSample

    transforms = []
    if resize_to is not None:
        height, width = resize_to
        transforms.append(ResizeSample(height=height, width=width))
    if normalization != "none":
        transforms.append(NormalizeImage(mode=normalization))
    if not transforms:
        return sample
    return ComposeTransforms(transforms).transform(sample)
