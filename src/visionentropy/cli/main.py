from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visionentropy")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an experiment configuration.")
    run_parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")

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
        print(f"Experiment runner scaffold ready for: {args.config}")
        return

    parser.print_help()
