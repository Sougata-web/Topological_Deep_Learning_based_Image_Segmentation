from __future__ import annotations

import argparse
import subprocess
import sys


def run(command: list[str], allow_failure: bool = False) -> bool:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        if allow_failure:
            print(
                "Command failed as expected for an unavailable optional "
                "backend.",
                flush=True,
            )
            return False
        raise SystemExit(result.returncode)

    return True


def main(config_path: str, device: str | None, include_ph: bool) -> None:
    python = sys.executable
    device_arguments = [] if device is None else ["--device", device]

    run(
        [
            python,
            "scripts/prepareisbi.py",
            "--config",
            config_path,
        ]
    )

    completed_methods: list[str] = []

    for method in ("baseline", "dect"):
        run(
            [
                
                python,
                "scripts/train.py",
                "--config",
                config_path,
                "--method",
                method,
                *device_arguments,
            ]
        )
        completed_methods.append(method)

    if include_ph:
        succeeded = run(
            [
                python,
                "scripts/train.py",
                "--config",
                config_path,
                "--method",
                "ph",
                *device_arguments,
            ],
            allow_failure=True,
        )
        if succeeded:
            completed_methods.append("ph")

    for method in completed_methods:
        run(
            [
                python,
                "scripts/evaluate.py",
                "--config",
                config_path,
                "--method",
                method,
                *device_arguments,
            ]
        )

    run(
        [
            python,
            "scripts/benchmarktopology.py",
            "--config",
            config_path,
            "--method",
            "dect",
            *device_arguments,
        ]
    )

    if "ph" in completed_methods:
        run(
            [
                python,
                "scripts/benchmarktopology.py",
                "--config",
                config_path,
                "--method",
                "ph",
                *device_arguments,
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/isbi.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--include-ph",
        action="store_true",
        help=(
            "Attempt PH training. It fails loudly until the GUDHI "
            "adapter is implemented."
        ),
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.device, arguments.include_ph)
