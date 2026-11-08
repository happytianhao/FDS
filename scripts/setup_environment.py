"""Create an independent FDS simulator environment for one benchmark."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = {
    "robomme_plus": ROOT / ".venv",
    "robomme": ROOT / ".venv-robomme",
}
SOURCES = {
    "robomme_plus": ROOT / "third_party" / "robomme_plus",
    "robomme": ROOT / "third_party" / "robomme",
}


def python_for(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if shutil.which("cmd.exe") else "bin/python")


def setup(name: str) -> Path:
    if shutil.which("uv") is None:
        raise RuntimeError("uv is required; install it before running setup_environment.py")
    source = SOURCES[name]
    if not (source / ".git").is_dir():
        raise RuntimeError(
            f"Missing {source}. Fetch it first with: "
            f"python scripts/setup_dependencies.py {name}"
        )
    environment = ENVIRONMENTS[name]
    python = python_for(environment)
    if not python.exists():
        subprocess.run(
            ["uv", "venv", str(environment), "--python", "3.11"], cwd=ROOT, check=True
        )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-e",
            ".[api]",
            "-e",
            str(source),
        ],
        cwd=ROOT,
        check=True,
    )
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark",
        choices=["robomme_plus", "robomme", "all"],
        default="robomme_plus",
        nargs="?",
        help="Benchmark environment to create; default: robomme_plus",
    )
    args = parser.parse_args()
    names = list(ENVIRONMENTS) if args.benchmark == "all" else [args.benchmark]
    for name in names:
        print(f"{name}: {setup(name)}")


if __name__ == "__main__":
    main()
