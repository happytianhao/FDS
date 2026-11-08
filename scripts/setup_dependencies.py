"""Fetch pinned external repositories without modifying existing checkouts."""

import argparse
import subprocess
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checkout(name):
    spec = tomllib.loads((ROOT / "configs/dependencies.toml").read_text())[name]
    target = ROOT / "third_party" / name
    if target.exists():
        head = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(target), "status", "--porcelain"], text=True
        ).strip()
        if head != spec["revision"] or dirty:
            raise RuntimeError(
                f"{target} differs from the pin or has local changes; left untouched"
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".fetch-{name}-", dir=target.parent) as temporary:
        checkout_path = Path(temporary) / name
        subprocess.run(["git", "init", str(checkout_path)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(checkout_path), "remote", "add", "origin", spec["url"]], check=True
        )
        subprocess.run(
            ["git", "-C", str(checkout_path), "fetch", "--depth=1", "origin", spec["revision"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout_path), "checkout", "--detach", "FETCH_HEAD"], check=True
        )
        checkout_path.rename(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Repository names from configs/dependencies.toml")
    names = parser.parse_args().names or ["robomme_plus"]
    available = tomllib.loads((ROOT / "configs/dependencies.toml").read_text())
    if set(names) - available.keys():
        parser.error("unknown repository name")
    for name in names:
        print(checkout(name))
