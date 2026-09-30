"""Create a checksum manifest for local, uncommitted dataset files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.utils import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="data/dataset_checksums.json")
    args = parser.parse_args()
    root = Path(args.data_dir)
    manifest = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(root.glob("*.csv"))
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote checksums for {len(manifest)} files to {output}")


if __name__ == "__main__":
    main()
