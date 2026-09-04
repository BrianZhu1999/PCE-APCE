from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an HDF5 layout without loading arrays")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items: list[dict[str, object]] = []
    with h5py.File(args.path, "r") as handle:
        def visit(name: str, value: h5py.Group | h5py.Dataset) -> None:
            if len(items) >= args.max_items or name.count("/") >= args.max_depth:
                return
            item: dict[str, object] = {
                "path": name,
                "kind": "dataset" if isinstance(value, h5py.Dataset) else "group",
            }
            if isinstance(value, h5py.Dataset):
                item["shape"] = list(value.shape)
                item["dtype"] = str(value.dtype)
                item["chunks"] = None if value.chunks is None else list(value.chunks)
            items.append(item)

        handle.visititems(visit)
        result = {
            "path": str(args.path.resolve()),
            "root_keys": list(handle.keys()),
            "root_attributes": {str(key): str(value) for key, value in handle.attrs.items()},
            "items": items,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
