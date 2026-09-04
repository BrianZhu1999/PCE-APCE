from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.wave_scenario_assets import WaveScenarioAssets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bad: list[str] = []
    for record in manifest["records"]:
        name = str(record["name"])
        try:
            assets = WaveScenarioAssets.load(Path(str(record["path"])))
            if assets.array_digest != record["array_digest"]:
                bad.append(name + ":digest")
            if assets.nx != 41 or assets.ensemble_size != 18:
                bad.append(name + ":shape")
        except Exception as exc:
            bad.append(name + ":" + type(exc).__name__)
    print("ASSET_VERIFY", len(manifest["records"]) - len(bad), "/", len(manifest["records"]), "BAD", bad)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
