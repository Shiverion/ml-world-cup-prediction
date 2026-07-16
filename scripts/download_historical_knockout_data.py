from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from worldcup_prediction.historical_knockout import (
    HISTORICAL_WORLD_CUP_YEARS,
    world_cup_knockout_results_from_payload,
)


OPENFOOTBALL_WORLD_CUP_URL_TEMPLATE = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a reproducible historical World Cup knockout archive.")
    parser.add_argument(
        "--output",
        default=ROOT / "data" / "external" / "world_cup_knockout_results_2002_2022.csv",
    )
    parser.add_argument("--url-template", default=OPENFOOTBALL_WORLD_CUP_URL_TEMPLATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    for year in HISTORICAL_WORLD_CUP_YEARS:
        url = str(args.url_template).format(year=year)
        with urlopen(url, timeout=30) as response:
            payload = json.load(response)
        frames.append(world_cup_knockout_results_from_payload(payload, year, source_url=url))

    archive = pd.concat(frames, ignore_index=True)
    counts = archive.groupby("year").size()
    if len(archive) != 15 * len(HISTORICAL_WORLD_CUP_YEARS) or not counts.eq(15).all():
        raise ValueError(f"Expected 15 decisive knockout ties per World Cup, got: {counts.to_dict()}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.to_csv(output, index=False)
    print(f"historical knockout ties: {len(archive)} -> {output}")
    print(f"penalty-decided ties: {(archive['winner_method'] == 'penalties').sum()}")


if __name__ == "__main__":
    main()
