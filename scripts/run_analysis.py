from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from worldcup_prediction.pipeline import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the World Cup prediction analysis pipeline.")
    parser.add_argument("--data-config", default=ROOT / "configs" / "data_config.yaml")
    parser.add_argument("--model-config", default=ROOT / "configs" / "model_config.yaml")
    parser.add_argument("--backtest-config", default=ROOT / "configs" / "backtest_config.yaml")
    parser.add_argument("--tournament-config", default=ROOT / "configs" / "tournament_2026.yaml")
    parser.add_argument("--live", action="store_true", help="Lock completed 2026 matches and write live probabilities.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Simulation runtime profile from tournament config, for example dev, local, or publication.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        outputs = run_analysis(
            data_config_path=args.data_config,
            model_config_path=args.model_config,
            backtest_config_path=args.backtest_config,
            tournament_config_path=args.tournament_config,
            root=ROOT,
            live=args.live,
            simulation_profile=args.profile,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Pipeline failed: {exc}") from exc
    print("Analysis pipeline completed.")
    for name, path in outputs.items():
        print(f"{name}: {path if path is not None else 'skipped'}")


if __name__ == "__main__":
    main()
