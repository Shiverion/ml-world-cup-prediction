from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(args: list[str]) -> None:
    print(f"> {' '.join(args)}")
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    run_step([sys.executable, "scripts/download_data.py"])
    run_step([sys.executable, "scripts/run_analysis.py", "--live", "--profile", "dev"])


if __name__ == "__main__":
    main()
