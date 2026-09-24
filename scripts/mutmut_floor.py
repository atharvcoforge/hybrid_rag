"""Fail the process when the killed ratio is under evals/mutmut_floor.txt."""

import json
import sys
from pathlib import Path

STATS = Path("mutants/mutmut-cicd-stats.json")
FLOOR = Path("evals/mutmut_floor.txt")


def main() -> None:
    if not STATS.exists():
        raise SystemExit(f"missing {STATS}; run mutmut run first")
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    killed = int(stats["killed"])
    survived = int(stats["survived"])
    decided = killed + survived
    if decided == 0:
        raise SystemExit("mutmut decided nothing")
    ratio = killed / decided
    floor = float(FLOOR.read_text(encoding="utf-8").strip())
    print(
        f"killed={killed} survived={survived} ratio={ratio:.3f} floor={floor:.3f} "
        f"total={stats.get('total')}"
    )
    if ratio + 1e-12 < floor:
        raise SystemExit(f"mutation score {ratio:.3f} is under {floor:.3f}")


if __name__ == "__main__":
    main()
    sys.exit(0)
