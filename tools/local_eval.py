"""Play the public ARC-AGI-3 games offline and print the official score.

Usage (needs the competition data and a Python >= 3.12 env with arc_agi installed):
    python tools/local_eval.py --data <path to competition data> [--games ls20,ft09] [--max-actions 4000]

<path to competition data> is the extracted `kaggle competitions download arc-prize-2026-arc-agi-3`
folder (it contains ARC-AGI-3-Agents/ and environment_files/).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--games", default="", help="Comma-separated game prefixes (default: all)")
    parser.add_argument("--max-actions", type=int, default=None)
    args = parser.parse_args()

    os.environ["OPERATION_MODE"] = "offline"
    os.environ["ENVIRONMENTS_DIR"] = str(args.data / "environment_files")
    os.environ["RECORDINGS_DIR"] = str(ROOT / "submissions" / "recordings")

    # Same trick as the Kaggle notebook: work on a copy whose agents/__init__.py only
    # imports the base classes (the original pulls in langgraph, smolagents, ...).
    work = ROOT / "submissions" / "_work" / "ARC-AGI-3-Agents"
    if not work.exists():
        shutil.copytree(args.data / "ARC-AGI-3-Agents", work, ignore=shutil.ignore_patterns(".git"))
    (work / "agents" / "__init__.py").write_text(
        "from .agent import Agent, Playback\nfrom .swarm import Swarm\n", encoding="utf-8"
    )
    sys.path.insert(0, str(work))
    sys.path.insert(0, str(ROOT / "src"))
    logging.basicConfig(level=logging.WARNING)

    from arc_agi import Arcade, OperationMode
    from graph_explorer_agent import GraphExplorer

    if args.max_actions:
        GraphExplorer.MAX_ACTIONS = args.max_actions

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=os.environ["ENVIRONMENTS_DIR"])
    games = [e.game_id for e in arc.get_environments()]
    if args.games:
        prefixes = args.games.split(",")
        games = [g for g in games if any(g.startswith(p) for p in prefixes)]
    card_id = arc.open_scorecard(tags=["local_eval"])

    agents = [
        GraphExplorer(
            card_id=card_id,
            game_id=g,
            agent_name="graphexplorer",
            ROOT_URL="",
            record=False,
            arc_env=arc.make(g, scorecard_id=card_id),
        )
        for g in sorted(games)
    ]
    t0 = time.time()
    threads = [threading.Thread(target=a.main, daemon=True) for a in agents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    card = arc.close_scorecard(card_id)
    print(f"\n{'game':<16}{'score':>8}{'levels':>10}{'actions':>10}")
    for env in sorted(card.environments, key=lambda e: e.id or ""):
        run = env.runs[-1] if env.runs else None
        if run is None:
            continue
        print(f"{(env.id or '')[:15]:<16}{run.score:>8.2f}{f'{run.levels_completed}/{run.number_of_levels}':>10}{run.actions:>10}")
    print(f"\nTOTAL SCORE: {card.score:.2f}   ({len(games)} games, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
