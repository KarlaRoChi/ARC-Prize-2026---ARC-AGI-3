"""Parametric Sensitivity Analysis and Grid Study of Corridor Momentum for ARC-AGI-3.

Evaluates momentum decay coefficients (gamma) and reversal penalties (rho):
- gamma in [0.5, 0.7, 0.85, 0.95]
- rho in [0.0, 0.1, 0.25, 0.5]
Across sensitive diagnostic games: ar25, g50t, r11l, vc33.
MIN_CLICK_DISTANCE strictly locked at 0 (Zero Spatial NMS).
Also includes Step 3.2 extended 2500-action scalability and Blake2b hash collision verification on r11l and ls20.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Environment configuration
os.environ["OPERATION_MODE"] = "offline"
data_dir = ROOT / "data"
os.environ["ENVIRONMENTS_DIR"] = str(data_dir / "environment_files")
os.environ["RECORDINGS_DIR"] = str(ROOT / "submissions" / "recordings")

work = ROOT / "submissions" / "_work" / "ARC-AGI-3-Agents"
if not work.exists():
    shutil.copytree(data_dir / "ARC-AGI-3-Agents", work, ignore=shutil.ignore_patterns(".git"))
(work / "agents" / "__init__.py").write_text(
    "from .agent import Agent, Playback\nfrom .swarm import Swarm\n", encoding="utf-8"
)
sys.path.insert(0, str(work))
sys.path.insert(0, str(ROOT / "src"))
logging.basicConfig(level=logging.WARNING)

from arc_agi import Arcade, OperationMode
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent
from graph_explorer_agent import (
    OPPOSITE,
    grid_of,
    playfield_of,
    state_key,
    is_flash_frame,
    click_targets,
    MAX_CLICK_TARGETS,
    MIN_CLICK_DISTANCE,
)


class ParametricMomentumExplorer(Agent):
    """Explorer agent with parameterized corridor momentum persistence (gamma) and reversal penalty (rho)."""

    def __init__(
        self,
        *args: Any,
        gamma: float = 0.85,
        rho: float = 0.25,
        max_actions: int = 2000,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        seed = int.from_bytes(hashlib.blake2b(str(self.game_id).encode(), digest_size=4).digest(), "little")
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.rho = rho
        self.max_actions = max_actions

        self._reset_memory()
        self.prev_key: str | None = None
        self.prev_action: tuple | None = None
        self.prev_levels = 0
        self.plan: list[tuple] = []

        # Diagnostics & hash collision tracking
        self.action_count = 0
        self.reset_count = 0
        self.backtrack_attempts = 0
        self.hash_to_playfield: dict[str, bytes] = {}
        self.detected_hash_collisions = 0
        self.total_unique_states = 0
        self.failure_reason = "in_progress"

    def _reset_memory(self) -> None:
        self.untried: dict[str, list[tuple]] = {}
        self.edges: dict[str, dict[tuple, str]] = {}
        self.dead: set[tuple[str, tuple]] = set()
        self.start_key: str | None = None
        self.depth: dict[str, int] = {}

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            self.failure_reason = "WIN"
            return True
        if self.action_count >= self.max_actions:
            self.failure_reason = "MAX_ACTIONS_REACHED"
            return True
        return False

    def _actions_for(self, frame: FrameData, grid: np.ndarray | None) -> list[tuple]:
        available = [int(a) for a in (frame.available_actions or [1, 2, 3, 4, 5, 6, 7]) if int(a) != 0]
        acts: list[tuple] = []
        for a in available:
            if a == 6:
                if grid is not None:
                    # Strict MIN_CLICK_DISTANCE = 0 (Zero Spatial NMS)
                    targets = click_targets(grid)
                    acts.extend((6, x, y) for x, y in targets)
            else:
                acts.append((a,))

        # Deterministic shuffle
        perm = self.rng.permutation(len(acts))
        acts = [acts[i] for i in perm]

        if self.prev_action and len(self.prev_action) == 1:
            prev_a = self.prev_action[0]
            opp = OPPOSITE.get(prev_a)

            def sort_key(t: tuple) -> float:
                if t[0] == 6:
                    return 10.0
                if t[0] == prev_a:
                    return 1.0 - self.gamma  # Higher gamma -> lower sort key -> higher priority
                if t[0] == opp:
                    self.backtrack_attempts += 1
                    return 1.0 + self.rho    # Higher rho -> higher sort key -> lower priority
                return 1.0                  # Lateral turns: baseline 1.0

            acts.sort(key=sort_key)
        else:
            acts.sort(key=lambda t: 10.0 if t[0] == 6 else 1.0)
        return acts

    def _record_transition(self, key: str, frame: FrameData) -> None:
        if self.prev_key is None or self.prev_action is None:
            return
        if frame.state is GameState.GAME_OVER:
            self.dead.add((self.prev_key, self.prev_action))
            return
        if key == self.prev_key:
            self.edges.setdefault(self.prev_key, {})[self.prev_action] = key
            return
        if key == self.start_key and self.depth.get(self.prev_key, 0) > 2:
            if self.prev_key in self.untried and self.prev_action not in self.untried[self.prev_key]:
                self.untried[self.prev_key].insert(0, self.prev_action)
            return
        self.edges.setdefault(self.prev_key, {})[self.prev_action] = key
        if key not in self.depth:
            self.depth[key] = self.depth.get(self.prev_key, 0) + 1
        else:
            self.depth[key] = min(self.depth[key], self.depth.get(self.prev_key, 0) + 1)

    def _path_to_frontier(self, start: str) -> list[tuple]:
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if self.untried.get(node):
                return path
            for action, nxt in self.edges.get(node, {}).items():
                if (node, action) in self.dead or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [action]))
        return []

    @staticmethod
    def _to_game_action(action: tuple) -> GameAction:
        ga = GameAction.from_id(action[0])
        if action[0] == 6:
            ga.set_data({"x": int(action[1]), "y": int(action[2])})
        return ga

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        self.action_count += 1
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.GAME_OVER:
                self._record_transition("", latest_frame)
            self.prev_key, self.prev_action, self.plan = None, None, []
            return GameAction.RESET

        if latest_frame.levels_completed > self.prev_levels:
            self.prev_levels = latest_frame.levels_completed
            self._reset_memory()
            self.prev_key, self.prev_action, self.plan = None, None, []

        grid = grid_of(latest_frame)
        if is_flash_frame(grid) and self.start_key is not None:
            if self.prev_key in self.untried and self.prev_action is not None:
                if self.prev_action not in self.untried[self.prev_key]:
                    self.untried[self.prev_key].insert(0, self.prev_action)
            self.prev_key, self.prev_action, self.plan = None, None, []
            avail = [int(a) for a in (latest_frame.available_actions or [1]) if int(a) != 0]
            act = (avail[0],) if avail and avail[0] != 6 else (6, 32, 32)
            return self._to_game_action(act)

        # Hash collision tracking
        pf = playfield_of(grid)
        pf_bytes = pf.tobytes() if pf is not None else b""
        key = state_key(grid)
        if key in self.hash_to_playfield:
            if self.hash_to_playfield[key] != pf_bytes:
                self.detected_hash_collisions += 1
        else:
            self.hash_to_playfield[key] = pf_bytes
            self.total_unique_states += 1

        if self.start_key is None:
            self.start_key = key
            self.depth[key] = 0

        expected = self.edges.get(self.prev_key or "", {}).get(self.prev_action)
        self._record_transition(key, latest_frame)
        if key not in self.untried:
            self.untried[key] = self._actions_for(latest_frame, grid)

        if self.plan and expected is not None and expected != key:
            self.plan = []

        if not self.plan and self.untried[key]:
            action = self.untried[key].pop(0)
        else:
            if not self.plan:
                self.plan = self._path_to_frontier(key)
            if self.plan:
                action = self.plan.pop(0)
            else:
                known = [a for a in self.edges.get(key, {}) if (key, a) not in self.dead]
                if known:
                    idx = int(self.rng.integers(0, len(known)))
                    action = known[idx]
                else:
                    self.dead.clear()
                    self.untried[key] = self._actions_for(latest_frame, grid)
                    action = self.untried[key].pop(0) if self.untried[key] else (1,)

        self.prev_key, self.prev_action = key, action
        return self._to_game_action(action)


def evaluate_grid_point(
    gamma: float,
    rho: float,
    games: list[str],
    max_actions: int = 2000,
) -> dict[str, Any]:
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=os.environ["ENVIRONMENTS_DIR"])
    card_id = arc.open_scorecard(tags=[f"tune_g{gamma}_r{rho}"])

    agents: dict[str, ParametricMomentumExplorer] = {}
    for g in sorted(games):
        agent = ParametricMomentumExplorer(
            card_id=card_id,
            game_id=g,
            agent_name=f"tune_g{gamma}_r{rho}",
            ROOT_URL="",
            record=False,
            arc_env=arc.make(g, scorecard_id=card_id),
            gamma=gamma,
            rho=rho,
            max_actions=max_actions,
        )
        agents[g] = agent

    t0 = time.time()
    threads = [threading.Thread(target=a.main, daemon=True) for a in agents.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    card = arc.close_scorecard(card_id)

    per_game = {}
    for env in card.environments:
        gid = env.id or ""
        run = env.runs[-1] if env.runs else None
        if run is None:
            continue
        ag = agents.get(gid)
        per_game[gid] = {
            "score": run.score,
            "levels_completed": run.levels_completed,
            "actions": run.actions,
            "backtracks": ag.backtrack_attempts if ag else 0,
            "collisions": ag.detected_hash_collisions if ag else 0,
            "unique_states": ag.total_unique_states if ag else 0,
        }

    return {
        "gamma": gamma,
        "rho": rho,
        "total_score": card.score,
        "runtime_s": elapsed,
        "per_game": per_game,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-scalability", action="store_true", help="Run extended 2500-step test on r11l and ls20")
    parser.add_argument("--out", type=Path, default=ROOT / "momentum_tuning_results.json")
    args = parser.parse_args()

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=os.environ["ENVIRONMENTS_DIR"])
    all_envs = [e.game_id for e in arc.get_environments()]

    # Diagnostic games: ar25, g50t, r11l, vc33
    diag_prefixes = ["ar25", "g50t", "r11l", "vc33"]
    diag_games = [g for g in all_envs if any(g.startswith(p) for p in diag_prefixes)]
    print(f"Discovered {len(diag_games)} diagnostic games: {diag_games}")

    if args.run_scalability:
        print("\n=== STEP 3.2: HUD-INVARIANT STATE HASHING EXTENDED SCALABILITY (2,500 ACTIONS) ===")
        scalability_prefixes = ["r11l", "ls20"]
        scal_games = [g for g in all_envs if any(g.startswith(p) for p in scalability_prefixes)]
        print(f"Target scalability games: {scal_games}")
        
        res = evaluate_grid_point(gamma=0.85, rho=0.25, games=scal_games, max_actions=2500)
        print(f"Scalability Run Complete in {res['runtime_s']:.2f}s:")
        total_collisions = 0
        total_unique = 0
        for gid, stats in res["per_game"].items():
            print(f"  Game {gid}: actions={stats['actions']}, unique_states={stats['unique_states']}, collisions={stats['collisions']}, score={stats['score']:.4f}")
            total_collisions += stats['collisions']
            total_unique += stats['unique_states']
        
        collision_rate = (total_collisions / max(total_unique, 1)) * 100
        print(f"Total Unique States Sampled: {total_unique}")
        print(f"Total Blake2b Hash Collisions: {total_collisions} ({collision_rate:.4f}%)")
        assert total_collisions == 0, f"Expected 0 collisions, found {total_collisions}"
        print("CONFIRMED: Mathematical hash stability verified (0.00% collision rate).")
        return

    # Step 3.1: Grid search
    gammas = [0.5, 0.7, 0.85, 0.95]
    rhos = [0.0, 0.1, 0.25, 0.5]
    
    print("\n=== STEP 3.1: PARAMETRIC GRID STUDY (gamma x rho) ===")
    print(f"Grid: gamma in {gammas}, rho in {rhos} (16 combinations)")
    
    grid_results = []
    t_start = time.time()
    
    for gamma in gammas:
        for rho in rhos:
            res = evaluate_grid_point(gamma=gamma, rho=rho, games=diag_games, max_actions=2000)
            scores = {gid.split("-")[0]: round(s["score"], 4) for gid, s in res["per_game"].items()}
            print(f"gamma={gamma:<4} rho={rho:<4} | Total: {res['total_score']:.4f} | Per-game: {scores} ({res['runtime_s']:.1f}s)")
            grid_results.append(res)

    total_time = time.time() - t_start
    print(f"\nGrid Search completed in {total_time:.1f}s.")
    
    best_point = max(grid_results, key=lambda x: x["total_score"])
    print(f"\nTOP CONFIGURATION: gamma={best_point['gamma']}, rho={best_point['rho']} with Total Score = {best_point['total_score']:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "grid_evaluated": len(grid_results),
            "best_config": {
                "gamma": best_point["gamma"],
                "rho": best_point["rho"],
                "total_score": best_point["total_score"],
            },
            "results": grid_results,
        }, f, indent=2)
    print(f"Results persisted to {args.out}")


if __name__ == "__main__":
    main()
