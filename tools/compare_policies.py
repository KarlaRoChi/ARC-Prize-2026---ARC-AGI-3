"""compare_policies.py - Systematic controlled comparison of exploration policies for ARC-AGI-3.

Evaluates 4 policies across all 25 public ARC-AGI-3 games:
1. Policy A (v3 Baseline): Pure Graph Explorer (no directional momentum bias, MIN_CLICK_DISTANCE=0)
2. Policy B (v8 Momentum): Aggressive Directional Momentum (straight=0, lateral=1, opp=5, click=10)
3. Policy C (v8 Spatial NMS): Spatial Suppression (MIN_CLICK_DISTANCE=4)
4. Policy D (Balanced Momentum): Branch-Preserving Momentum (straight=0, lateral=0, opp=2, click=5)

Records per-game score, level completions, actions, runtime, and aggregate competition score.
Outputs results to JSON and CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Set environment
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
)


def custom_click_targets(grid: np.ndarray, min_click_dist: int = 0) -> List[Tuple[int, int]]:
    h, w = grid.shape
    y_min, y_max = (1, min(h, 63)) if h >= 64 else (0, h)
    x_min, x_max = (1, min(w, 63)) if w >= 64 else (0, w)
    sub = grid[y_min:y_max, x_min:x_max]
    vals, counts = np.unique(sub, return_counts=True)
    background = int(vals[np.argmax(counts)])
    seen = np.zeros_like(grid, dtype=bool)
    fg_objects: List[Tuple[int, int, int]] = []
    bg_objects: List[Tuple[int, int, int]] = []

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if seen[y, x]:
                continue
            color = grid[y, x]
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
            touches_border = False
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                if cy in (y_min, y_max - 1) or cx in (x_min, x_max - 1):
                    touches_border = True
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if y_min <= ny < y_max and x_min <= nx < x_max and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            my = sum(c[0] for c in cells) / len(cells)
            mx = sum(c[1] for c in cells) / len(cells)
            cy, cx = min(cells, key=lambda c: (c[0] - my) ** 2 + (c[1] - mx) ** 2)
            if color != background:
                fg_objects.append((len(cells), cx, cy))
            elif not touches_border and len(cells) <= 64:
                bg_objects.append((len(cells), cx, cy))

    fg_objects.sort()
    bg_objects.sort()
    combined = fg_objects if len(fg_objects) >= 8 else (fg_objects + bg_objects)

    kept: List[Tuple[int, int]] = []
    for _, cx, cy in combined:
        if min_click_dist <= 0 or all((cx - kx) ** 2 + (cy - ky) ** 2 >= min_click_dist ** 2 for kx, ky in kept):
            kept.append((cx, cy))
        if len(kept) >= MAX_CLICK_TARGETS:
            break
    return kept


class ParametricExplorer(Agent):
    """Configurable graph explorer for controlled ablation testing."""

    MAX_ACTIONS: int = 400

    def __init__(
        self,
        *args: Any,
        policy_mode: str = "baseline_v3",
        min_click_dist: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        seed = int.from_bytes(hashlib.blake2b(str(self.game_id).encode(), digest_size=4).digest(), "little")
        self.rng = random.Random(seed)
        self.policy_mode = policy_mode
        self.min_click_dist = min_click_dist

        self._reset_memory()
        self.prev_key: str | None = None
        self.prev_action: Tuple | None = None
        self.prev_levels = 0
        self.plan: List[Tuple] = []

    def _reset_memory(self) -> None:
        self.untried: Dict[str, List[Tuple]] = {}
        self.edges: Dict[str, Dict[Tuple, str]] = {}
        self.dead: set[Tuple[str, Tuple]] = set()
        self.start_key: str | None = None
        self.depth: Dict[str, int] = {}

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _actions_for(self, frame: FrameData, grid: np.ndarray | None) -> List[Tuple]:
        available = [int(a) for a in (frame.available_actions or [1, 2, 3, 4, 5, 6, 7]) if int(a) != 0]
        acts: List[Tuple] = []
        for a in available:
            if a == 6:
                if grid is not None:
                    acts.extend((6, x, y) for x, y in custom_click_targets(grid, self.min_click_dist))
            else:
                acts.append((a,))
        self.rng.shuffle(acts)

        if self.policy_mode == "baseline_v3":
            # Pure baseline: simple actions before clicks, uniform random navigation order
            acts.sort(key=lambda t: t[0] == 6)

        elif self.policy_mode == "momentum_v8":
            # Aggressive corridor momentum: straight ahead first, deprioritize lateral turns
            if self.prev_action and len(self.prev_action) == 1:
                prev_a = self.prev_action[0]
                opp = OPPOSITE.get(prev_a)
                def sort_key(t: Tuple) -> int:
                    if t[0] == 6:
                        return 10
                    if t[0] == prev_a:
                        return 0
                    if t[0] == opp:
                        return 5
                    return 1
                acts.sort(key=sort_key)
            else:
                acts.sort(key=lambda t: t[0] == 6)

        elif self.policy_mode == "balanced_branch":
            # Branch-preserving momentum: treats continuing straight and lateral branches with EQUAL priority (0)
            # Only deprioritizes direct 180-degree reversals (2) and puts clicks after navigation (5)
            if self.prev_action and len(self.prev_action) == 1:
                prev_a = self.prev_action[0]
                opp = OPPOSITE.get(prev_a)
                def sort_key_b(t: Tuple) -> int:
                    if t[0] == 6:
                        return 5
                    if t[0] == opp:
                        return 2
                    return 0  # straight and lateral turns get equal opportunity!
                acts.sort(key=sort_key_b)
            else:
                acts.sort(key=lambda t: t[0] == 6)

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

    def _path_to_frontier(self, start: str) -> List[Tuple]:
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node != start and self.untried.get(node):
                return path
            for action, nxt in self.edges.get(node, {}).items():
                if nxt not in visited and (node, action) not in self.dead:
                    visited.add(nxt)
                    queue.append((nxt, path + [action]))
        return []

    @staticmethod
    def _to_game_action(action: Tuple) -> GameAction:
        ga = GameAction.from_id(action[0])
        if action[0] == 6:
            ga.set_data({"x": int(action[1]), "y": int(action[2])})
        return ga

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
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

        key = state_key(grid)
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
                    action = self.rng.choice(known)
                else:
                    self.dead.clear()
                    self.untried[key] = self._actions_for(latest_frame, grid)
                    action = self.untried[key].pop(0) if self.untried[key] else (1,)

        self.prev_key, self.prev_action = key, action
        return self._to_game_action(action)


def evaluate_policy(
    arc: Arcade,
    games: List[str],
    policy_name: str,
    policy_mode: str,
    min_click_dist: int,
    max_actions: int = 400,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"EVALUATING POLICY: {policy_name}")
    print(f"Mode: {policy_mode}, min_click_dist: {min_click_dist}, max_actions: {max_actions}")
    print(f"=======================================================")

    card_id = arc.open_scorecard(tags=[policy_name])
    ParametricExplorer.MAX_ACTIONS = max_actions

    agents = [
        ParametricExplorer(
            card_id=card_id,
            game_id=g,
            agent_name=policy_name,
            ROOT_URL="",
            record=False,
            arc_env=arc.make(g, scorecard_id=card_id),
            policy_mode=policy_mode,
            min_click_dist=min_click_dist,
        )
        for g in sorted(games)
    ]

    t0 = time.time()
    threads = [threading.Thread(target=a.main, daemon=True) for a in agents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    card = arc.close_scorecard(card_id)

    per_game = {}
    solved_levels_total = 0
    total_actions = 0
    for env in sorted(card.environments, key=lambda e: e.id or ""):
        run = env.runs[-1] if env.runs else None
        if run is None:
            continue
        per_game[env.id] = {
            "score": round(float(run.score), 4),
            "levels_completed": run.levels_completed,
            "number_of_levels": run.number_of_levels,
            "actions": run.actions,
        }
        solved_levels_total += run.levels_completed
        total_actions += run.actions

    print(f"Aggregate Score: {card.score:.4f} | Solved Levels: {solved_levels_total} | Total Actions: {total_actions} | Time: {elapsed:.1f}s")
    for gid, res in per_game.items():
        if res["levels_completed"] > 0:
            print(f"  {gid[:15]:<16} score: {res['score']:>6.2f} levels: {res['levels_completed']}/{res['number_of_levels']} actions: {res['actions']}")

    return {
        "policy_name": policy_name,
        "policy_mode": policy_mode,
        "min_click_dist": min_click_dist,
        "max_actions": max_actions,
        "aggregate_score": round(float(card.score), 4),
        "solved_levels_total": solved_levels_total,
        "total_actions": total_actions,
        "runtime_s": round(elapsed, 2),
        "per_game": per_game,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--out", type=Path, default=ROOT / "policy_comparison_results.json")
    args = parser.parse_args()

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=os.environ["ENVIRONMENTS_DIR"])
    games = [e.game_id for e in arc.get_environments()]
    print(f"Discovered {len(games)} public ARC-AGI-3 games.")

    experiments = [
        ("Policy A (v3 Baseline)", "baseline_v3", 0),
        ("Policy B (v8 Momentum)", "momentum_v8", 0),
        ("Policy C (v8 Spatial NMS)", "baseline_v3", 4),
        ("Policy D (Balanced Branch)", "balanced_branch", 0),
    ]

    all_results = []
    for name, mode, nms in experiments:
        res = evaluate_policy(arc, games, name, mode, nms, max_actions=args.max_actions)
        all_results.append(res)

    args.out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nAll policy comparison results saved to {args.out}")


if __name__ == "__main__":
    main()
