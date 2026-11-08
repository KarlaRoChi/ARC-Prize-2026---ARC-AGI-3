"""Ablation study comparing v3, NMS-only, Momentum-only, and Combined (v4) on ARC-AGI-3 games.

Instruments:
- Score, levels completed, total actions, resets, and runtime
- HUD-invariant hashing state collision detection (hash collisions across distinct playfields)
- Target suppression count by NMS
- Backtracking frequency and cycle detection under directional momentum
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

# Setup environment
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
    MAX_CLICK_TARGETS,
    MIN_CLICK_DISTANCE,
)


def get_click_targets(grid: np.ndarray, enable_nms: bool) -> tuple[list[tuple[int, int]], int]:
    """Returns (targets, num_suppressed)."""
    h, w = grid.shape
    y_min, y_max = (1, min(h, 63)) if h >= 64 else (0, h)
    x_min, x_max = (1, min(w, 63)) if w >= 64 else (0, w)
    sub = grid[y_min:y_max, x_min:x_max]
    vals, counts = np.unique(sub, return_counts=True)
    background = int(vals[np.argmax(counts)])
    seen = np.zeros_like(grid, dtype=bool)
    fg_objects: list[tuple[int, int, int]] = []
    bg_objects: list[tuple[int, int, int]] = []

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

    if not enable_nms:
        return [(cx, cy) for _, cx, cy in combined[:MAX_CLICK_TARGETS]], 0

    kept: list[tuple[int, int]] = []
    suppressed = 0
    for _, cx, cy in combined:
        if all((cx - kx) ** 2 + (cy - ky) ** 2 >= MIN_CLICK_DISTANCE ** 2 for kx, ky in kept):
            kept.append((cx, cy))
        else:
            suppressed += 1
        if len(kept) >= MAX_CLICK_TARGETS:
            break
    return kept, suppressed


class ConfigurableExplorer(Agent):
    def __init__(
        self,
        *args: Any,
        enable_nms: bool = True,
        enable_momentum: bool = True,
        max_actions: int = 2000,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        seed = int.from_bytes(hashlib.blake2b(str(self.game_id).encode(), digest_size=4).digest(), "little")
        self.rng = random = np.random.default_rng(seed)
        self.enable_nms = enable_nms
        self.enable_momentum = enable_momentum
        self.max_actions = max_actions

        self._reset_memory()
        self.prev_key: str | None = None
        self.prev_action: tuple | None = None
        self.prev_levels = 0
        self.plan: list[tuple] = []

        # Diagnostics instrumentation
        self.action_count = 0
        self.reset_count = 0
        self.suppressed_targets_total = 0
        self.backtrack_attempts = 0
        self.backtrack_blocked = 0
        self.hash_to_playfield: dict[str, bytes] = {}
        self.detected_hash_collisions = 0
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
                    targets, supp = get_click_targets(grid, self.enable_nms)
                    self.suppressed_targets_total += supp
                    acts.extend((6, x, y) for x, y in targets)
            else:
                acts.append((a,))
        
        # Shuffle deterministically
        perm = self.rng.permutation(len(acts))
        acts = [acts[i] for i in perm]

        if self.enable_momentum and self.prev_action and len(self.prev_action) == 1:
            prev_a = self.prev_action[0]
            opp = OPPOSITE.get(prev_a)
            def sort_key(t: tuple) -> int:
                if t[0] == 6:
                    return 10
                if t[0] == prev_a:
                    return 0
                if t[0] == opp:
                    self.backtrack_attempts += 1
                    return 5
                return 1
            acts.sort(key=sort_key)
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

    def _path_to_frontier(self, start: str) -> list[tuple]:
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
    def _to_game_action(action: tuple) -> GameAction:
        ga = GameAction.from_id(action[0])
        if action[0] == 6:
            ga.set_data({"x": int(action[1]), "y": int(action[2])})
        return ga

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        self.action_count += 1
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.reset_count += 1
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

        # Audit HUD-invariant state hashing collisions
        pf = playfield_of(grid)
        pf_bytes = pf.tobytes() if pf is not None else b""
        key = state_key(grid)
        if key in self.hash_to_playfield:
            if self.hash_to_playfield[key] != pf_bytes:
                self.detected_hash_collisions += 1
        else:
            self.hash_to_playfield[key] = pf_bytes

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


def run_variant(
    variant_name: str,
    enable_nms: bool,
    enable_momentum: bool,
    game_list: list[str],
    max_actions: int,
) -> dict[str, Any]:
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=os.environ["ENVIRONMENTS_DIR"])
    card_id = arc.open_scorecard(tags=[f"ablation_{variant_name}"])

    agents: dict[str, ConfigurableExplorer] = {}
    for g in sorted(game_list):
        agent = ConfigurableExplorer(
            card_id=card_id,
            game_id=g,
            agent_name=f"agent_{variant_name}",
            ROOT_URL="",
            record=False,
            arc_env=arc.make(g, scorecard_id=card_id),
            enable_nms=enable_nms,
            enable_momentum=enable_momentum,
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
            "number_of_levels": run.number_of_levels,
            "actions": run.actions,
            "resets": ag.reset_count if ag else 0,
            "suppressed_targets": ag.suppressed_targets_total if ag else 0,
            "backtrack_attempts": ag.backtrack_attempts if ag else 0,
            "hash_collisions": ag.detected_hash_collisions if ag else 0,
            "failure_reason": ag.failure_reason if ag else "unknown",
        }

    return {
        "variant": variant_name,
        "enable_nms": enable_nms,
        "enable_momentum": enable_momentum,
        "total_score": card.score,
        "runtime_s": elapsed,
        "games_count": len(game_list),
        "per_game": per_game,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="", help="Comma separated games, or empty for all 25")
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=ROOT / "ablation_results.json")
    args = parser.parse_args()

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=os.environ["ENVIRONMENTS_DIR"])
    all_games = [e.game_id for e in arc.get_environments()]
    if args.games:
        prefixes = args.games.split(",")
        game_list = [g for g in all_games if any(g.startswith(p) for p in prefixes)]
    else:
        game_list = all_games

    print(f"=== ARC-AGI-3 ABLATION STUDY: 4 Variants across {len(game_list)} games ===")
    print(f"Max actions per game: {args.max_actions}")
    
    variants = [
        ("v3_baseline", False, False),
        ("nms_only", True, False),
        ("momentum_only", False, True),
        ("v4_combined", True, True),
    ]

    results = {}
    for name, nms, mom in variants:
        print(f"\n--- Running Variant: {name} (NMS={nms}, Momentum={mom}) ---")
        res = run_variant(name, nms, mom, game_list, args.max_actions)
        results[name] = res
        print(f"Result {name}: Total Score = {res['total_score']:.4f}, Runtime = {res['runtime_s']:.1f}s")
        for gid, stats in sorted(res["per_game"].items()):
            if stats["score"] > 0:
                print(f"  {gid:<16}: score={stats['score']:.2f}, levels={stats['levels_completed']}/{stats['number_of_levels']}, actions={stats['actions']}, resets={stats['resets']}")

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAll results saved to {args.out}")


if __name__ == "__main__":
    main()
