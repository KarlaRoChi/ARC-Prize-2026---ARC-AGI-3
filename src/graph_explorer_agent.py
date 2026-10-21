"""Graph-explorer agent for ARC-AGI-3 (v3, CPU only).

Treats each game level as an unknown deterministic state machine with HUD-invariant state hashing:
- Strips the 1-pixel outer HUD step-counter border (and 2px bottom HUD bar in games like ls20)
  before computing the Blake2b state hash so step-counter ticks do not poison wall detection
  or invalidate BFS plans (`self.plan`).
- Detects full-screen flash penalty frames and soft-resets back to `start_key` from deep states,
  preserving interrupted frontier actions instead of recording fake transitions to `start_key`.
- Uses per-instance deterministic `random.Random(seed)` for thread-safe multi-game execution.
- Extracts both foreground connected components and enclosed background cavities (<=64 px)
  inside the playfield as `ACTION6` click targets.
- Escapes exhausted leaf traps by clearing `self.dead` and refilling `self.untried[key]`
  instead of repeating `(1,)` indefinitely.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

MAX_CLICK_TARGETS = 16  # ACTION6 candidates kept per state
MIN_CLICK_DISTANCE = 4  # Spatial NMS radius in pixels
OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}


def grid_of(frame: FrameData) -> np.ndarray | None:
    """Last animation frame as a 2D array (frames can contain several sub-frames)."""
    if not frame.frame:
        return None
    return np.asarray(frame.frame[-1], dtype=np.int16)


def playfield_of(grid: np.ndarray | None) -> np.ndarray | None:
    """Strip 1px outer HUD border, plus 2px bottom HUD bar if present (e.g. ls20)."""
    if grid is None:
        return None
    h, w = grid.shape
    if h >= 64 and w >= 64:
        pf = grid[1:63, 1:63].copy()
        if np.array_equal(grid[61, 13:63], grid[62, 13:63]):
            pf[60:62, 12:] = 0
        return pf
    return grid


def state_key(grid: np.ndarray | None) -> str:
    pf = playfield_of(grid)
    if pf is None:
        return "empty"
    return hashlib.blake2b(pf.tobytes() + bytes(pf.shape), digest_size=12).hexdigest()


def is_flash_frame(grid: np.ndarray | None) -> bool:
    pf = playfield_of(grid)
    if pf is None:
        return True
    return bool(np.all(pf == pf[0, 0]))


def click_targets(grid: np.ndarray) -> list[tuple[int, int]]:
    """Centres of connected same-colour regions inside the playfield, smallest first."""
    h, w = grid.shape
    y_min, y_max = (1, min(h, 63)) if h >= 64 else (0, h)
    x_min, x_max = (1, min(w, 63)) if w >= 64 else (0, w)
    sub = grid[y_min:y_max, x_min:x_max]
    vals, counts = np.unique(sub, return_counts=True)
    background = int(vals[np.argmax(counts)])
    seen = np.zeros_like(grid, dtype=bool)
    fg_objects: list[tuple[int, int, int]] = []  # (size, x, y)
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
            # click the cell closest to the centroid so it is inside the object
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
    
    kept: list[tuple[int, int]] = []
    for _, cx, cy in combined:
        if all((cx - kx) ** 2 + (cy - ky) ** 2 >= MIN_CLICK_DISTANCE ** 2 for kx, ky in kept):
            kept.append((cx, cy))
        if len(kept) >= MAX_CLICK_TARGETS:
            break
    return kept


class GraphExplorer(Agent):
    """Systematic HUD-invariant explorer: untried actions first, BFS to nearest frontier state."""

    MAX_ACTIONS = 4000  # per game; efficiency is scored per level, so cap the flailing

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int.from_bytes(hashlib.blake2b(str(self.game_id).encode(), digest_size=4).digest(), "little")
        self.rng = random.Random(seed)
        self._reset_memory()
        self.prev_key: str | None = None
        self.prev_action: tuple | None = None
        self.prev_levels = 0
        self.plan: list[tuple] = []

    def _reset_memory(self) -> None:
        self.untried: dict[str, list[tuple]] = {}
        self.edges: dict[str, dict[tuple, str]] = {}
        self.dead: set[tuple[str, tuple]] = set()
        self.start_key: str | None = None
        self.depth: dict[str, int] = {}

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # -- bookkeeping -------------------------------------------------------
    def _actions_for(self, frame: FrameData, grid: np.ndarray | None) -> list[tuple]:
        available = [int(a) for a in (frame.available_actions or [1, 2, 3, 4, 5, 6, 7]) if int(a) != 0]
        acts: list[tuple] = []
        for a in available:
            if a == 6:
                if grid is not None:
                    acts.extend((6, x, y) for x, y in click_targets(grid))
            else:
                acts.append((a,))
        self.rng.shuffle(acts)
        # Apply directional momentum: maintain current heading down corridors and deprioritize immediate reversals
        if self.prev_action and len(self.prev_action) == 1:
            prev_a = self.prev_action[0]
            opp = OPPOSITE.get(prev_a)
            def sort_key(t: tuple) -> int:
                if t[0] == 6:
                    return 10  # clicks after navigation
                if t[0] == prev_a:
                    return 0   # continue straight along corridor
                if t[0] == opp:
                    return 5   # backtracking is lowest directional priority
                return 1       # lateral turns
            acts.sort(key=sort_key)
        else:
            # simple actions before clicks: they are few and usually move the player
            acts.sort(key=lambda t: t[0] == 6)
        return acts

    def _record_transition(self, key: str, frame: FrameData) -> None:
        if self.prev_key is None or self.prev_action is None:
            return
        if frame.state is GameState.GAME_OVER:
            self.dead.add((self.prev_key, self.prev_action))
            return
        if key == self.prev_key:
            # Record self-loop edge for BFS so it knows this action does not change the state,
            # but do NOT add to self.dead so we don't break multi-step triggers or stateful buttons.
            self.edges.setdefault(self.prev_key, {})[self.prev_action] = key
            return
        # Detect soft-reset to start_key from deep state (step timeout / life lost)
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
        """Shortest known action sequence from start to a state with untried actions."""
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

    # -- policy ------------------------------------------------------------
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
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
        # Skip uniform flash/penalty frames so they do not pollute the transition graph
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

        # follow a planned path only while the world behaves as remembered
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
                    # Leaf-trap recovery: clear dead set and refill current state's untried queue
                    self.dead.clear()
                    self.untried[key] = self._actions_for(latest_frame, grid)
                    action = self.untried[key].pop(0) if self.untried[key] else (1,)

        self.prev_key, self.prev_action = key, action
        return self._to_game_action(action)
