"""Graph-explorer agent for ARC-AGI-3 (v1 baseline, CPU only).

Treats each game level as an unknown deterministic state machine:
- A state is the hash of the latest rendered grid.
- For every state it remembers which actions were already tried and where they led.
- It always prefers an untried action in the current state; when the current state is
  exhausted it walks (BFS over the known transition graph) to the nearest state that
  still has untried actions.
- ACTION6 (click) candidates are the centres of the connected colour objects in the grid,
  so clicks land on things instead of random pixels.
- Transitions that end in GAME_OVER are marked as dead ends and never retried.
- When a level is completed the graph is cleared, because the next level is a new map.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

MAX_CLICK_TARGETS = 48  # ACTION6 candidates kept per state


def grid_of(frame: FrameData) -> np.ndarray | None:
    """Last animation frame as a 2D array (frames can contain several sub-frames)."""
    if not frame.frame:
        return None
    return np.asarray(frame.frame[-1], dtype=np.int16)


def state_key(grid: np.ndarray | None) -> str:
    if grid is None:
        return "empty"
    return hashlib.blake2b(grid.tobytes() + bytes(grid.shape), digest_size=12).hexdigest()


def click_targets(grid: np.ndarray) -> list[tuple[int, int]]:
    """Centres of connected same-colour regions, smallest (most salient) first."""
    h, w = grid.shape
    vals, counts = np.unique(grid, return_counts=True)
    background = int(vals[np.argmax(counts)])
    seen = np.zeros_like(grid, dtype=bool)
    objects: list[tuple[int, int, int]] = []  # (size, x, y)
    for y in range(h):
        for x in range(w):
            if seen[y, x] or grid[y, x] == background:
                continue
            color = grid[y, x]
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            # click the cell closest to the centroid so it is inside the object
            my = sum(c[0] for c in cells) / len(cells)
            mx = sum(c[1] for c in cells) / len(cells)
            cy, cx = min(cells, key=lambda c: (c[0] - my) ** 2 + (c[1] - mx) ** 2)
            objects.append((len(cells), cx, cy))
    objects.sort()
    return [(x, y) for _, x, y in objects[:MAX_CLICK_TARGETS]]


class GraphExplorer(Agent):
    """Systematic explorer: untried actions first, BFS to the nearest unexplored state."""

    MAX_ACTIONS = 4000  # per game; efficiency is scored per level, so cap the flailing

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(hash(self.game_id) & 0xFFFFFFFF)
        self._reset_memory()
        self.prev_key: str | None = None
        self.prev_action: tuple | None = None
        self.prev_levels = 0
        self.plan: list[tuple] = []

    def _reset_memory(self) -> None:
        self.untried: dict[str, list[tuple]] = {}
        self.edges: dict[str, dict[tuple, str]] = {}
        self.dead: set[tuple[str, tuple]] = set()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # -- bookkeeping -------------------------------------------------------
    def _actions_for(self, frame: FrameData, grid: np.ndarray | None) -> list[tuple]:
        available = frame.available_actions or [1, 2, 3, 4, 5, 6, 7]
        acts: list[tuple] = []
        for a in available:
            a = int(a)
            if a == 0:
                continue
            if a == 6:
                if grid is not None:
                    acts.extend((6, x, y) for x, y in click_targets(grid))
            else:
                acts.append((a,))
        random.shuffle(acts)
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
            # Action resulted in no change (wall collision or ineffective click)
            self.dead.add((self.prev_key, self.prev_action))
            return
        self.edges.setdefault(self.prev_key, {})[self.prev_action] = key

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
        key = state_key(grid)
        expected = self.edges.get(self.prev_key or "", {}).get(self.prev_action)
        self._record_transition(key, latest_frame)
        if key not in self.untried:
            self.untried[key] = self._actions_for(latest_frame, grid)

        # follow a planned path only while the world behaves as remembered
        if self.plan and expected is not None and expected != key:
            self.plan = []

        options = [a for a in self.untried[key] if (key, a) not in self.dead]
        if not self.plan and options:
            action = options[0]
            self.untried[key].remove(action)
        else:
            if not self.plan:
                self.plan = self._path_to_frontier(key)
            if self.plan:
                action = self.plan.pop(0)
            else:
                known = [a for a in self.edges.get(key, {}) if (key, a) not in self.dead]
                action = random.choice(known) if known else (1,)

        self.prev_key, self.prev_action = key, action
        return self._to_game_action(action)
