# 🕹️ Experiment Log: ARC Prize 2026 — ARC-AGI-3

**Metric:** per-game mean of `min(human_actions / agent_actions, 1)²` weighted by level index (0–100).  
**Resource:** CPU (0 GPU slots).  
**Local CV:** `python tools/local_eval.py --data data` over the 25 public games in `environment_files/`.

| Version | Date | Hypothesis / Description | Technical Changes | Local CV (25 public) | LB Score | Status / Conclusions |
|---|---|---|---|---|---|---|
| `v1_graph_explorer` | 2026-10-17 | Systematic exploration of the state graph: no LLM, treating each level as a deterministic state machine | State = grid hash; untried actions prioritized first; BFS to nearest known state with remaining untried actions; `ACTION6` targeted at centroids of connected objects; transitions to `GAME_OVER` blacklisted; graph reset on level completion; 4000 action cap per game | `0.28` (10 levels across 8 of 25 games) | **`0.19`** | ✅ **100% PRIZE-ELIGIBLE.** Functional baseline (Ref: 57316253). Solves levels but uses many actions, and squared efficiency penalizes excess exploration heavily. Local smoke (`ls20`, `vc33`, `r11l`): 2.27; Kaggle commit run v1: 0.61. |
| `v2_self_loop_pruning` | 2026-10-18 | **Self-loop dead-end transition pruning** to eliminate no-op wall collisions and ineffective clicks + deterministic seeding | Prunes `(state, action)` transitions where `next_state == state` from the BFS frontier; translates documentation and agent comments to English | `0.28` | `0.03` | ✅ **100% PRIZE-ELIGIBLE.** Evaluated on Kaggle (Ref: 57374895, LB `0.03`). Strict self-loop pruning reduced action counts on deterministic mazes but prematurely pruned stateful no-op actions on multi-step trigger games. |
