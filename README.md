# 🕹️ ARC Prize 2026 — ARC-AGI-3 (Interactive Reasoning)

> **Kaggle Competition:** [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)  
> **Compute Resource:** 💻 CPU (0 GPU slots) in v1–v2. The competition pool also offers RTX 6000 (`g4-standard-48`) for LLM-based agents.  
> **Internet:** ❌ Disabled  
> **Metric:** Completion × efficiency relative to human baselines, 0–100 (see [OVERVIEW.md](OVERVIEW.md))  
> **Best LB Score:** **`0.23`** (`v3_hud_invariant_soft_self_loop`, own agent, 100% prize-eligible)  
> **Final Leaderboard Rank:** 🏅 **2214 / 4923** teams (top 45%)  
> **Deliverable:** Generated automatically by the Kaggle evaluation gateway during the rerun.  
> **Key Dates:** 2026-10-26 (entry / team merger deadline) · **2026-11-02 (final submission deadline)**

---

## 🎯 Competition Objective

Unlike ARC-AGI-2 (predicting a static grid from demonstration pairs), here the agent **plays unknown interactive video games**:
it receives frames (a grid up to 64×64 with colors 0–15), chooses actions (`RESET`, `ACTION1`–`ACTION7`, where `ACTION6` is a coordinate click `x, y`), and must discover through exploration what each action does, what the goal is, and how to beat each level.

Evaluation runs on **110 private games** unseen by the agent (55 for the public LB, 55 for the private LB). Twenty-five public games are provided in the dataset for local development.

---

## 🏗️ Technical Approach: Graph Explorer Agent (written from scratch)

[`src/graph_explorer_agent.py`](src/graph_explorer_agent.py) models each level as an unknown deterministic finite-state machine:

1. **State Representation:** Hash of the visible 2D grid.
2. **Untried-Action Priority:** Untried actions at the current state are executed first.
3. **HUD-Invariant State Hashing (`v3`):** The outer HUD border and bottom bar are stripped before hashing, so step counters do not create false new states.
4. **Soft Self-Loop Recording (`v3`):** Transitions where `next_state == state` are recorded instead of hard-pruned (hard pruning in `v2` dropped the LB to 0.03 because it blacklisted stateful buttons).
5. **Breadth-First Search (BFS) Frontier Navigation:** Navigates over known transitions to the nearest visited state that still has untried actions.
6. **Object-Centric Clicks (`ACTION6`):** Targets centroids of connected color components and enclosed background cavities.
7. **Failure Memory:** Transitions leading to `GAME_OVER` are blacklisted and avoided after reset.
8. **Per-Level Graph Reset:** Clears the state transition graph whenever a level is completed.

> [!NOTE]
> **100% Prize-Eligible (CC BY 4.0):** This agent is written entirely from scratch with zero external weights or unverified dependencies.

### 📈 Results and ablation

| Version | Change | LB |
|---|---|:---:|
| `v1` | Graph explorer baseline | 0.19 |
| `v2` | Hard self-loop pruning | 0.03 |
| `v3` | HUD-invariant hashing + soft self-loops | **0.23** |

A later 25-game offline ablation (`tools/ablation_study.py`) showed that **directional momentum** nearly doubles the local score (0.19 → 0.37) while spatial click NMS hurts it, and a γ × ρ sweep (`tools/tune_momentum.py`) showed that a reversal penalty ρ ≥ 0.1 is needed to stop corridor ping-ponging. See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for details.

### LLM agent baseline (not my work)

[`kernel_duck_m2/`](kernel_duck_m2/) packages the open-sourced Milestone 2 winning agent by Daniel Franzen ([da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution), Apache 2.0, built on [Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness)) to compare against my heuristic agent. My contribution there is only the build script (`tools/build_duck_m2.py`) and the reduced commit-run settings.

---

## 📁 Directory Structure

```text
./
├── kaggle_arc3_agent.ipynb      # Kaggle notebook (GENERATED from src/, do not edit by hand)
├── kernel-metadata.json         # Remote kernel metadata (CPU, offline)
├── src/
│   └── graph_explorer_agent.py  # Single source of truth for the agent
├── tools/
│   ├── build_notebook.py        # Compiles src/ → kaggle_arc3_agent.ipynb
│   ├── local_eval.py            # Offline evaluation over the 25 public games
│   ├── ablation_study.py        # 25-game ablation (momentum vs NMS)
│   └── tune_momentum.py         # γ × ρ parameter sweep
├── kernel_duck_m2/              # Packaged third-party LLM agent (see above)
├── EXPERIMENT_LOG.md            # Version history and CV/LB results
├── OVERVIEW.md                  # Official competition overview summary
├── DATASET_DESCRIPTION.md       # Game API, frame format, and scoring specification
├── RULES.md                     # Prize eligibility & CC BY 4.0 rules
├── data/                        # (git-ignored) local competition data
└── submissions/                 # (git-ignored) downloaded kernel outputs
```

---

## 🚀 How to Run

```powershell
# 1. Download competition data (once)
python -m kaggle competitions download arc-prize-2026-arc-agi-3 -p data
Expand-Archive data\arc-prize-2026-arc-agi-3.zip data

# 2. Local environment (Python >= 3.12) with the official toolkit
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install data\arc_agi_3_wheels\arc_agi-0.9.8-py3-none-any.whl data\arc_agi_3_wheels\arcengine-0.9.3-py3-none-any.whl python-dotenv pandas pyarrow

# 3. Edit src/graph_explorer_agent.py and evaluate on the 25 public games
.venv\Scripts\python tools\local_eval.py --data data
.venv\Scripts\python tools\local_eval.py --data data --games ls20,vc33 --max-actions 1500

# 4. Rebuild the notebook and push it to Kaggle (requires the Kaggle CLI)
.venv\Scripts\python tools\build_notebook.py
kaggle kernels push -p .
```

The Kaggle *commit run* plays 3 public games (`ls20`, `vc33`, `r11l`) offline as a smoke test and prints `*** Smoke score`. To score on the leaderboard, **submit the notebook version** (code competition):

```powershell
python -m kaggle competitions submit arc-prize-2026-arc-agi-3 -k rochikarla/arc-prize-2026-arc-agi-3-agent -v <version> -f submission.parquet -m "v2 graph explorer"
```
