# 🕹️ ARC Prize 2026 — ARC-AGI-3 (Interactive Reasoning)

> **Kaggle Competition:** [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)  
> **Compute Resource:** 💻 CPU (0 GPU slots) in v1–v2. The competition pool also offers RTX 6000 (`g4-standard-48`) for LLM-based agents.  
> **Internet:** ❌ Disabled  
> **Metric:** Completion × efficiency relative to human baselines, 0–100 (see [OVERVIEW.md](OVERVIEW.md))  
> **Best LB Score:** **`0.19`** (`v1_graph_explorer`, 100% prize-eligible)  
> **Deliverable:** Generated automatically by the Kaggle evaluation gateway during the rerun.  
> **Key Dates:** 2026-10-26 (entry / team merger deadline) · **2026-11-02 (final submission deadline)**

---

## 🎯 Competition Objective

Unlike ARC-AGI-2 (predicting a static grid from demonstration pairs), here the agent **plays unknown interactive video games**:
it receives frames (a grid up to 64×64 with colors 0–15), chooses actions (`RESET`, `ACTION1`–`ACTION7`, where `ACTION6` is a coordinate click `x, y`), and must discover through exploration what each action does, what the goal is, and how to beat each level.

Evaluation runs on **110 private games** unseen by the agent (55 for the public LB, 55 for the private LB). Twenty-five public games are provided in the dataset for local development.

---

## 🏗️ Technical Approach (v1–v2: Graph Explorer Agent)

[`src/graph_explorer_agent.py`](src/graph_explorer_agent.py) models each level as an unknown deterministic finite-state machine:

1. **State Representation:** Hash of the visible 2D grid.
2. **Untried-Action Priority:** Untried actions at the current state are executed first.
3. **Self-Loop & Dead-End Pruning (`v2`):** Transitions that result in `next_state == state` (wall collisions, no-op clicks) are pruned so the agent does not waste actions repeating ineffective moves.
4. **Breadth-First Search (BFS) Frontier Navigation:** Navigates over known transitions to the nearest visited state that still has untried actions.
5. **Object-Centric Clicks (`ACTION6`):** Targets centroids of connected color components in the current frame.
6. **Failure Memory:** Transitions leading to `GAME_OVER` are blacklisted and avoided after reset.
7. **Per-Level Graph Reset:** Clears the state transition graph whenever a level is completed.

> [!NOTE]
> **100% Prize-Eligible (CC BY 4.0):** This agent is written entirely from scratch with zero external weights or unverified dependencies.

---

## 📁 Directory Structure

```text
ARC Prize 2026 - ARC-AGI-3/ARC Prize 2026 - ARC-AGI-3/
├── kaggle_arc3_agent.ipynb      # Kaggle notebook (GENERATED from src/, do not edit by hand)
├── kernel-metadata.json         # Remote kernel metadata (CPU, offline)
├── src/
│   └── graph_explorer_agent.py  # Single source of truth for the agent
├── tools/
│   ├── build_notebook.py        # Compiles src/ → kaggle_arc3_agent.ipynb
│   └── local_eval.py            # Offline evaluation over the 25 public games
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

# 4. Rebuild notebook and push to Kaggle (from workspace root Kaggle/)
.venv\Scripts\python tools\build_notebook.py
python kaggle_manager.py run arc3
```

The Kaggle *commit run* plays 3 public games (`ls20`, `vc33`, `r11l`) offline as a smoke test and prints `*** Smoke score`. To score on the leaderboard, **submit the notebook version** (code competition):

```powershell
python -m kaggle competitions submit arc-prize-2026-arc-agi-3 -k rochikarla/arc-prize-2026-arc-agi-3-agent -v <version> -f submission.parquet -m "v2 graph explorer"
```
