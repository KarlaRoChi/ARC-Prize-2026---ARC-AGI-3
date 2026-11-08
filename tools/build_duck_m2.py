"""Build kernel_duck_m2/arc3_duck_m2.ipynb from the vendored upstream notebook.

Upstream: Daniel Franzen, "ARC-AGI-3 Milestone 2 Solution" (Kaggle notebook, Apache 2.0;
code at github.com/da-fr/arc-agi-3-solution, Apache 2.0), built on Tufa Labs' Duck harness
(ARC-AGI-3 Milestone 1 winner). The upstream notebook is kept byte-identical under
kernel_duck_m2/upstream/; this script only:

  1. replaces the header markdown with our own description, credits and eligibility note;
  2. shrinks the *commit run* (interactive "Save & Run" on the public demo games) so a push
     costs ~30 min of RTX PRO 6000 time instead of ~8 h. The competition rerun
     (KAGGLE_IS_COMPETITION_RERUN) keeps the upstream settings unchanged.

Usage:
    python tools/build_duck_m2.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernel_duck_m2"
UPSTREAM = KERNEL_DIR / "upstream" / "franzen_m2_v3.ipynb"
OUT = KERNEL_DIR / "arc3_duck_m2.ipynb"

# Commit run only: a handful of quick public games, one pass, short per-game cap.
COMMIT_GAMES = ["ar25", "ft09", "lp85", "r11l", "vc33"]
COMMIT_MINUTES_PER_GAME = 15
ALL_PUBLIC_GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20",
                    "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87",
                    "tu93", "vc33", "wa30"]

HEADER = """# ARC Prize 2026 — ARC-AGI-3: Duck harness + Qwen3.8-Flash-Next (Milestone 2 stack)

**What this is.** A fork of Daniel Franzen's open-sourced
[ARC-AGI-3 Milestone 2 Solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution)
(write-up: <https://www.kaggle.com/writeups/dfranzen/arc-agi-3-milestone-2>, code:
<https://github.com/da-fr/arc-agi-3-solution>, Apache 2.0), which is itself built on
[Tufa Labs' Duck harness](https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner)
(Milestone 1 winner, <https://github.com/Tufalabs/duck-harness>). An LLM agent
(Qwen3.8-Flash-Next, Intel W4A16 AutoRound, with Albucino's MTP draft head for speculative decoding)
is served locally through the Pennyroyal SGLang fork (Apache 2.0) on one RTX PRO 6000 and plays every
game through the Duck tool-calling harness, with priority scheduling of the LLM slots across games.

**What we changed.** The competition rerun (`KAGGLE_IS_COMPETITION_RERUN`) uses the upstream settings
unchanged (1 pass, 120 concurrent slots, 532 min per game). Only the interactive commit run is shortened
(1 pass on {n_games} public games, {minutes} min per game) so each push costs ~30 min of GPU instead of ~8 h.

**Prize eligibility.** Harness and serving code are Apache 2.0 / CC BY 4.0 (milestone-winner open-source
obligation). The model weights are under the Qwen Community License 1.0, which is not OSI-approved; the
competition rules (section 5) state that pretrained models with an incompatible license do not need to be
re-licensed, and both milestone prizes were awarded on this stack. Logged as *eligible per section 5
(Qwen Community License 1.0 model)*. Internet is off; everything is attached as Kaggle inputs.

All credit for the solver, harness and patches goes to Daniel Franzen, Jeroen Cottaar / Tufa Labs,
John Pezzulli, Mamy Ratsimbazafy, Gabriel Olympie, Albucino, Intel and the Qwen team.
"""


def _src(cell) -> str:
    return "".join(cell["source"])


def _set_src(cell, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def build() -> None:
    nb = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # 1. header
    assert cells[0]["cell_type"] == "markdown" and "About this notebook" in _src(cells[0])
    _set_src(cells[0], HEADER.format(n_games=len(COMMIT_GAMES), minutes=COMMIT_MINUTES_PER_GAME))

    # 2. customization hook: shrink the commit run only (the rerun branch already forces n_passes = 1)
    hook = next(c for c in cells if c["cell_type"] == "code" and "bm.n_passes = 4" in _src(c))
    text = _src(hook)
    replacements = [
        ("bm.n_passes = 4", "bm.n_passes = 1  # commit run: one pass (the rerun branch below also sets 1)"),
        ("demo_excluded_games = [] if TRUE_SUBMISSION else []",
         "# Commit run: keep only a few quick public games (the rerun plays every competition game).\n"
         f"COMMIT_GAMES = {COMMIT_GAMES!r}\n"
         f"demo_excluded_games = [] if TRUE_SUBMISSION else [g for g in {ALL_PUBLIC_GAMES!r} if g not in COMMIT_GAMES]"),
        ("        bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110",
         f"        bm.solver.max_runtime_s_per_game = {COMMIT_MINUTES_PER_GAME}*60  # commit run only"),
    ]
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    _set_src(hook, text)

    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
