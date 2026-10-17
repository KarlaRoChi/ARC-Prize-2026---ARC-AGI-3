# ⚖️ ARC Prize 2026 — ARC-AGI-3: Rules & Prize Eligibility

**Official rules:** <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules>
**Last checked against the official pages:** 2026-10-17
*This is a summary for internal use. The official rules page is authoritative.*

---

## 1. Key terms

| Item | Value |
|---|---|
| Sponsor | ARC Prize Foundation |
| Prize pool | $850,000. **Top-score track $150k:** final LB $40k / $15k / $10k / $5k / $5k, plus two milestones of $25k / $7.5k / $5k each. **Bonus $700k**, only if a team reaches 100 %, split among the top 5 at 100 % |
| Winner license | **CC BY 4.0** |
| Data license | Apache 2.0 |
| Team size | Max 5 |
| Code requirements | Notebook only. CPU or GPU ≤ 9 h. **Internet disabled.** Free, public external data and pre-trained models allowed. The submission file is generated automatically by the gateway |
| Accelerators | RTX 6000 (`g4-standard-48`). Only for this competition, internet must be off. Using them for anything else can lead to an account ban |

## 2. Deadlines (23:59 UTC)

| Date | Milestone |
|---|---|
| 2026-06-30 | Milestone 1 (passed) |
| 2026-09-30 | Milestone 2 (passed) |
| **2026-10-26** | Entry + team merger deadline |
| **2026-11-02** | **Final submission deadline** |
| 2026-12-04 | Winners announced |

## 3. What a submission needs to be prize-eligible

1. **Open source is mandatory.** Open-source system, model and weights, per the OSI Open Source AI Definition. Release the solution under CC BY 4.0.
2. **Milestone prizes** required the notebook to be **published** by the milestone date. Both milestones have passed.
3. Offline rerun ≤ 9 h. The agent must talk only to the competition gateway (`http://gateway:8001`) during the rerun.
4. No hand-labelling or human play of the private games, and no private sharing.
5. **Winner obligations.** Training and inference code, environment description, interview with the sponsor, prize documents.

## 4. Audit of our notebook (2026-10-17)

| Kernel | Internet | External inputs | Eligibility |
|---|---|---|---|
| `rochikarla/arc-prize-2026-arc-agi-3-agent` (`kaggle_arc3_agent.ipynb`) | Off | Only the competition's own `arc_agi_3_wheels` (installed with `--no-index`). No models | ✅ Eligible |

- ✅ The only network call is `curl` to the **local competition gateway** during `KAGGLE_IS_COMPETITION_RERUN`, which is the official pattern. `ARC_API_KEY=test-key-123` is the gateway placeholder, not a real secret.
- ✅ The agent is our own code: a CPU graph explorer with no external weights.
- ⚠️ Publish the notebook (or a repo) under CC BY 4.0 before the prize review. The kernel is currently private.
