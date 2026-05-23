# solver_v0 — Design Pointer

`solver_v0/` is the deterministic ARC-AGI-3 solver. The first-principles
strategy choice, input/output shapes, candidate-strategy comparison,
chosen strategy rationale, and offline test surface live in
`design/integration-design.md` **Part 11** (lines 841-1034). This file is
a pointer so a fresh reader landing in `solver_v0/` finds the design
without grep.

## Modules

| Module | Goal | Lines | Role |
|---|---|---|---|
| `solver_v0/perception.py` | g-315-64 (completed) | 182 | Feature extractor: `extract(frame, available_actions, history) -> FrameFeatures`; classifies per-cell role as static / mobile / rare / unknown via churn. Pure function, no I/O. |
| `solver_v0/signatures.py` | g-315-65 (completed) | 239 | Pattern signature registry + 4 seed signatures (sig-12 cross-class available_actions filter, sig-13/14/15 ls20-specific). `filter_actions(candidates, features)` composes filters deterministically. |
| `solver_v0/policy.py` | g-315-66 (completed) | 147 | `HandBuiltPolicy.choose(features) -> int` deterministic action selector. Encodes ls20-class.md Solver Implications (sig-12 gate, ACTION2 noop-skip, ACTION4 rate-limit, ACTION3 default, ACTION1 tiebreaker, RESET fallback). |
| `solver_v0/client_adapter.py` | g-315-67 (completed) | — | Uniform Live / Mock / Replay interface so the same solver code runs against the live AyoAI stream, a deterministic mock, or a `recordings/*.recording.jsonl` replay. |

## Data flow

```
ARC-AGI-3 frame → AyoAI streaming → AyoaiV1StreamClient.choose_action(frame_data)
                                  → client_adapter (Live | Mock | Replay)
                                  → perception.extract() → FrameFeatures
                                  → signatures.filter_actions() → candidates
                                  → policy.choose() → action_id
                                  → Decision dict → streaming response
```

## Compute envelope

Per Part 11 §11.4 (tiny-compute-safe rationale). The "Memory" column
distinguishes INPUT (bytes read per frame) from OUTPUT (per-FrameFeatures
peak after extract). The two are NOT the same — see g-315-92 microbench
(2026-05-22) for measured values.

| Stage | Per-tick wallclock | Input read | Output peak (tracemalloc, single call) |
|---|---|---|---|
| `perception.extract` | ~55 ms @ 64×64 + history=5 | ≤16 KiB grid | **~480 KiB FrameFeatures** (4096 CellAttribute dataclass instances dominate) |
| `signatures.filter_actions` | ~10 µs @ 4 sigs × 7 actions | — | O(1) |
| `policy.choose` | ~19 µs @ history=50 | — | O(\|history\|), append-only |
| `client_adapter` | O(1) per dispatch | — | O(1) |

The 480 KiB output peak fits comfortably under the 8 GB / 2 vCPU envelope
(0.012% of 4 GB RAM, headroom for ~8000 concurrent FrameFeatures). Earlier
revisions of this table conflated input read with output peak and claimed
"≤16 KiB per FrameFeatures" — that was the input number. g-315-92 measured
the output empirically; g-315-93 corrected the conflation. The microbench
in `tests/perf/test_solver_v0_envelope.py` is the regression guard-rail
(threshold set ~2× measured baseline).

If the per-cell dataclass shape ever needs to shrink (e.g., for batch
inference of N frames simultaneously), the lever is parallel arrays
(`values: list[int]`, `roles: list[str]`, `churns: list[float]`) instead
of `list[CellAttribute]`. Expected reduction: ~5× peak (~100 KiB). Filed
as a future Idea goal under asp-315.

No LLM in the hot path. The decision authority for any future BitNet/LLM
seeding sits in `policy.choose` and is explicitly deferred to v2+ per
Part 11 §11.6.

## Test surface

Seven offline contract tests defined in Part 11 §11.5. Run with
`uv run pytest tests/` from this repo. No live ARC backend, no live AyoAI
server — replay corpus in `recordings/*.recording.jsonl` is the ground
truth for tests 2-3 / 7.

## Cross-references

- `design/integration-design.md` Part 11 — first-principles design (g-315-45)
- `world/knowledge/tree/intelligence/ayoai-game-integration/game-system-instances/arc-agi-3/solver-strategy-primer.md` — implementation mechanism (the HOW; this file is the WHAT)
- `rb-1028` (available_actions filter cross-class invariant)
- `rb-1031` (bootstrap-then-score methodology)
