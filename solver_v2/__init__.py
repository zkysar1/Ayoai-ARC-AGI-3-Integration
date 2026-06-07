"""solver_v2 — episode-seeded deterministic ARC-AGI-3 solver (spine).

Per g-315-134-a (offline-executable v2 spine). solver_v2 inverts solver_v0's
"decide every tick from scratch" model with a two-tier architecture:

  - ONCE PER EPISODE: a SeedProvider produces an EpisodePrior (the "seed").
    The real seed (g-315-134-d) is a BitNet/LLM pass; the spine ships a
    DETERMINISTIC ORACLE stub so the whole pipeline runs + is testable
    offline in-process, exactly like solver_v0's --use-solver-v0.
  - PER TICK: a deterministic executor consumes the EpisodePrior and the
    current FrameFeatures to choose an action. NO LLM is in the per-tick
    hot path — that is the entire point of the v2 design (tiny-compute-safe
    per echo/self.md Constraint 1; the expensive seed is amortized across
    an episode).

Modules (filed under asp-315):
- episode (episode.py) — EpisodePrior, EpisodeContext, EpisodeBoundaryDetector
- seed_provider (seed_provider.py) — SeedProvider interface + deterministic
  oracle stub
- executor (executor.py) — deterministic per-tick executor
- streaming_adapter (streaming_adapter.py) — SolverV2StreamingAdapter,
  conforming to the AyoaiStreamingClient public surface (framework-routed
  per echo/self.md Constraint 2), wired via main.py --use-solver-v2.

Perception is SHARED with solver_v0 (solver_v0.perception.extract) — feature
extraction is decision-source-agnostic, so the spine reuses it rather than
duplicating (single source of truth).

Offline-testable: no HTTP, no DNS, no sockets, no LLM. Pure in-process Python.
"""
