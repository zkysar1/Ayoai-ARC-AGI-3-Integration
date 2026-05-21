"""solver_v0 — hand-built deterministic ARC-AGI-3 solver.

Per g-315-05 + g-315-63 inventory: solver_v0 is the v0 deterministic
solver (no LLM hot path) consisting of perception, pattern signatures,
policy, and client_adapter. Each component is offline-testable against
the ls20 random recording.

Modules (filed under asp-315):
- perception (g-315-64) — feature extractor + role hints
- signatures (g-315-65) — seed pattern signatures
- policy (g-315-66) — hand-built deterministic policy
- client_adapter (g-315-67) — uniform Live/Mock/Replay interface
"""
