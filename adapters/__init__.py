"""adapters/ -- environment-SPECIFIC slot implementations for the env-agnostic brain.

The env-agnostic exploration primitives live in `primitives/` (e.g.
`primitives.frontier_coverage.FrontierCoverage`). They know NOTHING about any
environment -- they operate on opaque integer Cells + action ids plus INJECTED
seams. To run a primitive in a concrete environment, that environment must supply
the slot implementations the primitive's contract names (the 6-slot
`EnvironmentAdapter`: WorldBuilder / Executor / Clock / ProximityModel /
KnowledgePolicy / Vocabulary -- `universal-environment-abstraction` Plan 7.2.A).

This package is the home for those per-environment slot implementations, kept
SEPARATE from `primitives/` so the agnostic core stays free of env literals
(generalization gate 3). Each environment gets its own module:

  - arc.py: ARC-AGI-3 grid slots (the parity oracle's environment).
  - football.py: a contested entity world, the one non-ARC slot set kept here.

Retired (g-376-57, 2026-09-28): roblox.py and vinheim.py. They were offline
simulations of worlds whose live sessions already reach the vessel through the
env-server's front door. Under the One Body plan the plain code that picks moves
lives in the vessel, so this repo keeps no stand-in for them. Mentions of them in
the other modules' docstrings are design history (the base.py Protocols were
extracted from them). Restore from git if needed:
`git show 46de733:adapters/roblox.py`.

Boundary (g-315-236-d handoff): echo extracts + owns the env-agnostic primitive
cores in `primitives/`; the owning agent supplies each environment's slots here.
Adding a slot module here NEVER modifies a `primitives/` core -- the regression
gate for the cores is the existing suite.
"""
