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

  - roblox.py (g-315-248, delta): the 3 highest-cross-env-variance slots for
    Roblox NPC exploration -- WorldBuilder (instance-tree -> UnitSet),
    ProximityModel (PATH-distance + learned-displacement projection seam),
    Executor (behavior-tree move-toward). Composes the UNMODIFIED
    `primitives.frontier_coverage.FrontierCoverage`.

Boundary (g-315-236-d handoff): echo extracts + owns the env-agnostic primitive
cores in `primitives/`; the owning agent supplies each environment's slots here
(delta = Roblox, alpha = vinheim/shared). Adding a slot module here NEVER modifies
a `primitives/` core -- the regression gate for the cores is the existing suite.
"""
