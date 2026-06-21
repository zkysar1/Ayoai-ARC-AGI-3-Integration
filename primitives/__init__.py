"""primitives/ -- env-AGNOSTIC exploration primitives shared across environments.

Home for the reusable, environment-independent cores of AyoAI's exploration
behavior, carved out of the ARC-specific solver so the SAME primitive can drive
exploration in any environment (2D ARC grids, 3D Roblox, virtual Vinheim).
Zachary's generalization directive (g-315-236): the exploration techniques the
ARC vertical discovers "should be baked into ayoai, and be applicable for all
environment types."

A primitive in this package knows NOTHING about ARC FrameData, cursors, learned
displacement models, or any environment's perception. It operates on opaque
integer coordinates and action ids plus INJECTED seams (e.g. a projection
callable that maps an action to the cell it would land on). Environment-specific
perception stays in the environment's solver, which COMPOSES the primitive and
supplies the seam.

Members:
  - frontier_coverage.FrontierCoverage (g-315-236-c): usage-balanced
    novelty-seeking turn selection over a visit-count map.
  - reachability_nav.ReachabilityNav (g-315-251): path-distance-aware
    navigation toward a target with BFS routing, greedy fallback, and
    knowledge-conditional stall/abandon/exhaust.
"""
