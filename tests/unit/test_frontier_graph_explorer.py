"""Unit tests for the env-agnostic frontier-graph exploration primitive (g-355-45).

Pins the FrontierGraphExplorer 3-tier policy (from arXiv 2512.24156 Algorithm 1, made
env-agnostic): (1) untried action in the CURRENT state; (2) else BFS over OBSERVED edges
to the nearest state with an untried action, first action on the shortest path; (3) else
None (fully explored -> the caller degrades, strict-superset rb-4578). The load-bearing
behaviors: NO predictive model (path-finding over OBSERVED edges only, no projection --
the distinction from frontier_coverage.py), deterministic tie-breaks, self-loop (wall)
safety, and the COMPOSE-with-V4Arm degrade (g-355-44 verdict, rb-4583). The env is
simulated (apply the chosen action to get the next state) to drive the explorer.
"""

from __future__ import annotations

from primitives.frontier_graph_explorer import FrontierGraphExplorer

LINE_ACTIONS = ("L", "R")


def _apply_line(state: int, action: str) -> int:
    """The simulated 1-D line environment: R -> +1, L -> -1."""
    return state + 1 if action == "R" else state - 1


def _prio_R_high(_state, action: str) -> float:
    """Salience seam: R is high-priority, L low."""
    return 1.0 if action == "R" else 0.0


# --------------------------------------------------------------------------- #
# Tier 1: untried action in the current state.                                #
# --------------------------------------------------------------------------- #


def test_tier1_returns_an_untried_action_in_current_state() -> None:
    ex = FrontierGraphExplorer()
    a = ex.next_action(0, LINE_ACTIONS)
    assert a in LINE_ACTIONS  # a fresh state -> some untried action


def test_tier1_uniform_priority_is_stable_first_in_order() -> None:
    """No priority seam -> max() keeps the FIRST action in the caller's order."""
    ex = FrontierGraphExplorer()
    assert ex.next_action(0, LINE_ACTIONS) == "L"  # first in ("L", "R")


def test_tier1_priority_seam_picks_highest_priority_untried() -> None:
    ex = FrontierGraphExplorer(action_priority=_prio_R_high)
    assert ex.next_action(0, LINE_ACTIONS) == "R"  # R has priority 1.0 > L 0.0


def test_tier1_skips_tried_actions() -> None:
    """After observing (0,'L',-1), L is tried -> next_action from 0 returns R."""
    ex = FrontierGraphExplorer()
    ex.observe(0, "L", -1)
    assert ex.next_action(0, LINE_ACTIONS) == "R"  # L tried, R still untried


# --------------------------------------------------------------------------- #
# Tier 2: BFS shortest-path to the nearest frontier (OBSERVED edges only).     #
# --------------------------------------------------------------------------- #


def test_tier2_bfs_heads_toward_the_nearest_frontier() -> None:
    """State 0 is exhausted; a one-hop-away state (1) still has an untried action ->
    the explorer returns the first action on the shortest path to it."""
    ex = FrontierGraphExplorer()
    ex.observe(0, "L", 0)   # L is a wall (self-loop) -> tried
    ex.observe(0, "R", 1)   # R -> 1, tried. State 0 now exhausted (L, R tried).
    ex.observe(1, "L", 0)   # from 1, L tried (back to 0). R still UNTRIED -> 1 is frontier.
    # 0 has no untried action; the nearest frontier is 1, reached via "R".
    assert ex.next_action(0, LINE_ACTIONS) == "R"


def test_tier2_prefers_the_shorter_path() -> None:
    """A distance-1 frontier beats a distance-3 frontier: the explorer heads toward the
    nearer one (BFS guarantees minimum edge distance)."""
    actions = ("A", "B", "C")
    ex = FrontierGraphExplorer()
    # State 0 exhausted; A -> chain (1,2,3), B -> near frontier 4, C -> self-loop.
    ex.observe(0, "A", 1)
    ex.observe(0, "B", 4)
    ex.observe(0, "C", 0)
    # chain 1,2 exhausted; 3 is a distance-3 frontier.
    for s, nxt in ((1, 2), (2, 3)):
        ex.observe(s, "A", nxt)
        ex.observe(s, "B", s)
        ex.observe(s, "C", s)
    ex.observe(3, "A", 2)   # 3 has B, C untried -> frontier at distance 3.
    ex.observe(4, "A", 4)   # 4 has B, C untried -> frontier at distance 1 (via "B").
    assert ex.next_action(0, actions) == "B"  # toward the nearer frontier (4), not "A"


# --------------------------------------------------------------------------- #
# Tier 3: fully explored -> None (strict-superset degrade).                    #
# --------------------------------------------------------------------------- #


def test_tier3_exhausted_reachable_graph_returns_none() -> None:
    """Every reachable state has every action tried -> None (the caller degrades)."""
    ex = FrontierGraphExplorer()
    # A 2-state closed world: 0 and 1, both actions tried from both, no untried anywhere.
    ex.observe(0, "L", 0)   # wall
    ex.observe(0, "R", 1)
    ex.observe(1, "L", 0)
    ex.observe(1, "R", 1)   # wall
    assert ex.next_action(0, LINE_ACTIONS) is None


def test_self_loop_wall_is_tried_and_bfs_terminates() -> None:
    """An action that returns to the same state is marked tried (tier 1 won't re-pick
    it) and BFS does not loop on it."""
    ex = FrontierGraphExplorer()
    ex.observe(0, "L", 0)   # wall self-loop
    # L is tried; R still untried -> tier 1 returns R (not the wall L).
    assert ex.next_action(0, LINE_ACTIONS) == "R"
    ex.observe(0, "R", 0)   # R is ALSO a wall now -> 0 fully exhausted, only state.
    assert ex.next_action(0, LINE_ACTIONS) is None  # BFS terminates, no infinite loop


def test_successor_and_seen_states_inspection() -> None:
    ex = FrontierGraphExplorer()
    ex.observe(0, "R", 1)
    assert ex.successor(0, "R") == 1
    assert ex.successor(0, "L") is None      # untried
    assert ex.seen_states == frozenset({0, 1})


# --------------------------------------------------------------------------- #
# Env-agnostic contract: a different state/action encoding.                    #
# --------------------------------------------------------------------------- #


def test_env_agnostic_grid_encoding() -> None:
    """A 2-D grid env (tuple states, compass actions) drives the SAME explorer -- no env
    semantics in the core."""
    actions = ("N", "S", "W", "E")
    ex = FrontierGraphExplorer()
    assert ex.next_action((0, 0), actions) == "N"  # tier 1: uniform -> first in order
    # Exhaust (0,0): N/S/W are walls (self-loops), only E opens to a NEW state. This
    # isolates a SINGLE reachable frontier so tier 2 must return the right action (if all
    # four successors were fresh they would be equidistant frontiers -> deterministic
    # first-in-order "N", which tests nothing about path direction).
    ex.observe((0, 0), "N", (0, 0))  # wall
    ex.observe((0, 0), "S", (0, 0))  # wall
    ex.observe((0, 0), "W", (0, 0))  # wall
    ex.observe((0, 0), "E", (0, 1))  # opens east to a fresh state
    # (0,0) is exhausted; the ONLY reachable frontier is (0,1) via "E" (tier 2 BFS).
    assert ex.next_action((0, 0), actions) == "E"


def test_full_coverage_sweep_reaches_all_then_exhausts() -> None:
    """Integration: an observe->next_action loop over a small bounded line sweeps every
    reachable state, then returns None (fully explored)."""
    ex = FrontierGraphExplorer()

    def clamp(x: int) -> int:
        return max(-2, min(2, x))  # walls at -2 and 2

    state = 0
    for _ in range(50):  # generous bound; the world has 5 cells
        a = ex.next_action(state, LINE_ACTIONS)
        if a is None:
            break
        nxt = clamp(_apply_line(state, a))
        ex.observe(state, a, nxt)
        state = nxt
    assert ex.next_action(state, LINE_ACTIONS) is None       # exhausted
    assert ex.seen_states == frozenset({-2, -1, 0, 1, 2})    # every cell reached


# --------------------------------------------------------------------------- #
# COMPOSE with V4Arm: the explorer is the arm's degrade target (rb-4583).      #
# --------------------------------------------------------------------------- #


def test_composition_with_v4arm_degrade() -> None:
    """The compose pattern (g-355-44 verdict): the caller wires the explorer as V4Arm's
    fallback, so a plan-less (cold) arm degrades to a FRONTIER action rather than a blind
    v3 default -- upgrading the rb-4578 strict-superset degrade target."""
    from primitives.v4_arm import V4Arm
    from primitives.world_model_synthesizer import NoOpSynthesizer

    explorer = FrontierGraphExplorer()
    arm = V4Arm(NoOpSynthesizer(), horizon=4)  # NoOp -> cold model -> never plans
    state = 0
    # Caller computes the fallback from the explorer (frontier action), then v3 default.
    fb = explorer.next_action(state, LINE_ACTIONS) or "v3_default"
    action = arm.step(state, lambda s: s == 999, LINE_ACTIONS, fallback_action=fb)
    # The cold arm has no plan -> it degrades to the explorer's frontier action.
    assert action == fb
    assert action in LINE_ACTIONS      # a real untried action, NOT "v3_default"
    assert fb != "v3_default"          # the explorer had an untried action to offer
