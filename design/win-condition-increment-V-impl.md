# Win-Condition Discovery — Increment V Implementation Design

Source-grounded implementation plan for increment V (live-wire the synthesized
goal_predicate into V4Arm + A/B). Increments I–IV are BUILT (`analysis/`); this
doc grounds the V build in the REAL `solver_v2` internals so the implementer does
not guess the integration surface. Authored g-315-465 (echo).

## The four load-bearing source findings

1. **`goal_predicate`'s `state` is a frozen grid tuple, not a `FrameFeatures`.**
   `StreamingAdapter.set_v4_arm(..., goal_predicate: Callable[[Any], bool])`
   (streaming_adapter.py:603-645) wires the objective seam. V4Arm plans over
   `_v4_state(frame)` (streaming_adapter.py:647): a **hashable encoding of the
   layered ARC grid** — nested lists → nested tuples. With `history_k == 0` the
   state IS the bare frozen current grid (rows of palette values); with
   `history_k >= 1` it is `(current, prev_1, ..., prev_k)` frozen grids,
   None-padded per-episode. So the extractor receives a frozen grid tuple and
   MUST branch on `history_k` to pick `current` (`state[0]` when k≥1, else the
   whole `state`).

2. **`_components` is HUD-stateful and lives on `FrameProcessor`.**
   `FrameProcessor._components(features: FrameFeatures, hud: frozenset[int])`
   (state_graph.py:376-381) returns the canonical sorted
   `(palette_value, size, bbox)` tuples via 4-connected single-colour CC
   labelling, excluding HUD cells + the frequency-derived background. HUD masking
   is accumulated across an episode (change-rate × occupancy joint gate, frozen
   after `_HUD_WARMUP_FRAMES`). A single planning-state extraction CANNOT
   reproduce the live episode's HUD set. **Design decision:** the extractor uses
   an **empty HUD** (`frozenset()`) for single-state evaluation — a documented
   approximation (over-includes HUD counters as components, a conservative,
   deterministic choice). If the live FrameProcessor's frozen HUD is threaded
   through at wiring time, prefer it; the empty-HUD path is the fallback.

3. **`_CONFIG_PRIORS` maps the 3 prior keys to functions over the components list.**
   `_CONFIG_PRIORS = {"orderedness": _config_orderedness, "compression":
   _config_compression_gain, "symmetry": _config_symmetry}` (state_graph.py:283-287).
   Each takes `list[tuple[int,int,tuple[int,int,int,int]]]` (the components) →
   `float`. So priors are computed FROM the extracted components, no second pass.

4. **A `RewardStateMemory` recognizer already occupies the default `goal_predicate`
   seam — and its emptiness IS the score-0 wall.** When `v4_arm` is set with no
   explicit `goal_predicate`, the default is `RewardStateMemory().goal_predicate`
   (streaming_adapter.py:636-645): exact-set membership over states seen at score
   INCREASES. On a game stuck at score 0 there are no increases → the recognizer
   is EMPTY → its predicate matches nothing → the never-goal default → the v3
   fallback (this is exactly why g-315-445's v4 arm scored 0 identically to v2).
   **The synthesized goal_predicate REPLACES this default** via an explicit
   `set_v4_arm(goal_predicate=synthesized)`. The contrast is the whole point:
   RewardStateMemory fires only on OBSERVED-reward states (useless at score 0);
   the synthesized predicate fires on a STRUCTURAL PROXY of the win-state,
   discoverable with zero reward signal.

## The extractor interface (offline-buildable + testable)

```
# analysis/win_condition_extractor.py  (NEW — the increment-II to_state_predicate
# seam's real extractor; imports solver_v2, so it stays in analysis/, NOT primitives/)

def state_to_cc_signature(state, *, history_k: int = 0) -> CCSignature:
    grid = state[0] if history_k >= 1 else state          # frozen current grid
    height = len(grid); width = len(grid[0]) if height else 0
    values = [v for row in grid for v in row]             # flatten
    features = FrameFeatures(height=height, width=width, values=values)  # confirm ctor
    comps = _components_on(features, hud=frozenset())     # empty-HUD approximation
    priors = {k: fn(comps) for k, fn in _CONFIG_PRIORS.items()}
    components = tuple(Component(palette=p, size=s, bbox=b) for (p, s, b) in comps)
    return CCSignature(components=components, priors=priors)
```

`_components_on` wraps the real `FrameProcessor._components` (instantiate a fresh
`FrameProcessor` or refactor `_components` to a module function if it does not
touch `self` beyond `features`/`hud` — CHECK: it reads only `features`+`hud`+
locals, so a thin free-function wrapper is clean). Offline-testable: build a
`FrameFeatures` from a small hand-authored grid, assert the CCSignature's
components/priors match a hand-computed expectation.

## The offline synthesis composition

```
def synthesize_goal_predicate(episodes, *, max_rounds=5, history_k=0) -> Callable[[Any], bool]:
    summary = summarize_session(episodes)                 # increment I
    validation_frames = [(state_to_cc_signature(s, history_k=history_k), score)
                         for (s, score) in buffered_frames(episodes)]
    result = hypothesize_until_viable(summary, HeuristicHypothesizer(),
                                      compile_spec, validation_frames, max_rounds=max_rounds)  # III+IV
    sig_pred = result.predicate                           # Callable[[CCSignature], bool]
    return to_state_predicate(sig_pred,                   # increment II seam
                              lambda s: state_to_cc_signature(s, history_k=history_k))
```

Returns a live-ready `Callable[[Any], bool]` for `set_v4_arm(goal_predicate=...)`.

## The live A/B (the score-breaking test)

1. Buffer episodes on ls20 (offline replay or a warmup live session).
2. `synthesize_goal_predicate(buffered)` → `pred`.
3. `adapter.set_v4_arm(V4Arm(NoOpSynthesizer(), horizon=N), goal_predicate=pred, history_k=3)`.
4. Live play ls20; record score. Baseline arm = default (empty RewardStateMemory).
5. Report the score delta. Requires LIVE ARC play (429 rate limit: ≥12min/720s
   between sessions). Tiny-compute hot path preserved (synthesis is offline outer
   loop; the compiled predicate is a cheap per-state call).

## Open verification points for the implementer (rb-4948 discipline)

- Confirm the `FrameFeatures` constructor signature (height/width/values vs a raw
  frame) — read `state_graph.py` `FrameFeatures` definition before wiring.
- Confirm `_components` reads only `features`+`hud` (no other `self` state) so the
  free-function wrapper is safe; else instantiate a `FrameProcessor`.
- Decide history_k for the A/B (g-355-55 measured k=3 cuts transition aliasing
  ~63%; the design recommends k=3 for the machinery→score A/B).
- The empty-HUD approximation may over-count HUD counters as components; if the
  live A/B underperforms, threading the episode's frozen HUD is the first knob.
