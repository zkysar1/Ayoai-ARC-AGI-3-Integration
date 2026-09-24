"""The one per-game action budget every way this repo plays a game defaults to.

ARC-AGI-3 puts no cap on the actions in a game (rb-3262). A GAME_OVER ends one
attempt at a level, not the game: RESET after an attempt restarts the current
level and keeps the levels already won (arcengine base_game.handle_reset calls
level_reset unless no action was taken or the game is won, and always when
ONLY_RESET_LEVELS is set). A completed level
scores (baseline_actions / actions_taken) ** 2 * 100, capped at 115
(arc_agi/scorecard.py), so extra actions only lower the score of a level
that gets finished anyway. Stopping early means the rest of the game scores zero.

So the budget is only a loop guard, and it is sized so that it cuts nothing off
while a level is still worth finishing: one attempt of HUMAN_MULTIPLE human
lengths at each of LEVELS_ASSUMED levels. The old default of 80 was the
reference agent kit's guard against looping forever, not an ARC limit (rb-1521).

Entry points that default to DEFAULT_ACTION_BUDGET (g-376-05):
main.py (run_game_loop and --max-actions), offline_run.py (--max-actions) and
kaggle_salvage/my_agent.py (MyAgent.MAX_ACTIONS) count every action they send,
RESET included. adapters/arc.py (run_arc_episode) and
adapters/live_arc_transport.py (run_live_arc_episode and --max-ticks) count ticks
instead; a tick that follows a GAME_OVER also sends a RESET.
tests/unit/test_action_budget.py fails if one of them goes back to a number of its own.
"""

from __future__ import annotations

HUMAN_ACTIONS_PER_LEVEL: int = 40
"""The solver's conservative estimate of a human's actions per level
(solver_v2.state_graph._RHAE_HUMAN_BASELINE; the test pins the two together).
The real per-level baselines are in the scorecard, not in the frame."""

HUMAN_MULTIPLE: int = 5
"""Human lengths budgeted per level. A level finished at five times the human
count scores (1/5) ** 2 = 4% of a human-paced finish, so more buys almost
nothing. Same multiple as solver_v2.state_graph._RHAE_MULT."""

LEVELS_ASSUMED: int = 10
"""Levels budgeted per game. ls20 reports win_levels=7
(tests/fixtures/offline_frame_ls20.json); ten leaves room for longer games."""

DEFAULT_ACTION_BUDGET: int = HUMAN_ACTIONS_PER_LEVEL * HUMAN_MULTIPLE * LEVELS_ASSUMED
"""40 * 5 * 10 = 2000 actions per game, set by g-376-05."""
