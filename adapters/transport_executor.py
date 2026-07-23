"""adapters/transport_executor.py -- the env-agnostic Executor skeleton.

g-315-449/g-315-452 (echo). The Executor slot's ``execute`` / ``declare_actions`` /
``position`` / ``world_state`` / ``__init__`` were byte-near-identical across all four
adapters (arc / roblox / vinheim / football; verified g-315-452, rb-4884): every one
routes a primitive's ``Decision`` through an injected transport and maps the
``(ok, reason)`` result onto the shared ``Result`` taxonomy. The ONLY differences were
three cosmetic ``reason`` strings (no test asserts on them) and the transport's coordinate
type. This module hoists the shared skeleton into one generic base so each adapter's
Executor COMPOSES it (rb-2166: composed, never modifying the primitive cores) and supplies
only the three reason labels via overridable class attributes.

Cognitive-load win (echo PRIMARY metric per self.md): ~30 logic-LoC drop per adapter (the
subclass shrinks to three class-attr overrides), and the NEXT environment inherits the
entire Executor skeleton instead of re-implementing it -- lowering "cost to add the next
environment". This is the sibling of ``primitives/learned_displacement.py`` (the
ProximityModel seam, g-315-449) and ``adapters/episode.py`` (the shared drive loop,
g-355-72); together they leave each adapter carrying only its genuinely env-specific slots.

Generic over the transport coordinate type: ``TransportExecutor[CoordT]`` holds a
``Transport[CoordT]`` and its ``position()`` returns ``CoordT``, so
``ArcExecutor(TransportExecutor[GridCoord])`` yields ``position() -> GridCoord`` with zero
per-adapter typing. This mirrors ``base.Transport``'s own ``Protocol[CoordT_co]`` shape.

Contract: an adapter's Executor INHERITS ``TransportExecutor[<its coord>]`` and overrides
the three ``_reason_*`` / ``_label_action_space`` class attributes with its own vocabulary.
It needs no ``__init__`` / method bodies of its own. Structural conformance to base.py's
runtime_checkable ``Executor`` Protocol is preserved (the inherited methods satisfy it).

Adoption note (football, g-315-452): arc/roblox/vinheim alias ``Transport[CoordT]``
directly, so their Executors specialize this base cleanly. ``football.py`` declares its own
standalone ``PitchTransport(Protocol)`` (structurally identical -- move/position/world_state
-- but not the base.py ``Transport``); its adopting owner either aliases
``PitchTransport = Transport[Coord]`` or casts at the single construction site (rb-2280),
exactly as it already casts its concrete slots.
"""

from __future__ import annotations

from typing import Generic, Mapping, Sequence, TypeVar

from adapters.base import Decision, Result, Transport

CoordT = TypeVar("CoordT")


class TransportExecutor(Generic[CoordT]):
    """The env-agnostic Executor skeleton: route a Decision through a transport.

    A concrete Executor inherits ``TransportExecutor[<coord>]`` and overrides the three
    class attributes below with its own reason vocabulary. Every method here is
    byte-identical (modulo those three strings) to the per-adapter copies it replaces
    (g-315-452 verified).
    """

    # Overridable reason vocabulary -- each env keeps its own taxonomy words. Defaults
    # match the roblox/vinheim/football wording; arc overrides to its no-op-echo taxonomy.
    _label_action_space: str = "action space"
    _reason_effective: str = "moved"
    _reason_ineffective: str = "blocked"

    def __init__(self, *, transport: Transport[CoordT], actions: Sequence[int]) -> None:
        if not actions:
            raise ValueError("Executor needs a non-empty action space")
        self._transport = transport
        self._actions = list(actions)

    def declare_actions(self) -> list[int]:
        return list(self._actions)

    def execute(self, decision: Decision) -> Result:
        if decision.action not in self._actions:
            return Result(
                outcome="fail",
                reason=f"action {decision.action} not in declared {self._label_action_space}",
                retry_safe=False,
            )
        try:
            ok, reason = self._transport.move(decision.action)
        except Exception as exc:  # transport failure -> unknown (Q10: fail:unconfirmed)
            return Result(outcome="fail", reason=f"transport error: {exc}", retry_safe=False)
        if ok:
            return Result(outcome="success", reason=reason or self._reason_effective, retry_safe=True)
        # A legal-but-ineffective action is safe to retry from a new state.
        return Result(outcome="fail", reason=reason or self._reason_ineffective, retry_safe=True)

    def position(self) -> CoordT:
        return self._transport.position()

    def world_state(self) -> Mapping[str, object]:
        return self._transport.world_state()
