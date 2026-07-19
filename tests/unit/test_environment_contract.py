"""Unit tests for the formal EnvironmentAdapter contract (adapters/base.py).

g-331-01 (alpha). Two things are proven here:

  1. The EXISTING adapters conform to the new contract with ZERO edits -- the
     roblox.py (delta) and vinheim.py (alpha) slot classes already satisfy the
     WorldBuilder / Executor / ProximityModel Protocols structurally. These are
     class-level ``issubclass`` checks: no adapter construction (and so no
     transport fixtures) is required, which keeps the contract decoupled from
     each adapter's constructor.

  2. A NEW environment can "register a conforming adapter" via the
     ``EnvironmentAdapter`` container, which validates each slot at construction
     and rejects a non-conforming one by name -- the ARC-AGI-3 registration path.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import pytest

from adapters.base import (
    Clock,
    ConformanceError,
    EnvironmentAdapter,
    Executor,
    ProximityModel,
    WorldBuilder,
)
from adapters.football import (
    FootballExecutor,
    FootballProximityModel,
    FootballWorldBuilder,
)
from adapters.roblox import RobloxExecutor, RobloxProximityModel, RobloxWorldBuilder
from adapters.vinheim import (
    VinheimExecutor,
    VinheimProximityModel,
    VinheimWorldBuilder,
)
from primitives.frontier_coverage import Cell


# --------------------------------------------------------------------------- #
# 1. Existing adapters conform structurally (zero-edit, byte-identical).        #
# --------------------------------------------------------------------------- #
def test_roblox_slot_classes_conform_to_contract() -> None:
    assert issubclass(RobloxWorldBuilder, WorldBuilder)
    assert issubclass(RobloxExecutor, Executor)
    assert issubclass(RobloxProximityModel, ProximityModel)


def test_vinheim_slot_classes_conform_to_contract() -> None:
    assert issubclass(VinheimWorldBuilder, WorldBuilder)
    assert issubclass(VinheimExecutor, Executor)
    assert issubclass(VinheimProximityModel, ProximityModel)


def test_football_slot_classes_conform_to_contract() -> None:
    """g-335-146: a fourth env registers with zero edits to base.py or primitives/."""
    assert issubclass(FootballWorldBuilder, WorldBuilder)
    assert issubclass(FootballExecutor, Executor)
    assert issubclass(FootballProximityModel, ProximityModel)


# --------------------------------------------------------------------------- #
# 2. Registration container -- minimal conforming + non-conforming fakes.       #
# --------------------------------------------------------------------------- #
class _FakeWorldBuilder:
    def build_units(self, world_state: Mapping[str, object]) -> Sequence[object]:
        return []


class _FakeExecutor:
    def declare_actions(self) -> list[int]:
        return [0]

    def execute(self, decision: object) -> object:
        return decision


class _FakeProximityModel:
    def distance(self, unit_a: object, unit_b: object) -> float:
        return 0.0

    def project_from(self, cell: Cell) -> Callable[[int], Optional[Cell]]:
        return lambda _action: None

    def quantize(self, position: object) -> Cell:
        return cell_zero()

    def record_effect(self, action: int, from_cell: Cell, to_cell: Cell) -> None:
        return None

    def learned_actions(self) -> set[int]:
        return set()

    def set_units(self, units: Sequence[object]) -> None:
        return None


class _NotAWorldBuilder:
    """Missing build_units -- must be rejected by name."""


def cell_zero() -> Cell:
    # frontier_coverage.Cell is an opaque hashable token; a 0-tuple stands in.
    return (0, 0)  # type: ignore[return-value]


def test_environment_adapter_registers_conforming_slots() -> None:
    adapter = EnvironmentAdapter(
        name="fake-env",
        world_builder=_FakeWorldBuilder(),
        executor=_FakeExecutor(),
        proximity_model=_FakeProximityModel(),
    )
    assert adapter.name == "fake-env"
    # Forward slots default to None until the env builds them (Plan 7.2.A phasing).
    assert adapter.clock is None
    assert adapter.knowledge_policy is None
    assert adapter.vocabulary is None


def test_environment_adapter_accepts_real_adapter_slot_classes() -> None:
    # The end-to-end registration path: a real env's slot classes pass the
    # container's runtime conformance check. RobloxProximityModel(cell_size=...)
    # and RobloxWorldBuilder()/RobloxExecutor need no live transport for the
    # ProximityModel + WorldBuilder; Executor requires a transport, so this test
    # uses the cheap-to-construct slots and a fake executor to exercise the
    # container while the issubclass tests above cover RobloxExecutor's class.
    adapter = EnvironmentAdapter(
        name="roblox",
        world_builder=RobloxWorldBuilder(),
        executor=_FakeExecutor(),
        proximity_model=RobloxProximityModel(),
    )
    assert adapter.name == "roblox"


def test_environment_adapter_rejects_nonconforming_mandatory_slot() -> None:
    with pytest.raises(ConformanceError) as exc:
        EnvironmentAdapter(
            name="broken-env",
            world_builder=_NotAWorldBuilder(),  # type: ignore[arg-type]
            executor=_FakeExecutor(),
            proximity_model=_FakeProximityModel(),
        )
    assert "world_builder" in str(exc.value)


def test_environment_adapter_validates_forward_slot_when_supplied() -> None:
    class _NotAClock:
        pass

    with pytest.raises(ConformanceError) as exc:
        EnvironmentAdapter(
            name="bad-clock-env",
            world_builder=_FakeWorldBuilder(),
            executor=_FakeExecutor(),
            proximity_model=_FakeProximityModel(),
            clock=_NotAClock(),  # type: ignore[arg-type]
        )
    assert "clock" in str(exc.value)


def test_environment_adapter_accepts_conforming_forward_slot() -> None:
    class _Clock:
        def on_heartbeat(self, tick: int) -> None:
            return None

    adapter = EnvironmentAdapter(
        name="clocked-env",
        world_builder=_FakeWorldBuilder(),
        executor=_FakeExecutor(),
        proximity_model=_FakeProximityModel(),
        clock=_Clock(),
    )
    assert isinstance(adapter.clock, Clock)
