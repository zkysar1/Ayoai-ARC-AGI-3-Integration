"""adapters/provision.py -- the envType -> EnvironmentAdapter provisioner (asp-331).

g-331-02 (alpha, universal-environment-abstraction Plan 7.2.A). "Register an envType in
the provisioner" means: add an entry to the ``_PROVISIONERS`` registry mapping an
environment-type string to a builder that returns a conformance-validated
``EnvironmentAdapter`` (adapters/base.py). ``provision(env_type, **kwargs)`` looks the
builder up and returns the constructed adapter -- so ``provision("arc-agi-3")`` returns
the arc-agi-3 session handle (g-331-02's verification outcome).

The registry is the single, explicit place an environment is registered. ARC-AGI-3
(``adapters/arc.py``, alpha) is registered here. The roblox (delta) and vinheim (alpha)
adapters predate this provisioner and supply their slots directly; their owners register
them via the SAME ``_PROVISIONERS`` entry pattern when an envType-keyed lookup is needed
for them -- this module does not reach across those ownership lanes to register them
unbidden (implementation-discipline: touch only what g-331-02 requires).

guard-795: ``provision("arc-agi-3")`` returns an adapter wired to the offline
``SimulatedArcGrid`` by default (see ``build_arc_adapter``); a live ARC transport must be
injected via ``provision("arc-agi-3", transport=<live>)`` deliberately (g-331-03,
guard-795-gated). Provisioning alone never touches a live backend.
"""

from __future__ import annotations

from typing import Any, Callable

from adapters.arc import build_arc_adapter
from adapters.base import EnvironmentAdapter


class UnknownEnvType(LookupError):
    """Raised by ``provision`` when ``env_type`` has no registered adapter builder."""


# envType -> builder. Each builder returns a conformance-validated EnvironmentAdapter and
# accepts the env's keyword arguments (e.g. arc-agi-3's optional ``transport`` / ``actions``).
_PROVISIONERS: dict[str, Callable[..., EnvironmentAdapter]] = {
    "arc-agi-3": build_arc_adapter,
}


def registered_env_types() -> list[str]:
    """The env-type strings currently registered with the provisioner (sorted)."""
    return sorted(_PROVISIONERS)


def provision(env_type: str, **kwargs: Any) -> EnvironmentAdapter:
    """Return the conformance-validated ``EnvironmentAdapter`` for ``env_type``.

    Looks ``env_type`` up in the registry and delegates to its builder, forwarding any
    keyword arguments (e.g. ``transport=`` / ``actions=`` for arc-agi-3). Raises
    ``UnknownEnvType`` -- naming the registered types -- when ``env_type`` is not
    registered, so a typo fails loudly here rather than silently returning nothing.
    """
    builder = _PROVISIONERS.get(env_type)
    if builder is None:
        raise UnknownEnvType(
            f"no adapter registered for env_type {env_type!r}; "
            f"registered: {registered_env_types()}"
        )
    return builder(**kwargs)
