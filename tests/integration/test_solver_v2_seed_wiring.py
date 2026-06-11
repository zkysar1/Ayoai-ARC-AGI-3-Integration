"""Integration test: main.py solver-v2 seed-provider wiring (g-315-158).

Closes the integration-path gap surfaced by the g-315-154 live litmus: the
unit tests in ``tests/unit/test_solver_v2_seed_provider.py`` cover
``BitNetSeedProvider`` in ISOLATION (fake HTTP session), and
``tests/integration/test_solver_v2_mock_game.py`` covers the adapter's
game-loop with the DEFAULT oracle — but neither exercises the main.py wiring
that builds the live BitNet provider from an open AyoAI session and injects it
into ``SolverV2StreamingAdapter`` (replacing the in-process oracle). That
wiring was verified only by the manual live litmus until now.

The seam under test is ``main.build_v2_seed_provider`` (extracted from the
inline construction at the adapter-construction site for exactly this
testability) plus the adapter's ``seed_provider`` injection point.

Asserts:
  - No session  -> None  -> adapter keeps its DeterministicOracleSeedProvider.
  - Open session -> BitNetSeedProvider with the /ArcEpisodeSeed endpoint
    derived from streaming_url (single source of truth, no port literal) and
    the AYOAI-API-KEY forwarded -> adapter's oracle is REPLACED.
"""

from __future__ import annotations

from ayoai_client import AyoaiSessionInfo
from main import build_v2_seed_provider
from solver_v2.seed_provider import (
    BitNetSeedProvider,
    DeterministicOracleSeedProvider,
)
from solver_v2.streaming_adapter import SolverV2StreamingAdapter

_HOST = "ec2-test.example.com"
_STREAMING_URL = f"https://{_HOST}:8787/AyoStreamingUpdates"
_EXPECTED_SEED_ENDPOINT = f"https://{_HOST}:8787/ArcEpisodeSeed"


def _fake_session(streaming_url: str = _STREAMING_URL) -> AyoaiSessionInfo:
    return AyoaiSessionInfo(
        ayo_server_key="card-test",
        ayo_environment_key="arc-agi-3",
        ayoai_hostname=_HOST,
        streaming_url=streaming_url,
        env_server_url=f"https://{_HOST}:8686",
        attempts=1,
        elapsed_s=0.0,
    )


def test_build_returns_none_without_session() -> None:
    """No open session -> None, so the adapter falls back to its oracle."""
    assert build_v2_seed_provider(None) is None
    assert build_v2_seed_provider(None, "any-key") is None


def test_build_returns_bitnet_provider_with_derived_endpoint() -> None:
    """An open session yields a BitNetSeedProvider whose endpoint is derived
    from streaming_url (shared host:port, path /ArcEpisodeSeed) — no port
    literal duplicated in main.py."""
    provider = build_v2_seed_provider(_fake_session(), "")
    assert isinstance(provider, BitNetSeedProvider)
    assert provider.SEED_SOURCE == "bitnet"
    assert provider._endpoint_url == _EXPECTED_SEED_ENDPOINT


def test_build_forwards_api_key() -> None:
    """The AYOAI-API-KEY is forwarded to the provider (authenticates the POST,
    same header as the streaming UPDATE)."""
    no_key = build_v2_seed_provider(_fake_session(), "")
    with_key = build_v2_seed_provider(_fake_session(), "SECRET-KEY")
    assert isinstance(no_key, BitNetSeedProvider)
    assert isinstance(with_key, BitNetSeedProvider)
    assert no_key._api_key == ""
    assert with_key._api_key == "SECRET-KEY"


def test_endpoint_derivation_uses_streaming_host_port() -> None:
    """Endpoint derivation is host:port-agnostic — a different host:port in
    streaming_url flows through to the seed endpoint unchanged except the path
    swap. Guards against a future hard-coded host:port regression."""
    alt_url = "https://other-host:9999/AyoStreamingUpdates"
    provider = build_v2_seed_provider(_fake_session(alt_url), "")
    assert isinstance(provider, BitNetSeedProvider)
    assert provider._endpoint_url == "https://other-host:9999/ArcEpisodeSeed"


def test_adapter_oracle_replaced_when_session_present() -> None:
    """The wiring that the live litmus exercised: an open session -> the
    adapter's seed_provider IS the live BitNetSeedProvider, NOT the default
    in-process oracle."""
    provider = build_v2_seed_provider(_fake_session(), "")
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card-test",
        arc_game_id="ls20-test",
        seed_provider=provider,
    )
    assert isinstance(adapter.seed_provider, BitNetSeedProvider)
    assert adapter.seed_provider.SEED_SOURCE == "bitnet"
    assert not isinstance(
        adapter.seed_provider, DeterministicOracleSeedProvider
    )


def test_adapter_keeps_oracle_when_no_session() -> None:
    """The None-fallback: no session -> adapter retains its in-process
    DeterministicOracleSeedProvider (byte-identical to the pre-g-315-154
    offline behavior)."""
    provider = build_v2_seed_provider(None)
    adapter = SolverV2StreamingAdapter(
        ayo_server_key="card-test",
        arc_game_id="ls20-test",
        seed_provider=provider,
    )
    assert isinstance(adapter.seed_provider, DeterministicOracleSeedProvider)
    assert adapter.seed_provider.SEED_SOURCE == "deterministic-oracle"
