"""absent != empty for the AYOAI-API-KEY resolver (g-315-540).

The defect this pins: the two consumers that carry an actual play -- the v2
seed provider and the streaming client -- are typed `api_key: str` and were
handed `os.getenv("AYOAI_API_KEY", "")`. With the env var unset that is `""`,
which is NOT None, so the AYO_OPERATOR_KEY fallback was skipped and both ran
UNAUTHENTICATED for the whole session. Nothing raised -- the streaming client
omits the header when empty and the seed provider silently degrades to its
in-process oracle -- so the run looked healthy.

Every test below therefore asserts on the ABSENT-vs-EMPTY distinction, not on
"does the fallback work". A fix that collapses the two (e.g. `api_key or
os.getenv(...)`) makes the fallback work AND silently re-authenticates mock
mode against a real operator key; these tests fail on it.
"""

import pytest

from ayoai_client import resolve_api_key


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("AYOAI_API_KEY", raising=False)
    monkeypatch.delenv("AYO_OPERATOR_KEY", raising=False)


class TestAbsentResolvesViaFallback:
    def test_none_falls_back_to_operator_key(self, monkeypatch):
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        assert resolve_api_key(None) == "op-key"

    def test_no_argument_is_the_same_as_none(self, monkeypatch):
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        assert resolve_api_key() == "op-key"

    def test_alias_still_wins_when_set(self, monkeypatch):
        monkeypatch.setenv("AYOAI_API_KEY", "alias-key")
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        assert resolve_api_key() == "alias-key"

    def test_empty_alias_does_not_shadow_the_operator_key(self, monkeypatch):
        # `or` semantics, asserted rather than assumed: an exported-but-empty
        # alias must not win over a real operator key.
        monkeypatch.setenv("AYOAI_API_KEY", "")
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        assert resolve_api_key() == "op-key"


class TestEmptyIsAChoiceAndIsPreserved:
    """The half a naive `api_key or getenv(...)` fix silently breaks."""

    def test_explicit_empty_stays_empty_even_with_a_key_available(self, monkeypatch):
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        assert resolve_api_key("") == "", (
            "mock mode passes '' deliberately; resolving it to a real operator "
            "key would authenticate a mock run against production"
        )

    def test_empty_and_none_do_not_resolve_the_same_way(self, monkeypatch):
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        assert resolve_api_key("") != resolve_api_key(None)

    def test_explicit_key_is_passed_through_untouched(self, monkeypatch):
        monkeypatch.setenv("AYO_OPERATOR_KEY", "op-key")
        assert resolve_api_key("caller-key") == "caller-key"


class TestNothingAvailable:
    def test_returns_empty_rather_than_raising(self):
        # open_ayoai_session owns the loud failure (`if not resolved_api_key`);
        # the resolver stays total so the seed/streaming consumers can decide
        # for themselves rather than crashing a run at import-adjacent depth.
        assert resolve_api_key() == ""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
