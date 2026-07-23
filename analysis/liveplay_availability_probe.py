"""g-315-442 live-ARC-play availability probe — trivial read-only auth check.
Hits the canonical /api/games list endpoint against the real API with the SAME
X-API-Key header shape as adapters.live_arc_transport._headers(). Prints ONLY
status + games count — NEVER the key, NEVER full response bodies. probe-before-defer
/ probe-with-canonical-code-path."""
import os
import sys

import requests

SCHEME = os.environ.get("SCHEME", "https")
HOST = os.environ.get("HOST", "three.arcprize.org")
PORT = os.environ.get("PORT", "443")
root = f"{SCHEME}://{HOST}" if str(PORT) in ("443", "80") else f"{SCHEME}://{HOST}:{PORT}"

key = os.getenv("ARC_API_KEY", "")
if not key:
    print("PROBE: ARC_API_KEY empty in env — cannot authenticate")
    sys.exit(2)
print(f"PROBE: key present (len={len(key)}, masked) | root={root}")

headers = {"X-API-Key": key, "Accept": "application/json"}
try:
    r = requests.get(f"{root}/api/games", headers=headers, timeout=15)
except Exception as e:
    print(f"PROBE: GET /api/games raised {type(e).__name__}: {str(e)[:120]}")
    sys.exit(3)

print(f"PROBE: GET /api/games -> HTTP {r.status_code}")
if r.status_code == 200:
    try:
        data = r.json()
        n = len(data) if isinstance(data, list) else len(data.get("games", []))
        print(f"PROBE: AUTHENTICATED — {n} games available")
        # show only game IDs (public identifiers), capped, never full payload
        ids = [g.get("game_id") if isinstance(g, dict) else str(g)
               for g in (data if isinstance(data, list) else data.get("games", []))][:5]
        print(f"PROBE: sample game_ids: {ids}")
    except Exception as e:
        print(f"PROBE: 200 but JSON parse failed: {type(e).__name__}")
elif r.status_code in (401, 403):
    print("PROBE: REACHABLE but auth REJECTED (key invalid/expired/unauthorized)")
else:
    print(f"PROBE: unexpected status; body[:120]={r.text[:120]!r}")
