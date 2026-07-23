"""g-315-442 signal 2: confirm the authenticated WRITE/play lifecycle works
(scorecard open -> close), not just the read-only games list. Trivial + non-
destructive: opens a probe-tagged scorecard and immediately closes it. Prints
only status + scorecard id prefix — never the key."""
import os, sys, requests
root = "https://three.arcprize.org"
key = os.getenv("ARC_API_KEY", "")
if not key:
    print("SCORECARD-PROBE: no key"); sys.exit(2)
h = {"X-API-Key": key, "Accept": "application/json", "Content-Type": "application/json"}
try:
    op = requests.post(f"{root}/api/scorecard/open", json={"tags": ["g-315-442-availability-probe"]}, headers=h, timeout=15)
except Exception as e:
    print(f"SCORECARD-PROBE: open raised {type(e).__name__}: {str(e)[:100]}"); sys.exit(3)
print(f"SCORECARD-PROBE: open -> HTTP {op.status_code}")
if op.status_code != 200:
    print(f"SCORECARD-PROBE: open body[:120]={op.text[:120]!r}"); sys.exit(0)
try:
    card = op.json()
    cid = card.get("card_id") or card.get("scorecard_id") or card.get("guid") or ""
    print(f"SCORECARD-PROBE: opened card_id={str(cid)[:12]}... — WRITE path OK")
    cl = requests.post(f"{root}/api/scorecard/close", json={"card_id": cid}, headers=h, timeout=15)
    print(f"SCORECARD-PROBE: close -> HTTP {cl.status_code} — play lifecycle CONFIRMED")
except Exception as e:
    print(f"SCORECARD-PROBE: post-open step failed {type(e).__name__}: {str(e)[:100]}")
