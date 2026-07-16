"""solver_v2/click_prior.py — Async action-effect click-prior (g-315-367).

Ports the Stochastic Goose action-effect CNN coordinate head
(DriesSmit/ARC3-solution, MIT — the ARC-AGI-3 preview-competition winner) as an
ASYNC learned click-prior for the DeterministicExecutor's untrusted-click
branch. Validated offline first (g-315-366): trained on ~1.6k random-click
triples per game, the coordinate head ranks frame-changing clicks at AUC
0.975/0.973 (ft09/lp85) with top-decile precision lift x6.6/x10.2 over random
clicking. Under RHAE quadratic scoring (action efficiency IS the per-level
score), click-waste reduction of that magnitude is a direct score lever on the
6 label-balance-qualifying games.

Architecture constraints (all load-bearing):

- HOT PATH IS TORCH-FREE. The measured full-model forward is 18.7 ms on cc-03
  — 9.4x OVER the 2000 us decide() envelope (g-315-363) — so NO tensor work
  ever happens on the decide path. ``observe()`` / ``suggest()`` are O(1)
  plain-Python; training AND scoring run in a LEARNER SUBPROCESS that streams
  back pre-ranked coordinate lists the hot path merely indexes.
- LEARNER IS A SUBPROCESS, NOT A THREAD. Two reasons, both proven in this
  goal's execution trace. (1) Crash isolation: torch 2.13 CPU on py3.12
  nondeterministically SIGSEGV/SIGABRTs at interpreter exit when its kernels
  only ever ran on a spawned thread ("terminate called without an active
  exception"; reproduced 6x, survived three antidotes — main-thread import,
  main-thread kernel, main-thread backward). In-process, that class of crash
  would kill the SOLVER mid-play; in a child process it is invisible. (2) The
  1.3 s/step CPU training never contends with the solver's GIL. The spawn
  context is used explicitly (portable; no COW state duplication). Caveat any
  ``multiprocessing`` user knows: flag-ON driver scripts must be import-safe
  (guard their entry point with ``if __name__ == "__main__"``).
- TORCH IS OPTIONAL. The package does not depend on torch (pyproject carries
  dotenv/pydantic/requests only). The CHILD lazy-imports torch; when
  unavailable it reports "disabled" and the engine permanently self-disables —
  the executor's coverage sweep is byte-identical to the flag-OFF path.
- DEFAULT OFF. Enabled only via the adapter kwarg or env
  ``SOLVER_V2_CLICK_PRIOR`` (same reversible-toggle pattern as
  ``SOLVER_V2_STATE_GRAPH``, g-315-230).
- RUNTIME LABEL-BALANCE GATE (guard-818 discipline). The g-315-366 audit
  showed 11/25 games are degenerate all-positive (~every click "changes" the
  frame — e.g. ls20 at 0.997), where the frame_changed label carries no signal
  and the CNN is inert. The engine self-gates on a rolling changed-rate over
  its observed clicks: only when the rate is < 0.8 (the audit's qualifying
  threshold) does it train or suggest. Degenerate games keep the pure
  coverage sweep automatically — no per-game configuration.
- HELD-OUT-AUC PUBLISH GATE. A ranking goes live ONLY once the model's
  held-out AUC (every _EVAL_NTH-th observation PER CLASS is never trained on;
  stratified because the target games are exactly the low-positive-rate ones)
  clears _PUBLISH_MIN_AUC with >= 3 samples of each class. The first
  validation run proved the necessity: a 4-step noise ranking driving clicks
  scored WORSE than random (ft09 0.086 vs 0.112 changed-rate). Until the gate
  opens, suggest() returns None and the coverage sweep floor holds. During
  warmup the learner trains CONTINUOUSLY on the buffered corpus (the
  validated recipe is ~150 steps over a static corpus; idling for new samples
  was the first-validation failure mode), then drops to new-data-only
  maintenance rounds.
- GOOSE PER-LEVEL RESET. Level dynamics differ; on a score increase the
  adapter calls ``reset()`` (buffer + model + prior + gate all cleared),
  mirroring Goose's buffer/model reset at each new level.

The slim model is the EXACT ActionModel coordinate path (conv1-4 backbone +
coord_conv1-4 head) with the parallel action-type head removed: the 65,536->512
``action_fc`` alone is 33.5M of the 34.3M params (98%) and contributes nothing
to coordinate logits (parallel heads off conv4). Training recipe is the
validated g-315-366 one: BCE-with-logits on the taken coordinate's logit plus
1e-5 coordinate-entropy regularization, Adam lr=1e-4, hash-deduped buffer.

Offline-testable: the engine's gating/ranking/dedup logic is pure Python;
learner-process behavior is exercised by a skip-marked smoke test and the
goose-venv validation harness (analysis/click_prior_validation_g315367.py).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import deque
from typing import Any, Optional

# ARC click space (structs.py contract: ACTION6 x, y each in [0, 63]).
_GRID = 64
_CELLS = _GRID * _GRID

# g-315-366 qualifying threshold: games whose random-click frame_changed rate
# is >= this are degenerate for the frame-change label (CNN inert — guard-818).
_GATE_MAX_CHANGED_RATE = 0.8
# Minimum observed clicks before the gate can OPEN (avoid deciding the game's
# label balance from a handful of clicks).
_GATE_MIN_SAMPLES = 30
# Rolling window for the changed-rate estimate.
_GATE_WINDOW = 200

# Experience buffer bound (Goose uses ~200k on GPU; cc-03 live play needs far
# less — 2048 x ~4 KiB grids ~= 9 MiB, and g-315-366 trained to AUC 0.97 on
# ~1.6k triples).
_BUFFER_MAX = 2048
# Held-out evaluation buffer: every _EVAL_NTH insertion PER CLASS is held out
# (never trained on) so publish-readiness is measured on unseen data,
# mirroring the g-315-366 80/20 split (stratified — see docstring).
_EVAL_MAX = 400
_EVAL_NTH = 5
# Maintenance cadence AFTER warmup: retrain when this many NEW observations
# have arrived (during warmup the learner trains CONTINUOUSLY on the buffer).
_TRAIN_EVERY = 16
# Steps per training round / batch size (g-315-366 recipe at live cadence).
_STEPS_PER_ROUND = 8
_BATCH = 32
# Minimum buffered triples before the FIRST training round.
_MIN_TRAIN_SAMPLES = 64
# Continuous-training step budget per level (the validated offline recipe
# reached AUC ~0.97 at 150 steps; beyond it, train only on new data).
_WARMUP_STEPS = 150
# Publish gate: held-out AUC must clear this bar (>= 3 samples of each class).
_PUBLISH_MIN_AUC = 0.75
_EVAL_MIN_SAMPLES = 30
# Published ranking depth. Top-256 of 4096 is the top ~6% — inside the
# validated top-decile lift band, deep enough that the rank-walk rarely wraps
# between publish refreshes.
_TOP_K = 256
# Every Nth suggestion slot falls back to the coverage sweep so the buffer
# keeps collecting off-prior (negative-rich) triples — deterministic
# exploration/exploitation interleave.
_EXPLORE_INTERLEAVE = 4

_SEED = 20260715


def _env_flag(name: str) -> bool:
    """Truthy env-var toggle, same convention as SOLVER_V2_STATE_GRAPH."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _grid_to_bytes(grid: Any) -> Optional[bytes]:
    """Flatten a layered ARC grid to 4096 row-major palette bytes.

    Takes ``FrameData.frame`` (list of 2-D layers; the LAST layer is the
    settled animation frame — same convention as the g-315-366 collector) or a
    bare 2-D grid. Pads to 64x64 with 0. Cell values are stored as raw bytes:
    the ARC palette contract is [0, 15]; a contract-violating value >15 packs
    as-is and simply one-hots to NO channel in the learner (16 channels
    compare 0..15) — benign for an input representation. Returns None for an
    empty/malformed grid — callers skip the observation rather than raise (a
    prior must never kill the solver).
    """
    try:
        layer = grid
        # Layered form: [layers][rows][cols] -> take the last layer.
        if layer and isinstance(layer[0], list) and layer[0] and isinstance(layer[0][0], list):
            layer = layer[-1]
        if not layer or not isinstance(layer[0], list):
            return None
        try:
            # Fast path: bytes(list_of_ints) is C-speed (~10x cheaper than a
            # per-cell Python loop: ~200 us -> ~20 us on a 64x64 grid — the
            # dominant cost of the per-tick observe()). Ints outside [0, 255]
            # raise and fall through to the byte-coercing loop below.
            packed = b"".join(
                bytes(row[:_GRID]).ljust(_GRID, b"\0") for row in layer[:_GRID]
            )
            return packed.ljust(_CELLS, b"\0")
        except (TypeError, ValueError):
            pass
        out = bytearray(_CELLS)
        for r, row in enumerate(layer[:_GRID]):
            base = r * _GRID
            for c, v in enumerate(row[:_GRID]):
                out[base + c] = int(v) & 0xFF
        return bytes(out)
    except (TypeError, ValueError, IndexError):
        return None


def _learner_main(
    conn: Any, seed: int, warmup_steps: int, min_publish_auc: float
) -> None:
    """Learner subprocess entry point (mp spawn target — module-level).

    Owns ALL torch state. Receives observation messages from the parent over
    ``conn`` (a multiprocessing Pipe end), trains the slim coordinate model,
    and streams back progress + AUC-gated ranking publishes. Message protocol
    (tuples; first element is the kind):

      parent -> child:
        ("obs", grid_bytes, coord_idx, label, is_eval)  deduped observation
        ("gate", changed_int)                           dedup-dropped click
                                                        (gate stats only)
        ("reset",)                                      level-up: clear all
        None                                            shutdown
      child -> parent:
        ("progress", rounds, steps, auc_or_None, buf_len, eval_len)
        ("published", generation, ranked_xy_list)
        ("disabled", reason)                            terminal

    Exit-crash containment: this process may abort inside torch teardown at
    exit (the documented reason it IS a process) — the parent never inspects
    its exit code, only the message stream.
    """
    try:
        import torch  # type: ignore[import-not-found]
        from torch import nn
        from torch.nn import functional as F  # type: ignore[import-not-found]
    except Exception:  # ImportError or a broken install — same outcome
        try:
            conn.send(("disabled", "torch-unavailable"))
        except (OSError, ValueError, BrokenPipeError):
            pass
        return
    try:
        import random as _random

        torch.manual_seed(seed)
        rng = _random.Random(seed)

        def build_model() -> Any:
            # EXACT ActionModel coordinate path (conv1-4 + coord_conv1-4),
            # action-type head removed (parallel branch; 98% of params, no
            # contribution to coordinate logits).
            return nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 32, 1), nn.ReLU(),
                nn.Conv2d(32, 1, 1),
            )

        model = build_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        channels = torch.arange(16, dtype=torch.uint8).view(16, 1, 1)

        def one_hot(b: bytes) -> Any:
            g = torch.frombuffer(bytearray(b), dtype=torch.uint8).view(_GRID, _GRID)
            return (g.unsqueeze(0) == channels).float()

        def rank_auc(scores: list[float], labels: list[float]) -> Optional[float]:
            """Rank-based (Mann-Whitney) AUC, pure Python — small n.

            Requires >= 3 samples of EACH class: with a single held-out
            positive the AUC is just that sample's rank percentile — noisy
            enough to spuriously clear the publish bar.
            """
            pos = sum(1 for lab in labels if lab > 0.5)
            neg = len(labels) - pos
            if pos < 3 or neg < 3:
                return None
            order = sorted(range(len(scores)), key=lambda i: scores[i])
            rank_sum = sum(r + 1 for r, i in enumerate(order) if labels[i] > 0.5)
            return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)

        buf: deque[tuple[bytes, int, float]] = deque(maxlen=_BUFFER_MAX)
        ev: deque[tuple[bytes, int, float]] = deque(maxlen=_EVAL_MAX)
        gate: deque[int] = deque(maxlen=_GATE_WINDOW)
        gate_sum = 0
        latest: Optional[bytes] = None
        pending = 0
        total_steps = 0
        rounds = 0
        generation = 0

        def gate_push(changed: int) -> None:
            nonlocal gate_sum
            if len(gate) == gate.maxlen:
                gate_sum -= gate[0]
            gate.append(changed)
            gate_sum += changed

        def gate_open() -> bool:
            n = len(gate)
            return n >= _GATE_MIN_SAMPLES and (gate_sum / n) < _GATE_MAX_CHANGED_RATE

        while True:
            ready = (
                len(buf) >= _MIN_TRAIN_SAMPLES
                and gate_open()
                and (total_steps < warmup_steps or pending >= _TRAIN_EVERY)
            )
            # Drain the pipe. Block (1s slices) only when there is no
            # training work; otherwise a zero-timeout sweep.
            drained_shutdown = False
            while conn.poll(0 if ready else 1.0):
                msg = conn.recv()
                if msg is None:
                    drained_shutdown = True
                    break
                kind = msg[0]
                if kind == "obs":
                    _, b, coord, label, is_eval = msg
                    (ev if is_eval else buf).append((b, coord, label))
                    latest = b
                    pending += 1
                    gate_push(1 if label > 0.5 else 0)
                elif kind == "gate":
                    gate_push(msg[1])
                elif kind == "reset":
                    buf.clear()
                    ev.clear()
                    gate.clear()
                    gate_sum = 0
                    latest = None
                    pending = 0
                    total_steps = 0
                    model = build_model()
                    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
                # Re-evaluate readiness against the new state.
                ready = (
                    len(buf) >= _MIN_TRAIN_SAMPLES
                    and gate_open()
                    and (total_steps < warmup_steps or pending >= _TRAIN_EVERY)
                )
            if drained_shutdown:
                return
            if not ready or latest is None:
                continue

            pending = 0
            snapshot = list(buf)
            model.train()
            for _ in range(_STEPS_PER_ROUND):
                batch = (
                    rng.sample(snapshot, _BATCH)
                    if len(snapshot) > _BATCH
                    else snapshot
                )
                xb = torch.stack([one_hot(b) for b, _, _ in batch])
                idx = torch.tensor([c for _, c, _ in batch]).unsqueeze(1)
                yb = torch.tensor([lab for _, _, lab in batch])
                opt.zero_grad()
                logits = model(xb).flatten(1)  # (B, 4096)
                sel = logits.gather(1, idx).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(sel, yb)
                # g-315-366 recipe: light coordinate-entropy regularization.
                loss = loss - 1e-5 * torch.sigmoid(logits).mean()
                loss.backward()
                opt.step()
            total_steps += _STEPS_PER_ROUND
            rounds += 1

            # Held-out AUC -> publish decision. An undertrained/unpredictive
            # model keeps the parent's _published=None -> suggest() returns
            # None -> the coverage sweep floor (never worse than flag-OFF).
            model.eval()
            auc: Optional[float] = None
            eval_snapshot = list(ev)
            if len(eval_snapshot) >= _EVAL_MIN_SAMPLES:
                scores: list[float] = []
                with torch.no_grad():
                    for i in range(0, len(eval_snapshot), _BATCH):
                        chunk = eval_snapshot[i : i + _BATCH]
                        xb = torch.stack([one_hot(b) for b, _, _ in chunk])
                        idx = torch.tensor([c for _, c, _ in chunk]).unsqueeze(1)
                        logits = model(xb).flatten(1)
                        sel = logits.gather(1, idx).squeeze(1)
                        scores.extend(torch.sigmoid(sel).tolist())
                auc = rank_auc(scores, [lab for _, _, lab in eval_snapshot])
            conn.send(("progress", rounds, total_steps, auc, len(buf), len(ev)))
            if auc is not None and auc >= min_publish_auc:
                with torch.no_grad():
                    cell_scores = model(one_hot(latest).unsqueeze(0)).flatten()
                    top = torch.topk(cell_scores, _TOP_K).indices.tolist()
                ranked = [(i % _GRID, i // _GRID) for i in top]
                generation += 1
                conn.send(("published", generation, ranked))
    except Exception as e:  # fail-safe: a prior bug must never kill play
        try:
            conn.send(("disabled", f"worker-error: {type(e).__name__}: {e}"))
        except (OSError, ValueError, BrokenPipeError):
            pass


class ClickPriorEngine:
    """Async learned click-prior with a torch-free hot path.

    Main-process API (the ONLY methods the adapter/executor call):

    - ``observe(grid, x, y, changed)`` — record the outcome of a previous
      ACTION6 click. O(1): dedup + counters + one pipe send.
    - ``suggest(click_index, width, height)`` — a prior-ranked (x, y) for this
      click, or None (caller falls back to the coverage sweep). O(K) worst
      case over the published top-K list; no tensor work.
    - ``reset()`` — new level (score increased): drop all learned state.
    - ``close()`` — stop the learner subprocess (idempotent).
    - ``stats()`` — small observability dict for provenance/validation.

    Process model: ALL torch state lives in a spawned learner subprocess (see
    module docstring for why a thread is not safe). The parent keeps only
    plain-Python bookkeeping: the label-balance gate window, the dedup set,
    buffer-size mirrors, and the last published ranking. Child messages are
    drained opportunistically (non-blocking ``poll``) inside observe/suggest/
    stats — no reader thread. Any child failure (or a dead pipe) permanently
    disables the engine; the solver's flag-OFF behavior is the floor.
    """

    def __init__(
        self,
        enabled: bool | None = None,
        *,
        seed: int = _SEED,
        warmup_steps: int = _WARMUP_STEPS,
        min_publish_auc: float = _PUBLISH_MIN_AUC,
    ) -> None:
        if enabled is None:
            enabled = _env_flag("SOLVER_V2_CLICK_PRIOR")
        self._enabled = bool(enabled)
        self._seed = seed
        self._warmup_steps = warmup_steps
        self._min_publish_auc = min_publish_auc
        # Hash-dedup on (grid_bytes, coord) — the Goose dedup key. Data lives
        # in the child; the parent keeps the dedup set + size mirrors.
        self._seen: set[tuple[int, int]] = set()
        self._inserts_pos = 0
        self._inserts_neg = 0
        self._buf_len = 0
        self._eval_len = 0
        # Rolling label-balance gate window (1 = click changed the frame).
        self._gate_window: deque[int] = deque(maxlen=_GATE_WINDOW)
        self._gate_sum = 0
        # Published ranking: (generation, [(x, y), ...]).
        self._published: Optional[tuple[int, list[tuple[int, int]]]] = None
        self._suggested = 0  # suggestions actually served (observability)
        self._rounds = 0  # learner rounds completed (from progress msgs)
        self._total_steps = 0
        self._last_auc: Optional[float] = None
        self._proc: Optional[Any] = None
        self._conn: Optional[Any] = None
        self._closed = False
        self._disabled_reason: Optional[str] = None

    # ---------- hot-path API ---------- #

    @property
    def enabled(self) -> bool:
        return self._enabled

    def observe(self, grid: Any, x: int, y: int, changed: bool) -> None:
        """Record a completed click's frame-change outcome (O(1), torch-free)."""
        if not self._enabled or self._closed:
            return
        b = _grid_to_bytes(grid)
        if b is None or not (0 <= x < _GRID and 0 <= y < _GRID):
            return
        if self._proc is None:
            self._start_learner()
            if not self._enabled:
                return
        self._drain_child()
        coord = y * _GRID + x
        self._gate_window_push(1 if changed else 0)
        key = (hash(b), coord)
        if key in self._seen:
            self._send(("gate", 1 if changed else 0))
            return
        self._seen.add(key)
        # Stratified split: every _EVAL_NTH-th insertion OF EACH CLASS is
        # held out, so eval holds ~20% of the positives even at a 2-3%
        # positive rate (AUC needs both classes).
        if changed:
            self._inserts_pos += 1
            count = self._inserts_pos
        else:
            self._inserts_neg += 1
            count = self._inserts_neg
        is_eval = count % _EVAL_NTH == 0
        if is_eval:
            self._eval_len = min(self._eval_len + 1, _EVAL_MAX)
        else:
            self._buf_len = min(self._buf_len + 1, _BUFFER_MAX)
        self._send(("obs", b, coord, 1.0 if changed else 0.0, is_eval))

    def suggest(
        self, click_index: int, width: int = _GRID, height: int = _GRID
    ) -> Optional[tuple[int, int]]:
        """A prior-ranked click for this tick, or None -> coverage sweep.

        Deterministic given the published ranking: walks the ranked list by
        ``click_index`` (top-1 first), reserving every ``_EXPLORE_INTERLEAVE``-th
        slot for the sweep (returns None) so observation stays diverse. Returns
        None whenever the prior cannot or should not drive: disabled, gate
        closed (degenerate label balance), or nothing published yet.
        """
        if not self._enabled or self._closed:
            return None
        if click_index % _EXPLORE_INTERLEAVE == _EXPLORE_INTERLEAVE - 1:
            return None  # deterministic exploration slot
        self._drain_child()
        published = self._published
        if published is None:
            return None
        if not self._gate_open():
            return None
        _, ranked = published
        if not ranked:
            return None
        # Rank-walk: consecutive prior slots advance down the ranking; the
        # interleave slots (None above) keep click_index advancing so the walk
        # position is (click_index - slots_reserved_so_far).
        pos = click_index - (click_index // _EXPLORE_INTERLEAVE)
        for offset in range(len(ranked)):
            x, y = ranked[(pos + offset) % len(ranked)]
            if x < width and y < height:
                self._suggested += 1
                return (x, y)
        return None

    def reset(self) -> None:
        """New level: drop all learned state (Goose per-level reset)."""
        self._seen.clear()
        self._inserts_pos = 0
        self._inserts_neg = 0
        self._buf_len = 0
        self._eval_len = 0
        self._gate_window.clear()
        self._gate_sum = 0
        self._published = None
        self._total_steps = 0
        self._last_auc = None
        if self._proc is not None:
            self._send(("reset",))

    def close(self) -> None:
        """Stop the learner subprocess (idempotent).

        The child's EXIT CODE is deliberately ignored — torch teardown may
        abort inside the child (the reason it is a process at all); only the
        message stream matters.
        """
        if self._closed:
            return
        self._closed = True
        if self._conn is not None:
            try:
                self._conn.send(None)
            except (OSError, ValueError, BrokenPipeError):
                pass
        if self._proc is not None:
            self._proc.join(timeout=5.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass

    def stats(self) -> dict[str, Any]:
        """Small observability snapshot (provenance / validation harness)."""
        if not self._closed:
            self._drain_child()
        n = len(self._gate_window)
        rate = (self._gate_sum / n) if n else None
        published = self._published
        return {
            "enabled": self._enabled,
            "gate_open": self._gate_open(),
            "changed_rate": round(rate, 3) if rate is not None else None,
            "gate_samples": n,
            "buffer": self._buf_len,
            "eval_buffer": self._eval_len,
            "rounds": self._rounds,
            "steps": self._total_steps,
            "auc": (
                round(self._last_auc, 3) if self._last_auc is not None else None
            ),
            "generation": published[0] if published else 0,
            "suggested": self._suggested,
            "disabled_reason": self._disabled_reason,
        }

    # ---------- internals ---------- #

    def _gate_window_push(self, changed: int) -> None:
        if len(self._gate_window) == self._gate_window.maxlen:
            self._gate_sum -= self._gate_window[0]
        self._gate_window.append(changed)
        self._gate_sum += changed

    def _gate_open(self) -> bool:
        n = len(self._gate_window)
        if n < _GATE_MIN_SAMPLES:
            return False
        return (self._gate_sum / n) < _GATE_MAX_CHANGED_RATE

    def _disable(self, reason: str) -> None:
        if not self._enabled:
            return  # keep the FIRST reason (e.g. a child's "torch-unavailable"
            # message must not be overwritten by the pipe-EOF that follows it)
        self._enabled = False
        self._disabled_reason = reason
        self._published = None

    def _start_learner(self) -> None:
        """Spawn the learner subprocess (once, on first observe)."""
        try:
            ctx = mp.get_context("spawn")
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_learner_main,
                args=(
                    child_conn,
                    self._seed,
                    self._warmup_steps,
                    self._min_publish_auc,
                ),
                name="click-prior-learner",
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent keeps only its end
            self._proc = proc
            self._conn = parent_conn
        except Exception as e:
            self._disable(f"spawn-failed: {type(e).__name__}: {e}")

    def _send(self, msg: tuple[Any, ...]) -> None:
        if self._conn is None:
            return
        try:
            self._conn.send(msg)
        except (OSError, ValueError, BrokenPipeError):
            self._disable("learner-pipe-broken")

    def _drain_child(self) -> None:
        """Consume any pending child messages (non-blocking, O(msgs))."""
        conn = self._conn
        if conn is None:
            return
        try:
            while conn.poll(0):
                msg = conn.recv()
                kind = msg[0]
                if kind == "published":
                    self._published = (msg[1], msg[2])
                elif kind == "progress":
                    _, self._rounds, self._total_steps, auc, buf_len, ev_len = msg
                    self._last_auc = auc
                    self._buf_len = buf_len
                    self._eval_len = ev_len
                elif kind == "disabled":
                    self._disable(msg[1])
        except EOFError:
            # Child died (crash or exit) — permanent disable, sweep floor.
            self._disable("learner-exited")
        except (OSError, ValueError, BrokenPipeError):
            self._disable("learner-pipe-broken")
