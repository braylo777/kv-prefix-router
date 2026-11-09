"""Discrete-event simulator of LLM replicas with a paged prefix (KV) cache. SIMULATED.

Each replica serves one request at a time (FIFO). Service = prefill of the tokens NOT
already cached + decode of output tokens. TTFT = queue wait + prefill. The KV cache is an
LRU of prompt blocks with a fixed block capacity. Timing constants are assumptions, not
measurements of any GPU or serving engine.
"""
from __future__ import annotations

import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from .prefix import BLOCK, block_hashes


@dataclass
class Request:
    rid: int
    t: float                      # arrival, ms
    tenant: int
    system_tokens: list[int]
    user_tokens: list[int]
    out_tokens: int


@dataclass
class Replica:
    name: str
    cache_blocks: int = 4000
    prefill_ms_per_token: float = 0.25
    decode_ms_per_token: float = 12.0
    base_ms: float = 15.0
    up: bool = True
    free_at: float = 0.0
    _finishes: deque = field(default_factory=deque)
    _cache: OrderedDict = field(default_factory=OrderedDict)
    served: int = 0

    def queue_len(self, now: float) -> int:
        while self._finishes and self._finishes[0] <= now:
            self._finishes.popleft()
        return len(self._finishes)

    def cached_prefix_blocks(self, blocks: list[str]) -> int:
        n = 0
        for b in blocks:
            if b in self._cache:
                n += 1
            else:
                break
        return n

    def serve(self, req: Request) -> dict:
        tokens = req.system_tokens + req.user_tokens
        blocks = block_hashes(tokens)
        hit_blocks = self.cached_prefix_blocks(blocks)
        cached_tokens = hit_blocks * BLOCK
        uncached = len(tokens) - cached_tokens
        prefill = self.base_ms + uncached * self.prefill_ms_per_token
        start = max(req.t, self.free_at)
        ttft = (start - req.t) + prefill
        finish = start + prefill + req.out_tokens * self.decode_ms_per_token
        self.free_at = finish
        self._finishes.append(finish)
        for b in blocks:  # insert / refresh as most-recent, evict LRU
            self._cache[b] = True
            self._cache.move_to_end(b)
        while len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        self.served += 1
        return {"ttft_ms": ttft, "cached_tokens": cached_tokens, "prompt_tokens": len(tokens),
                "wait_ms": start - req.t, "replica": self.name}

    def wipe(self) -> None:
        self._cache.clear()


def make_workload(n=6000, tenants=40, zipf_s=1.1, rps=30.0, seed=7):
    """Tenants share a long system prompt; user turns are short and unique."""
    rng = random.Random(seed)
    sys_prompts = []
    for t in range(tenants):
        L = rng.randint(800, 3000)
        sys_prompts.append([t * 1_000_003 + i for i in range(L)])
    weights = [1 / (k + 1) ** zipf_s for k in range(tenants)]
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(rps / 1000)
        ten = rng.choices(range(tenants), weights)[0]
        user = [rng.randint(5_000_000_000, 6_000_000_000) for _ in range(rng.randint(40, 300))]
        reqs.append(Request(i, t, ten, sys_prompts[ten], user, rng.randint(50, 250)))
    return reqs


def simulate(router, reqs, n_replicas=8, fail_at=None, fail_node="r3", recover_at=None, **replica_kw):
    reps = {f"r{i}": Replica(f"r{i}", **replica_kw) for i in range(n_replicas)}
    out = []
    for req in reqs:
        if fail_at is not None and req.t >= fail_at and reps[fail_node].up and (recover_at is None or req.t < recover_at):
            reps[fail_node].up = False
            reps[fail_node].wipe()
            router.mark_down(fail_node)
        if recover_at is not None and req.t >= recover_at and not reps[fail_node].up:
            reps[fail_node].up = True
            reps[fail_node].free_at = req.t
            router.mark_up(fail_node)
        node = router.route(req, reps, req.t)
        r = reps[node].serve(req)
        r["tenant"] = req.tenant
        out.append(r)
    return out, reps


def summarize(rows, reps) -> dict:
    tt = sorted(r["ttft_ms"] for r in rows)
    pct = lambda p: round(tt[min(len(tt) - 1, int(p / 100 * (len(tt) - 1)))], 1)
    served = [r.served for r in reps.values()]
    live = [s for s in served if s]
    return {
        "requests": len(rows),
        "prefix_cache_hit_rate_tokens": round(sum(r["cached_tokens"] for r in rows) / sum(r["prompt_tokens"] for r in rows), 4),
        "ttft_ms_p50": pct(50), "ttft_ms_p95": pct(95), "ttft_ms_p99": pct(99),
        "ttft_ms_mean": round(sum(tt) / len(tt), 1),
        "load_imbalance_max_over_mean": round(max(live) / (sum(live) / len(live)), 3),
        "served_per_replica": {k: v.served for k, v in reps.items()},
    }
