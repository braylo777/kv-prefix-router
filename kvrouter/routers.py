"""Routing policies. All take (request, replicas, now) and return a replica name."""
from __future__ import annotations

import itertools
import random

from .hashring import HashRing
from .prefix import routing_key


class PrefixAffinityRouter:
    """Consistent-hash the system-prompt prefix to a replica. If that replica is down or
    overloaded, walk the ring's preference list; if every replica is overloaded, take the
    least loaded.

    Overloaded = queue >= max_queue, or (bounded-load mode, `slack` set) queue > the
    cluster's shortest queue + slack.
    """

    def __init__(self, nodes: list[str], vnodes: int = 128, max_queue: int = 8, slack: int | None = None):
        self.ring = HashRing(nodes, vnodes)
        self.max_queue = max_queue
        self.slack = slack
        self.fallbacks = 0
        self.name = "prefix_affinity_naive" if slack is None else f"prefix_affinity_bounded_slack{slack}"

    def mark_down(self, node: str) -> None:
        self.ring.remove(node)

    def mark_up(self, node: str) -> None:
        self.ring.add(node)

    def route(self, req, replicas, now) -> str:
        pref = self.ring.preference(routing_key(req.system_tokens))
        limit = self.max_queue
        if self.slack is not None:
            limit = min(limit, min(r.queue_len(now) for r in replicas.values() if r.up) + self.slack + 1)
        for i, node in enumerate(pref):
            if replicas[node].up and replicas[node].queue_len(now) < limit:
                if i:
                    self.fallbacks += 1
                return node
        self.fallbacks += 1
        live = [r for r in replicas.values() if r.up]
        return min(live, key=lambda r: r.queue_len(now)).name


class RoundRobinRouter:
    name = "round_robin"

    def __init__(self, nodes):
        self._it = itertools.cycle(nodes)

    def mark_down(self, node): pass
    def mark_up(self, node): pass

    def route(self, req, replicas, now):
        for _ in range(len(replicas)):
            n = next(self._it)
            if replicas[n].up:
                return n
        raise RuntimeError("no replicas up")


class LeastLoadedRouter:
    name = "least_loaded"

    def __init__(self, nodes, seed: int = 0):
        self.rng = random.Random(seed)

    def mark_down(self, node): pass
    def mark_up(self, node): pass

    def route(self, req, replicas, now):
        live = [r for r in replicas.values() if r.up]
        best = min(r.queue_len(now) for r in live)
        return self.rng.choice([r.name for r in live if r.queue_len(now) == best])
