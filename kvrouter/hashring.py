"""Consistent hash ring with virtual nodes."""
from __future__ import annotations

import bisect
import hashlib


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


class HashRing:
    def __init__(self, nodes: list[str] | None = None, vnodes: int = 128):
        self.vnodes = vnodes
        self._keys: list[int] = []
        self._owner: dict[int, str] = {}
        self.nodes: set[str] = set()
        for n in nodes or []:
            self.add(n)

    def add(self, node: str) -> None:
        if node in self.nodes:
            return
        self.nodes.add(node)
        for i in range(self.vnodes):
            k = h64(f"{node}#{i}")
            self._owner[k] = node
            bisect.insort(self._keys, k)

    def remove(self, node: str) -> None:
        if node not in self.nodes:
            return
        self.nodes.discard(node)
        for i in range(self.vnodes):
            k = h64(f"{node}#{i}")
            del self._owner[k]
            self._keys.remove(k)

    def preference(self, key: str, n: int | None = None) -> list[str]:
        """Distinct nodes clockwise from hash(key): primary first, then fallbacks."""
        if not self._keys:
            return []
        n = min(n or len(self.nodes), len(self.nodes))
        i = bisect.bisect(self._keys, h64(key)) % len(self._keys)
        out: list[str] = []
        while len(out) < n:
            node = self._owner[self._keys[i]]
            if node not in out:
                out.append(node)
            i = (i + 1) % len(self._keys)
        return out

    def get(self, key: str) -> str | None:
        p = self.preference(key, 1)
        return p[0] if p else None
