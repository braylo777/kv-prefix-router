"""Prefix keys. Tokens are grouped into fixed-size blocks (like paged KV caches); each block
hash chains the previous one, so a block id identifies the whole prefix up to it."""
from __future__ import annotations

import hashlib

BLOCK = 16


def block_hashes(tokens: list[int], block: int = BLOCK) -> list[str]:
    out, prev = [], ""
    for i in range(0, len(tokens) - len(tokens) % block, block):
        prev = hashlib.blake2b((prev + ",".join(map(str, tokens[i:i + block]))).encode(), digest_size=8).hexdigest()
        out.append(prev)
    return out


def routing_key(system_tokens: list[int], block: int = BLOCK) -> str:
    """Key on the system-prompt prefix only, so every user turn of a tenant maps together."""
    hs = block_hashes(system_tokens, block)
    return hs[-1] if hs else "empty:" + ",".join(map(str, system_tokens))
