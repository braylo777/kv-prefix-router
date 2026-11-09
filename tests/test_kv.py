from collections import Counter

import pytest

from kvrouter.hashring import HashRing
from kvrouter.prefix import BLOCK, block_hashes, routing_key
from kvrouter.routers import LeastLoadedRouter, PrefixAffinityRouter, RoundRobinRouter
from kvrouter.sim import Replica, Request, make_workload, simulate, summarize

NODES = [f"r{i}" for i in range(8)]


def test_ring_is_deterministic_and_balanced():
    ring = HashRing(NODES, vnodes=128)
    c = Counter(ring.get(f"k{i}") for i in range(20_000))
    assert set(c) == set(NODES)
    assert max(c.values()) / min(c.values()) < 1.6
    assert ring.get("abc") == HashRing(NODES).get("abc")


def test_removing_a_node_only_moves_its_keys():
    ring = HashRing(NODES)
    before = {f"k{i}": ring.get(f"k{i}") for i in range(5000)}
    ring.remove("r3")
    moved = [k for k, n in before.items() if ring.get(k) != n]
    assert all(before[k] == "r3" for k in moved)
    assert 0.05 < len(moved) / 5000 < 0.25  # ~1/8 of keys


def test_preference_list_distinct_and_primary_first():
    ring = HashRing(NODES)
    p = ring.preference("tenant-1")
    assert len(p) == 8 and len(set(p)) == 8 and p[0] == ring.get("tenant-1")


def test_add_node_back_restores_mapping():
    ring = HashRing(NODES)
    before = ring.get("x")
    ring.remove("r1"); ring.add("r1")
    assert ring.get("x") == before


def test_block_hashes_chain_and_partial_block_ignored():
    a = list(range(40))
    hs = block_hashes(a)
    assert len(hs) == 40 // BLOCK
    b = list(range(32)) + [999] * 8
    assert block_hashes(b)[:2] == hs[:2]
    c = [7] + list(range(1, 40))
    assert block_hashes(c)[1] != hs[1]  # change in block 0 changes every later block


def test_routing_key_depends_only_on_system_prompt():
    sys_a = list(range(100))
    assert routing_key(sys_a) == routing_key(list(range(100)))
    assert routing_key(sys_a) != routing_key(list(range(1, 101)))
    assert routing_key([1, 2]).startswith("empty:")


def req(rid, tenant, t=0.0, sys_len=320, user_len=32):
    return Request(rid, t, tenant, [tenant * 10_000 + i for i in range(sys_len)],
                   [rid * 100 + i + 5_000_000 for i in range(user_len)], 10)


def test_replica_prefix_cache_hit_on_second_request():
    r = Replica("r0", cache_blocks=1000)
    a = r.serve(req(1, 0))
    b = r.serve(req(2, 0, t=10_000))
    assert a["cached_tokens"] == 0 and b["cached_tokens"] == 320
    assert b["ttft_ms"] < a["ttft_ms"]


def test_replica_lru_eviction():
    r = Replica("r0", cache_blocks=20)  # 320 tokens = 20 blocks
    r.serve(req(1, 0, t=0)); r.serve(req(2, 1, t=10_000))
    assert r.serve(req(3, 0, t=20_000))["cached_tokens"] == 0


def test_replica_queue_wait_counts_in_ttft():
    r = Replica("r0")
    r.serve(req(1, 0, t=0))
    second = r.serve(req(2, 1, t=0))
    assert second["wait_ms"] > 0 and second["ttft_ms"] > second["wait_ms"]


def test_affinity_routes_same_tenant_to_same_replica():
    reps = {n: Replica(n) for n in NODES}
    rt = PrefixAffinityRouter(NODES)
    targets = {rt.route(req(i, 5, t=i * 1e6), reps, i * 1e6) for i in range(20)}
    assert len(targets) == 1


def test_affinity_falls_back_when_primary_down():
    reps = {n: Replica(n) for n in NODES}
    rt = PrefixAffinityRouter(NODES)
    primary = rt.route(req(1, 5), reps, 0)
    reps[primary].up = False
    rt.mark_down(primary)
    assert rt.route(req(2, 5), reps, 0) != primary


def test_bounded_load_spills_when_primary_queue_exceeds_slack():
    reps = {n: Replica(n) for n in NODES}
    rt = PrefixAffinityRouter(NODES, max_queue=10**6, slack=0)
    primary = rt.route(req(1, 5), reps, 0)
    reps[primary].serve(req(1, 5))  # primary now has 1 in flight, others 0
    assert rt.route(req(2, 5), reps, 0) != primary and rt.fallbacks == 1


def test_round_robin_and_least_loaded_skip_down_replicas():
    reps = {n: Replica(n) for n in NODES}
    reps["r0"].up = False
    rr = RoundRobinRouter(NODES)
    assert "r0" not in {rr.route(None, reps, 0) for _ in range(16)}
    assert LeastLoadedRouter(NODES).route(None, reps, 0) != "r0"


def test_workload_is_seeded():
    a, b = make_workload(n=50, seed=3), make_workload(n=50, seed=3)
    assert [(r.tenant, r.t) for r in a] == [(r.tenant, r.t) for r in b]


@pytest.fixture(scope="module")
def small_runs():
    reqs = make_workload(n=1500, tenants=20, rps=2.0, seed=1)
    kw = dict(cache_blocks=800, prefill_ms_per_token=0.25, decode_ms_per_token=4.0)
    out = {}
    for rt in [LeastLoadedRouter(NODES), PrefixAffinityRouter(NODES, max_queue=10**6, slack=0)]:
        out[rt.name] = summarize(*simulate(rt, reqs, 8, **kw))
    return out


def test_bounded_affinity_raises_hit_rate_and_cuts_mean_ttft(small_runs):
    ll, pa = small_runs["least_loaded"], small_runs["prefix_affinity_bounded_slack0"]
    assert pa["prefix_cache_hit_rate_tokens"] > ll["prefix_cache_hit_rate_tokens"]
    assert pa["ttft_ms_mean"] < ll["ttft_ms_mean"]


def test_failure_scenario_serves_every_request():
    reqs = make_workload(n=600, tenants=10, rps=2.0, seed=2)
    rt = PrefixAffinityRouter(NODES, slack=0)
    rows, reps = simulate(rt, reqs, 8, fail_at=reqs[200].t, fail_node="r3")
    assert len(rows) == 600 and not reps["r3"].up
    assert all(r["replica"] != "r3" for r in rows[201:])
