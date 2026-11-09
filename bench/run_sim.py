"""Compare routing policies on the simulated cluster; write results/simulation.json. SIMULATED."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kvrouter.routers import LeastLoadedRouter, PrefixAffinityRouter, RoundRobinRouter  # noqa: E402
from kvrouter.sim import make_workload, simulate, summarize  # noqa: E402

NODES = [f"r{i}" for i in range(8)]
WORKLOAD = dict(n=6000, tenants=40, zipf_s=1.1, rps=3.0, seed=7)
REPLICA = dict(cache_blocks=1500, prefill_ms_per_token=0.25, decode_ms_per_token=4.0, base_ms=15.0)


def routers():
    return [RoundRobinRouter(NODES), LeastLoadedRouter(NODES),
            PrefixAffinityRouter(NODES, max_queue=1_000_000),          # pure affinity
            PrefixAffinityRouter(NODES, max_queue=1_000_000, slack=0)]  # bounded load


def main():
    reqs = make_workload(**WORKLOAD)
    span = reqs[-1].t
    out = {"kind": "SIMULATED (discrete-event model; timing constants are assumptions, not GPU measurements)",
           "date": time.strftime("%Y-%m-%d"), "replicas": len(NODES), "workload": WORKLOAD, "replica_model": REPLICA,
           "scenarios": {}}
    for scen, kw in {"steady": {}, "replica_r3_fails_at_40pct": {"fail_at": 0.4 * span}}.items():
        out["scenarios"][scen] = {}
        for rt in routers():
            rows, reps = simulate(rt, reqs, n_replicas=len(NODES), **kw, **REPLICA)
            s = summarize(rows, reps)
            if isinstance(rt, PrefixAffinityRouter):
                s["fallback_routes"] = rt.fallbacks
            out["scenarios"][scen][rt.name] = s
            print(scen, rt.name, {k: v for k, v in s.items() if k != "served_per_replica"})
    # cache-size sweep, steady state: where does affinity help?
    out["cache_sweep_steady"] = {}
    for blocks in (750, 1500, 4000):
        kw = {**REPLICA, "cache_blocks": blocks}
        row = {}
        for rt in routers()[1:]:
            rows, reps = simulate(rt, reqs, n_replicas=len(NODES), **kw)
            s = summarize(rows, reps)
            row[rt.name] = {k: s[k] for k in ("prefix_cache_hit_rate_tokens", "ttft_ms_p50", "ttft_ms_p95", "load_imbalance_max_over_mean")}
        out["cache_sweep_steady"][f"{blocks}_blocks_{blocks * 16}_tokens"] = row
        print("sweep", blocks, row)
    # load sweep, steady state, default cache
    out["load_sweep_steady"] = {}
    for rps in (2.0, 4.0, 6.0):
        rq = make_workload(**{**WORKLOAD, "rps": rps})
        row = {}
        for rt in routers()[1:]:
            rows, reps = simulate(rt, rq, n_replicas=len(NODES), **REPLICA)
            s = summarize(rows, reps)
            row[rt.name] = {k: s[k] for k in ("prefix_cache_hit_rate_tokens", "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_mean", "load_imbalance_max_over_mean")}
        out["load_sweep_steady"][f"{rps}_rps"] = row
        print("load", rps, row)
    st = out["scenarios"]["steady"]
    b = st["prefix_affinity_bounded_slack0"]
    out["headline"] = {
        "bounded_vs_least_loaded_ttft_p50_reduction": round(1 - b["ttft_ms_p50"] / st["least_loaded"]["ttft_ms_p50"], 4),
        "bounded_vs_least_loaded_ttft_p95_reduction": round(1 - b["ttft_ms_p95"] / st["least_loaded"]["ttft_ms_p95"], 4),
        "bounded_vs_least_loaded_ttft_mean_reduction": round(1 - b["ttft_ms_mean"] / st["least_loaded"]["ttft_ms_mean"], 4),
        "bounded_vs_round_robin_ttft_p95_reduction": round(1 - b["ttft_ms_p95"] / st["round_robin"]["ttft_ms_p95"], 4),
    }
    print(out["headline"])
    (ROOT / "results" / "simulation.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
