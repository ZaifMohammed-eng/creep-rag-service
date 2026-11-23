"""Load test: start uvicorn on the creep corpus and replay its questions over HTTP.

    python eval/bench_api.py --requests 1000 --concurrency 16

The load generator runs on the same machine as the server, so treat the numbers as
upper bounds. The answer cache is bypassed by sending each question once per phase
and suffixing a counter in the concurrent phase.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def wait_ready(url: str, timeout: float = 120) -> None:
    t = time.time()
    while time.time() - t < timeout:
        try:
            if httpx.get(f"{url}/healthz", timeout=60).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


async def burst(url: str, qs: list[str], conc: int) -> tuple[list[float], int, float]:
    lat, errors, sem = [], 0, asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=60) as c:
        async def one(q):
            nonlocal errors
            async with sem:
                t = time.perf_counter()
                r = await c.post(f"{url}/v1/query", json={"query": q})
                lat.append((time.perf_counter() - t) * 1000)
                errors += r.status_code != 200
        t = time.perf_counter()
        await asyncio.gather(*(one(q) for q in qs))
        return lat, errors, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "RAG_LLM": "extractive", "RAG_CORPUS": "creep"}
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "rag.api:app", "--port", str(args.port),
                            "--workers", str(args.workers), "--log-level", "warning"], cwd=ROOT, env=env)
    try:
        wait_ready(url)
        from rag.creep import build
        pool = [q.text for q in build().questions]
        random.Random(0).shuffle(pool)
        qs = pool[:args.requests]

        seq = []
        with httpx.Client(timeout=60) as c:
            for q in qs:
                t = time.perf_counter()
                r = c.post(f"{url}/v1/query", json={"query": q})
                r.raise_for_status()
                seq.append((time.perf_counter() - t) * 1000)
        conc_qs = [f"{q} ({i})" for i, q in enumerate(qs)]
        lat, errors, wall = asyncio.run(burst(url, conc_qs, args.concurrency))
        out = {
            "requests": len(qs), "workers": args.workers, "provider": "extractive (no LLM call)",
            "sequential_ms": {"p50": round(float(np.percentile(seq, 50)), 2),
                              "p95": round(float(np.percentile(seq, 95)), 2)},
            f"concurrent_{args.concurrency}_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                                                  "p95": round(float(np.percentile(lat, 95)), 2)},
            "throughput_rps": round(len(conc_qs) / wall, 1), "errors": errors,
            "cpus": os.cpu_count(),
        }
        (ROOT / "results" / f"latency_{args.workers}worker.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    finally:
        srv.terminate()
        srv.wait()


if __name__ == "__main__":
    main()
