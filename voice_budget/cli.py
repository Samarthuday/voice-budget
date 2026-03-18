#!/usr/bin/env python3
"""
voice_budget/cli.py

CLI entry point. Installed as `voice-budget` command via pyproject.toml.

Commands:
  voice-budget demo          Run a simulated pipeline demo showing TTFT decay + recovery
  voice-budget benchmark     Benchmark your LLM endpoint and print latency stats
  voice-budget version       Print version
"""

import argparse
import asyncio
import json
import random
import sys
import time


def cmd_version(args):
    from voice_budget import __version__
    print(f"voice-budget {__version__}")


async def _run_demo(turns: int, target_ms: int, verbose: bool):
    """
    Simulate a voice agent conversation where TTFT grows with context,
    then voice-budget kicks in and keeps it stable.
    Shows the before/after graph in ASCII.
    """
    from voice_budget import wrap

    print(f"\nvoice-budget demo — {turns} turns, target={target_ms}ms\n")

    # Simulate an LLM whose TTFT grows with message count (the real problem)
    call_count = [0]

    async def simulated_llm(messages, **kwargs):
        call_count[0] += 1
        n = len(messages)
        # TTFT grows linearly with context: ~50ms per message pair
        base_delay = 0.15 + (n * 0.04)
        jitter = random.uniform(-0.05, 0.10)
        await asyncio.sleep(max(0.05, base_delay + jitter))
        return f"Response {call_count[0]}"

    managed = wrap(
        simulated_llm,
        target_ms=target_ms,
        token_budget=300,
        window_size=8,
        use_semantic=False,
        verbose=verbose,
    )

    messages = [{"role": "system", "content": "You are a voice assistant helping with tasks."}]
    ttft_values = []

    print(f"  {'Turn':>4}  {'TTFT':>8}  {'Tokens':>7}  {'Action':<20}  Graph")
    print(f"  {'----':>4}  {'--------':>8}  {'-------':>7}  {'------':<20}  -----")

    for i in range(turns):
        user_msg = f"User turn {i+1}: " + " ".join([f"word{j}" for j in range(20)])
        messages.append({"role": "user", "content": user_msg})

        t0 = time.perf_counter()
        response = await managed(messages)
        ttft_ms = (time.perf_counter() - t0) * 1000
        ttft_values.append(ttft_ms)

        messages.append({"role": "assistant", "content": response})

        token_count = managed._measurer.count_tokens(messages)
        history = managed.compression_history()
        action = ""
        if history and history[-1].turn == managed._measurer._turn:
            ev = history[-1]
            action = f"[{ev.strategy}]"

        # ASCII bar chart
        bar_len = min(40, int(ttft_ms / 50))
        over = ttft_ms > target_ms
        bar = ("█" * bar_len) if not over else ("▓" * bar_len)
        marker = " ← OVER BUDGET" if over else ""

        print(f"  {i+1:>4}  {ttft_ms:>7.0f}ms  {token_count:>7}  {action:<20}  {bar}{marker}")

    print()
    managed.print_report()

    # ASCII trend line
    print("  TTFT over time (each ▪ = one turn):\n")
    max_ttft = max(ttft_values)
    height = 8
    width = len(ttft_values)
    grid = [[" "] * width for _ in range(height)]
    for col, val in enumerate(ttft_values):
        row = height - 1 - int((val / max_ttft) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "▪"

    target_row = height - 1 - int((target_ms / max_ttft) * (height - 1))
    target_row = max(0, min(height - 1, target_row))

    for r, row in enumerate(grid):
        prefix = f"  {max_ttft * (height - 1 - r) / (height - 1):>6.0f}ms │"
        suffix = " ← target" if r == target_row else ""
        print(prefix + "".join(row) + suffix)
    print("          └" + "─" * width)
    print(f"           Turn 1{' ' * (width - 10)}Turn {turns}\n")


async def _run_benchmark(endpoint: str, model: str, turns: int, target_ms: int):
    """Benchmark a real OpenAI-compatible endpoint."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("Error: 'openai' package required for benchmark. pip install openai")
        sys.exit(1)

    from voice_budget import wrap

    client = AsyncOpenAI(base_url=endpoint if endpoint != "openai" else None)

    async def llm_fn(messages, **kwargs):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=50,
        )
        return resp.choices[0].message.content

    managed = wrap(llm_fn, target_ms=target_ms, verbose=True)
    messages = [{"role": "system", "content": "You are a helpful assistant. Keep replies very short."}]

    print(f"\nBenchmarking {model} @ {endpoint} — {turns} turns, target={target_ms}ms\n")

    for i in range(turns):
        messages.append({"role": "user", "content": f"Say one sentence about topic {i}."})
        response = await managed(messages)
        messages.append({"role": "assistant", "content": response})

    managed.print_report()

    s = managed.stats()
    result = {
        "model": model,
        "endpoint": endpoint,
        "turns": turns,
        "target_ms": target_ms,
        "p50_ms": round(s.p50_ms, 1) if s else None,
        "p95_ms": round(s.p95_ms, 1) if s else None,
        "budget_met": s.budget_violated is False if s else None,
        "total_tokens_saved": s.total_tokens_saved if s else 0,
    }
    print("\nJSON result:")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="voice-budget",
        description="voice-budget — TTFT feedback loop for voice agent context management",
    )
    sub = parser.add_subparsers(dest="cmd")

    # demo
    dp = sub.add_parser("demo", help="Run a simulated demo showing TTFT decay + recovery")
    dp.add_argument("--turns", type=int, default=20, help="Number of conversation turns (default: 20)")
    dp.add_argument("--target-ms", type=int, default=800, help="TTFT target in ms (default: 800)")
    dp.add_argument("--verbose", action="store_true", help="Print compression decisions")

    # benchmark
    bp = sub.add_parser("benchmark", help="Benchmark a real LLM endpoint")
    bp.add_argument("--endpoint", default="openai", help="OpenAI-compatible base URL or 'openai'")
    bp.add_argument("--model", default="gpt-4o-mini", help="Model name")
    bp.add_argument("--turns", type=int, default=15, help="Number of turns")
    bp.add_argument("--target-ms", type=int, default=800, help="TTFT target in ms")

    # version
    sub.add_parser("version", help="Print version")

    args = parser.parse_args()

    if args.cmd == "demo":
        asyncio.run(_run_demo(args.turns, args.target_ms, args.verbose))
    elif args.cmd == "benchmark":
        asyncio.run(_run_benchmark(args.endpoint, args.model, args.turns, args.target_ms))
    elif args.cmd == "version":
        cmd_version(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()