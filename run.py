#!/usr/bin/env python3
"""
run.py — single command to run voice-budget demo.

Usage:
    python run.py                    # default: 20 turns, 800ms target
    python run.py --turns 30         # more turns
    python run.py --target-ms 600    # tighter budget
    python run.py --verbose          # show every compression decision
"""

import argparse
import asyncio
import random
import sys
import time


def _mock_tiktoken():
    """Install a tiktoken mock so the demo runs without network/install."""
    import types
    mock = types.ModuleType("tiktoken")
    class _Enc:
        def encode(self, text):
            return str(text).split()
    mock.encoding_for_model = lambda m: _Enc()
    mock.get_encoding       = lambda n: _Enc()
    sys.modules.setdefault("tiktoken", mock)


async def run_demo(turns: int, target_ms: int, verbose: bool):
    _mock_tiktoken()
    from voice_budget import wrap

    print(f"\n{'='*60}")
    print("  voice-budget live demo")
    print(f"  turns={turns}  target={target_ms}ms  strategy=sliding_window")
    print(f"{'='*60}\n")
    print("  Simulating a voice agent whose TTFT grows with context,")
    print(f"  then voice-budget kicks in and keeps it under {target_ms}ms.\n")

    call_n = [0]

    async def simulated_llm(messages, **kwargs):
        """TTFT grows ~15ms per message — exactly the real-world problem."""
        call_n[0] += 1
        n = len(messages)
        delay = 0.05 + (n * 0.015) + random.uniform(0, 0.02)
        await asyncio.sleep(delay)
        return f"Response {call_n[0]}"

    llm = wrap(
        simulated_llm,
        target_ms=target_ms,
        token_budget=200,
        window_size=6,
        use_semantic=False,
        verbose=verbose,
    )

    messages = [{"role": "system", "content": "You are a helpful voice assistant."}]
    ttft_values = []

    print(f"  {'Turn':>4}  {'TTFT':>8}  {'Tokens':>7}  {'Action':<26}  Latency bar")
    print(f"  {'----':>4}  {'--------':>8}  {'-------':>7}  {'------':<26}  -----------")

    for i in range(turns):
        messages.append({
            "role": "user",
            "content": f"Turn {i+1}: " + "word " * 20
        })

        t0 = time.perf_counter()
        response = await llm(messages)
        ttft_ms = (time.perf_counter() - t0) * 1000
        ttft_values.append(ttft_ms)

        messages.append({"role": "assistant", "content": response})

        token_count = llm._measurer.count_tokens(messages)
        history = llm.compression_history()
        action = ""
        if history and history[-1].turn == llm._measurer._turn - 1:
            ev = history[-1]
            action = f"[compressed: {ev.tokens_before}→{ev.tokens_after}t]"

        over = ttft_ms > target_ms
        bar_len = min(35, int(ttft_ms / 30))
        bar = ("▓" if over else "█") * bar_len
        marker = " ◄ OVER BUDGET" if over else ""

        print(f"  {i+1:>4}  {ttft_ms:>7.0f}ms  {token_count:>7}  {action:<26}  {bar}{marker}")

    # ── Final report ──────────────────────────────────────────────────────────
    llm.print_report()

    # ── ASCII trend chart ─────────────────────────────────────────────────────
    print("  TTFT over conversation (each char = 1 turn):\n")
    if ttft_values:
        max_v = max(ttft_values)
        height = 8
        grid = [[" "] * len(ttft_values) for _ in range(height)]
        for col, val in enumerate(ttft_values):
            row = height - 1 - int((val / max_v) * (height - 1))
            row = max(0, min(height - 1, row))
            grid[row][col] = "●"
        target_row = height - 1 - int((target_ms / max_v) * (height - 1))
        target_row = max(0, min(height - 1, target_row))
        for r, row in enumerate(grid):
            label = f"  {int(max_v * (height-1-r) / (height-1)):>6}ms │"
            suffix = f"  ← {target_ms}ms target" if r == target_row else ""
            print(label + "".join(row) + suffix)
        print(f"          └{'─' * len(ttft_values)}")
        print(f"           1{' ' * (len(ttft_values)-4)}{turns}\n")
        s = llm.stats()
        if s:
            status = "✓ BUDGET MET" if not s.budget_violated else "✗ BUDGET EXCEEDED"
            print(f"  Final: P50={s.p50_ms:.0f}ms  P95={s.p95_ms:.0f}ms  "
                  f"Target={target_ms}ms  {status}\n")


def main():
    parser = argparse.ArgumentParser(
        description="voice-budget demo — watch TTFT get controlled in real time"
    )
    parser.add_argument("--turns",     type=int,  default=20,  help="Conversation turns (default: 20)")
    parser.add_argument("--target-ms", type=int,  default=800, help="TTFT target ms (default: 800)")
    parser.add_argument("--verbose",   action="store_true",    help="Show every decision")
    args = parser.parse_args()
    asyncio.run(run_demo(args.turns, args.target_ms, args.verbose))


if __name__ == "__main__":
    main()