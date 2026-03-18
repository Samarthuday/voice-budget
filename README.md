# voice-budget

**The only voice agent context manager with a TTFT feedback loop.**

Every other context compression library does this:

```python
compress(messages, ratio=0.7)  # blindly compress. hope it helped.
```

`voice-budget` does this:

```python
llm = voice_budget.wrap(your_llm, target_ms=800)
# → measures TTFT per turn
# → detects when context growth is making it climb  
# → compresses automatically
# → measures whether it actually helped
# → rolls back if it didn't
```

---

## The problem

I was building a voice agent using Whisper + OpenAI + ElevenLabs. Latency started at ~1.2s. After 10 minutes of conversation it was at 4–5 seconds. I had no idea why.

The HuggingFace Voice Agent Latency Playbook (March 2026) confirmed it:

> *"Two traces hit 2.4s and 6.8s, both deep into multi-turn conversations where the full history was being sent."*
> *"Limiting context keeps TTFT flat"* — described as a manual fix. No library.

Every existing context compression tool — `context-compressor`, `reme-ai`, Pipecat's compaction, LangChain's `ConversationSummaryMemory` — compresses based on a fixed token ratio. None of them measure whether the compression actually reduced TTFT. None of them auto-tune. None of them roll back if compression made things worse.

**`voice-budget` closes that loop.**

---

## Install

```bash
pip install voice-budget

# With semantic compression (recommended):
pip install "voice-budget[semantic]"
```

**Dependencies:** `numpy`, `tiktoken` only. No GPU. No cloud API.

---

## Quick start

### Framework-agnostic

```python
import asyncio
from voice_budget import wrap

# Your existing async LLM function
async def my_llm(messages, **kwargs):
    response = await openai_client.chat.completions.create(
        model="gpt-4o", messages=messages, **kwargs
    )
    return response.choices[0].message.content

# Wrap it — one line
managed_llm = wrap(my_llm, target_ms=800, verbose=True)

# Use exactly like before
async def voice_loop():
    messages = [{"role": "system", "content": "You are a voice assistant."}]
    while True:
        user_input = await get_user_speech()
        messages.append({"role": "user", "content": user_input})

        response = await managed_llm(messages)   # ← voice-budget works here

        messages.append({"role": "assistant", "content": response})

        # Print stats every 10 turns
        if len(messages) % 20 == 0:
            managed_llm.print_report()
```

### Pipecat

```python
from pipecat.pipeline.pipeline import Pipeline
from voice_budget.pipecat_integration import VoiceBudgetProcessor

budget = VoiceBudgetProcessor(target_ms=800, verbose=True)

pipeline = Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),
    budget,          # ← insert between context aggregator and LLM
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

---

## How it works

```text

Turn 1:   TTFT=480ms  tokens=120  ✓ under budget
Turn 2:   TTFT=510ms  tokens=240  ✓ under budget
...
Turn 8:   TTFT=920ms  tokens=980  ↑ P95 > 800ms
          → trigger: sliding_window compression
          → tokens: 980 → 420
Turn 9:   TTFT=490ms  tokens=420  ✓ compression helped (delta=430ms)
          → keep ratio
...
Turn 14:  TTFT=850ms  tokens=720  ↑ P95 > 800ms again
          → trigger: semantic_trim compression (more precise)
          → tokens: 720 → 350
Turn 15:  TTFT=460ms  tokens=350  ✓ compression helped
```

### The feedback loop

```python
TTFT_P95 = np.percentile(ttft_window, 95)

if TTFT_P95 > target_ms:
    tokens_before = count_tokens(messages)
    messages = compress(messages)          # sliding window first
    tokens_after = count_tokens(messages)

    ttft_after = measure_next_turn()

    delta = ttft_before - ttft_after
    if delta < 0:                          # compression made it WORSE
        rollback()                         # restore full context
    else:
        ratio = ratio * (target_ms / TTFT_P95)  # auto-tune toward target
```

### Compression strategies (escalating cost)

| Strategy | Cost | When used |
| --- | --- | --- |
| `sliding_window` | Free | First attempt — drop oldest turns |
| `semantic_trim` | ~5ms (local embeddings) | If sliding window not enough |
| `summarise_tail` | 1 LLM call | If semantic trim not enough (opt-in) |

---

## Stats and reporting

```python
# Current rolling statistics
s = managed_llm.stats()
print(s.p50_ms, s.p95_ms, s.jitter_ms)

# Full report
managed_llm.print_report()
```

```text
============================================================
voice-budget Report
============================================================
  Total turns:          47
  Current P50 TTFT:     510ms
  Current P95 TTFT:     780ms
  Target:               800ms
  Budget met:           ✓
  Jitter (std):         94ms
  Compressions:         3
  Helpful:              3
  Harmful (rolled back):0
  Avg improvement:      412ms
  Total tokens saved:   1,840
  Strategies used:      sliding_window, semantic_trim
============================================================
```

---

## Configuration

```python
from voice_budget import VoiceBudget

budget = VoiceBudget(
    llm_fn=your_llm,
    target_ms=800,           # TTFT budget in ms (P95)
    model="gpt-4o",          # for tiktoken token counting
    window_size=20,          # rolling window for statistics
    token_budget=2000,       # target token count after compression
    use_semantic=True,       # semantic trim (needs sentence-transformers)
    use_summarise=False,     # LLM-based summarisation (costs 1 LLM call)
    verbose=True,            # print compression decisions
    on_compression=callback, # called after each compression event
    on_budget_violation=cb,  # called when P95 > target_ms
)
```

---

## Callbacks

```python
def on_compression(event):
    print(f"Compressed using {event.strategy}")
    print(f"Tokens: {event.tokens_before} → {event.tokens_after}")
    print(f"TTFT before: {event.ttft_before_ms:.0f}ms")
    # event.ttft_after_ms is filled after the next turn

def on_budget_violation(stats):
    print(f"Budget violated! P95={stats.p95_ms:.0f}ms > {stats.target_ms}ms")
    alert_your_monitoring_system(stats)

budget = VoiceBudget(
    llm_fn,
    on_compression=on_compression,
    on_budget_violation=on_budget_violation,
)
```

---

## What voice-budget does NOT do

- It does not modify your LLM, STT, or TTS providers
- It does not add network calls or cloud dependencies
- It does not change your pipeline architecture
- It does not make promises about accuracy (use `accuracy_fn` to verify)

---

## Why not use existing tools?

| Tool | TTFT-aware? | Feedback loop? | Voice-specific? | Auto-tune? |
| --- | --- | --- | --- | --- |
| context-compressor | ✗ | ✗ | ✗ | ✗ |
| reme-ai | ✗ | ✗ | ✗ | ✗ |
| Pipecat compaction | ✗ | ✗ | ✓ | ✗ |
| LangChain SummaryMemory | ✗ | ✗ | ✗ | ✗ |
| **voice-budget** | **✓** | **✓** | **✓** | **✓** |

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
