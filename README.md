# voice-budget

**TTFT feedback loop for voice agent context management.**

Other libraries compress blindly. `voice-budget` measures TTFT before and after, auto-tunes, and rolls back if compression hurts.

```python
import asyncio
from voice_budget import wrap

async def main():
    managed = wrap(your_llm, target_ms=800)
    response = await managed(messages)  # measures, compresses, verifies

asyncio.run(main())
```

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

async def my_llm(messages, **kwargs):
    resp = await openai_client.chat.completions.create(
        model="gpt-4o", messages=messages, **kwargs
    )
    return resp.choices[0].message.content

async def voice_loop():
    managed = wrap(my_llm, target_ms=800, verbose=True)
    messages = [{"role": "system", "content": "You are a voice assistant."}]
    while True:
        messages.append({"role": "user", "content": await get_user_speech()})
        response = await managed(messages)
        messages.append({"role": "assistant", "content": response})

asyncio.run(voice_loop())
```

### Pipecat

> **Note for Pipecat Users**: The provided `VoiceBudgetProcessor` in `pipecat_integration.py` is a blueprint. In order to properly integrate it with a full Pipecat pipeline, you will need to ensure it correctly inherits from `pipecat.processors.frame_processor.FrameProcessor` and wires up the `push_frame` and `process_frame` methods to pass frames down the pipeline.

```python
from pipecat.pipeline.pipeline import Pipeline
from voice_budget.pipecat_integration import VoiceBudgetProcessor

budget = VoiceBudgetProcessor(target_ms=800, verbose=True)

pipeline = Pipeline([
    transport.input(), stt, context_aggregator.user(),
    budget,          # ← insert before LLM
    llm, tts, transport.output(), context_aggregator.assistant(),
])
```

---

## How it works

```text
Turn 1:   TTFT=480ms  tokens=120  ✓ under budget
Turn 8:   TTFT=920ms  tokens=980  ↑ P95 > 800ms → sliding_window → 980→420 tokens
Turn 9:   TTFT=490ms  tokens=420  ✓ compression helped (delta=430ms)
Turn 14:  TTFT=850ms  tokens=720  ↑ P95 > 800ms → semantic_trim → 720→350 tokens
Turn 15:  TTFT=460ms  tokens=350  ✓ compression helped
```

### Compression strategies (escalating cost)

| Strategy | Cost | When used |
| --- | --- | --- |
| `sliding_window` | Free | First attempt — drop oldest turns |
| `semantic_trim` | ~5ms (local embeddings) | If sliding window not enough |
| `summarise_tail` | 1 LLM call | If semantic trim not enough (opt-in) |

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

## Stats and reporting

```python
s = managed.stats()
print(s.p50_ms, s.p95_ms, s.jitter_ms)

managed.print_report()
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
  Compressions:         3
  Helpful:              3
  Harmful (rolled back):0
  Total tokens saved:   1,840
  Strategies used:      sliding_window, semantic_trim
============================================================
```

---

## Why not use existing tools?

| Tool | TTFT-aware? | Feedback loop? | Auto-tune? |
| --- | --- | --- | --- |
| context-compressor | ✗ | ✗ | ✗ |
| reme-ai | ✗ | ✗ | ✗ |
| Pipecat compaction | ✗ | ✗ | ✗ |
| LangChain SummaryMemory | ✗ | ✗ | ✗ |
| **voice-budget** | **✓** | **✓** | **✓** |

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT

## Releases

When you publish a new release make sure to follow these steps so CI can build and publish to PyPI automatically:

1. Bump the version in two places:
   - `pyproject.toml` (the `version` field)
   - `voice_budget/__init__.py` (the `__version__` string)

2. Run the test and lint suite locally:

```bash
# Run unit tests
pytest tests/ -v

# Optional: run ruff if installed
ruff check voice_budget/
```

3. Commit the version bump and push to the remote repository:

```bash
git add pyproject.toml voice_budget/__init__.py
git commit -m "chore(release): bump version x.y.z"
git push origin HEAD
```

4. Create a git tag and push it (GitHub Actions will publish on tags that start with `v`):

```bash
# Create an annotated tag
git tag -a vX.Y.Z -m "Release vX.Y.Z"
# Push the tag
git push origin vX.Y.Z
```

5. CI (GitHub Actions) will run tests/lint and, on tag pushes, build and publish to PyPI using the `PYPI_API_TOKEN` secret. Make sure the repository has this secret configured in Settings → Secrets → Actions as `PYPI_API_TOKEN` before pushing tags.

Notes:
- Use semantic versioning (MAJOR.MINOR.PATCH) for tags (for example `v0.2.1`).
- If a tag already exists and you truly need to move it, coordinate with maintainers: force-updating tags that are already published to PyPI is discouraged.
