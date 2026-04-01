# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.3.0] - 2026-04-01
- LiveKit agent integration (`VoiceBudgetAgent` in `livekit_integration.py`)
  - Full TTFT measurement and context compression for LiveKit voice agents
  - TTFT timing now starts after compression, forwards `**kwargs`, and records first-chunk latency for streaming responses
  - See README section on LiveKit integration
- Pipecat integration already available (`VoiceBudgetProcessor` in `pipecat_integration.py`)
- Token-count-based strategy selection in `BudgetCompressor` with threshold ordering tests
- Validate `semantic_threshold` / `summarise_threshold` inputs in `BudgetCompressor`
- Improved compressor fallback selection to prefer the lowest resulting token count
- Improved LiveKit error logging and README streaming example
- Removed the unused `summarise_threshold` argument from `VoiceBudgetAgent`
- Minor cleanup: remove unused `run.py` (non-production demo). 
- Documentation updates and examples improvements.

## [0.2.1] - 2026-03-19
### Fixed
- Align public API: renamed `weekly_report` -> `snapshot_report` in `core.py`.
- Export `DEFAULT_SUMMARY_PROMPT` from `compressors.py` for easier reuse.
- Removed `accuracy_fn` parameter from `VoiceBudget` constructor in `wrapper.py` (simpler API).

### Docs
- Added Pipecat integration note to `README.md` explaining how to extend `VoiceBudgetProcessor` with `pipecat.processors.frame_processor.FrameProcessor`.
- Expanded release and tagging instructions in `README.md` and `CONTRIBUTING.md`.

### Other
- Version bumped to `0.2.1`.
