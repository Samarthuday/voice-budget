# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Prepare for v0.2.2
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

