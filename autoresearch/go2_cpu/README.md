# Autoresearch: Go2 Blueprint CPU/Sys Footprint

## Goal
Minimize `user + sys` CPU time for a full Go2 basic blueprint replay run.

## Command
```bash
time uv run dimos --replay --viewer=none --replay-dir=unitree_go2_bigoffice run unitree-go2
```

## Metric
- **Primary score**: `user + sys` seconds from `time` (lower is better)
- **Secondary**: peak CPU%, peak memory, avg thread count, I/O bytes (reported, not optimized)

## Constraints
- Only modify `optimizations.py` — monkey-patches applied before the replay runs
- All LCM messages must still flow correctly (same count per topic, same payloads)
- If validation fails, the run is rejected (score = infinity)

## Approach
1. Run `eval.py` once with no optimizations to get baseline + cProfile data
2. Read `profile_output.txt` to identify hot functions
3. Add monkey-patches to `optimizations.py` targeting the hot paths
4. Re-run `eval.py` — check score improved and validation passed
5. Iterate

## Known Hot Paths (from codebase exploration)
- `lcmservice._lcm_loop` — 50ms polling timeout, runs per module instance
- `ModuleBase.__init__` — creates 50-worker thread pool per module
- `connection.publish_camera_info` — 1Hz busy-wait loop in daemon thread
- LCM serialization/deserialization overhead
- Per-module asyncio event loop (could share across modules)

## Files
- `eval.py` — profile + benchmark + validate
- `optimizations.py` — the ONE file you edit
- `run.sh` — entry point
- `baseline_record.json` — generated on first run (do not edit)
