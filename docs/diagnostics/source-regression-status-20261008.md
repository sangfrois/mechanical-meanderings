# Source-regression diagnosis: verified status

Scope: `C:/Users/MAGen/git/mechanical-meanderings`, compared against HEAD `67c580b`. The user's original revision on the other working computer has not been established. No live installation writes, audio capture, OSC injections, model/context changes or runtime restarts were performed during this continuation.

## Confirmed regression corrected

The introduced five-second total-generation default rejected healthy progressive responses. Before the correction, `test_default_accepts_six_seconds_of_progress_without_blocking_raw_udp` failed with `GenerationTimeout` after an otherwise healthy stream. After the correction, the same real loopback HTTP/UDP test completed in 6.005 seconds, forwarded raw redraws concurrently, committed one result and left no deadline watcher.

The default budget now inherits the existing agent/CLI HTTP timeout (120 seconds by default). Explicit `--generation-budget` values remain supported. The explicit five-second stall fixture ended at 5.004 seconds with no visual/history commit. This corrects a cutoff regression, not the intermittent 23–57-second decode fault.

Source changes: `mechanical_meanderings/final_turn.py`, `interactive.py`, `__main__.py`; default/fixture tests and documentation updated. The running service has not been relaunched and is not claimed to contain these source corrections.

## Inference-source audit and route check

Historical user logs in session `20261008_161926_ac87d3`, messages 12119 and 12121, contain slow plain `GENERATE` requests as well as interactive `MEDIATE` requests. The plain commands explicitly set `--num-ctx 128000` and `--history-turns 10` (some later runs use three history turns). Plain output delays include 45.55 seconds, 55.72 seconds and 23.37 seconds. Those are application elapsed timings, not newly measured server timings.

Parent independently captured production HEAD/current payloads with injected transports that raise before networking. Controls used the actual supplied 128000 context, ten history turns, no supplied seed and no supplied prediction override. Network requests: zero.

| Route | System characters | Serialized schema characters | Options |
|---|---:|---:|---|
| HEAD plain | 4331 | 0 | num_ctx=128000 |
| Current plain | 5784 | 0 | num_ctx=128000 |
| HEAD interactive | 8225 | 877 | num_ctx=128000, num_predict=512 |
| Current interactive | 14963 | 1844 | num_ctx=128000, num_predict=512 |

Both plain envelopes have identical model/options/think/constructed-stream fields. The plain system prompt changed and its requested word allowance expanded from 20 to 50. Its final word normalization remains 20. This adds possible output work but does not establish a dramatic per-token stall.

The expanded interactive schema is a legitimate diagnostic seam but cannot be the common explanation for schema-free plain stalls. Constructed `stream=false` above describes captured agent payloads before controlled transport; controlled HTTP changes the outer request to streaming.

## TD candidate checked against deployment

The parent reproduced 65 settings-update callbacks for an enabled eight-second Step transition in the pure fixture. HEAD's analogous path has no automated Step writes. This is not a measured live GPU stall.

A narrow read-only query of TD instance `127.0.0.1_20856` found that its interactivity logic matches the working source, while `Speechdrive=false` and `Stepsdrive=false`, with `Smoothing=3.6`. Its deployed Step callback still clamps at 10/31 rather than the source fixture's 1/2 floors. Therefore the enabled fixture is not proof of current live update traffic or historical toggle state. No TD parameters or DATs were changed.

## Outstanding reproduction prerequisite

Fresh read-only `/api/ps` at `2026-10-09T00:23:38.335545+00:00` showed resident `gemma4:12b` at context 32768. The prior direct/A/B trials reused that resident context to avoid a runner reload. They did not reproduce the user's failing 128000-context configuration and do not clear the changes.

A request for permission to run the actual 128000-context baseline/current replay timed out without an answer. No permission was assumed and no context-changing request was sent. The intermittent decode-stall root cause remains unresolved; no driver/GPU/schema fix is claimed.

## Fresh verification

Focused combined speech/budget selection: 115 passed in 73.41 seconds.
Full suite: 313 passed, 5 failed in 117.08 seconds. All five failures are in `tests/test_td_speech_controls.py`, expecting old 10/31 limits while source targets/clamps use 1/2. These tests and TD source were left untouched by this continuation. No full-suite pass is claimed.

Verification is source/isolated-fixture evidence, not microphone-to-display timing or a live 128000-context rehearsal.
