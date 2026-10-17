# High-velocity speech-to-image prompt engine

`python -m mechanical_meanderings.velocity` is a small, independent route for the installation:

```text
rt-stt authoritative transcript.jsonl
                 |
                 v
sentence-aware coalescer -> Ollama gemma3:4b -> /txt2img + /txt2img/trigger -> TouchDesigner
```

It does not capture audio, load Whisper, launch Ollama, restart TouchDesigner, or alter the StreamDiffusion patch. The currently running rt-stt can keep sending subtitles directly to TouchDesigner on its existing OSC route.

## Why it follows JSONL instead of subtitle OSC

rt-stt writes one authoritative record only after an utterance is decoded. Its `/transcription` output is a paginated screen subtitle: provisional rewrites, queued page drains, and `/trigger 0` ordering make it unsuitable as an exact final-text packet. Reading new JSONL records avoids generating from half-rewritten subtitle pages and requires no rt-stt restart or port reroute.

By default, existing records are skipped. Only records appended after this engine starts are processed. `--from-start` is intended for controlled replay, not a live installation.

## Sentence timing

The ASR record arrives after its acoustic hangover, so the scheduler applies a short semantic delay rather than rt-stt's longer phrase timeout:

| Boundary | Default additional wait |
|---|---:|
| `.`, `?`, `!`, `…` | 100 ms |
| ordinary complete fragment | 350 ms |
| comma, colon, semicolon, dash, trailing conjunction | 850 ms |
| rt-stt `cut=maxlen` forced split | 1200 ms |
| hard maximum for a growing unrendered batch | 2500 ms |

New records arriving before the deadline are joined. While Ollama is busy, speech continues accumulating; there is only one inference request at a time. If newer speech arrives during inference, the obsolete result is discarded and the next request contains the full unrendered batch. Chronological speech is not deleted merely because visual work is coalesced.

`--min-interval 0.75` also caps generation starts. This, `num_predict=64`, `num_ctx=4096`, response-capacity limits, and the 20-word post-generation cap bound routine work. HTTP timeout measures socket inactivity, not total generation duration: loading and progressive decoding are not cut off by a whole-request watchdog. Change these flags deliberately rather than lowering rt-stt quality.

Defensive capacities are 64 channels, 8192 characters per utterance, 32768 pending characters per channel, 65536 characters per JSONL record, and 65536 response bytes. Over-capacity records are logged and rejected rather than allowed to exhaust memory.

## Model

The installed four-billion-parameter model on this workstation is `gemma3:4b` (4.3B, Q4_K_M). That is the default. It is not a model named `gemma4:4b`; no such local tag is currently installed.

The request uses:

- `think: false`
- `stream: false`
- `keep_alive: -1`
- `num_ctx: 4096`
- `num_predict: 64`
- `temperature: 1.3`
- `top_p: 0.9`
- six accepted speech/prompt pairs of contextual history by default (configurable from 0 to 30)

The exact instruction is editable at `mechanical_meanderings/prompts/velocity.txt2img.md`. It asks for one coherent, concrete 16–20-word scene rather than a short list of abstract tags. Output is normalized to one line and hard-capped at 20 whitespace-separated words before OSC delivery.

Loading `gemma3:4b` may evict another resident Ollama model or change GPU memory pressure. The engine does not preload it. Start it only when the operator is ready to make that runtime change.

Ollama is loopback-only by default because requests contain private microphone-derived speech and retained context. A non-loopback URL is rejected unless the operator explicitly supplies `--allow-remote-ollama`; use HTTPS for any authorized remote endpoint.

## Inspect without inference

```bash
cd C:/Users/MAGen/git/mechanical-meanderings
.venv/Scripts/python.exe -m mechanical_meanderings.velocity --help
.venv/Scripts/python.exe -m mechanical_meanderings.velocity --print-system-prompt
```

Neither command opens OSC nor calls Ollama.

## Live command

Confirm the active TouchDesigner OSC input port and address first. For the current intended route:

```bash
cd C:/Users/MAGen/git/mechanical-meanderings
.venv/Scripts/python.exe -u -m mechanical_meanderings.velocity \
  --transcript C:/Users/MAGen/git/rt-stt/transcript.jsonl \
  --model gemma3:4b \
  --output-host 127.0.0.1 --output-port 9001 \
  --output-address /txt2img
```

This sends the same `/txt2img` and `/txt2img/trigger` route as the other
Mechanical Meanderings prompt modules: TouchDesigner receives it on UDP 9001.
It does not need rt-stt's OSC port. Keep rt-stt running exactly as configured
if its JSONL file is the path above.

Use `--no-osc` for a controlled console-only rehearsal; it still calls Ollama. Use `--text "..."` for a one-shot model check; it prints a prompt and sends no OSC. Both can load or replace the resident Ollama model.

## Tuning

```text
--terminal-delay 0.10
--complete-delay 0.35
--continuation-delay 0.85
--forced-cut-delay 1.20
--max-batch-wait 2.50
--min-interval 0.75
--history-turns 6
--num-ctx 4096
--num-predict 64
--timeout 120
```

For less GPU work, raise `--min-interval`; do not reduce speech fidelity. For more fragmented-speech continuity, raise `--continuation-delay` or `--history-turns`. A larger context/history increases prompt evaluation cost. `--timeout` now bounds socket inactivity (120 seconds by default), not the complete exchange. A request can run longer while bytes continue arriving. An explicit old `--timeout 8` still allows only eight seconds without socket progress, so remove that override when allowing a cold model to load. There is no whole-request cancellation watcher. A genuinely stalled socket still fails without committing a partial prompt, and completed speech remains pending. Disconnecting does not prove backend computation stopped. Existing running engines must be restarted by the operator to load this source change; rt-stt, TouchDesigner and StreamDiffusion need not restart.

OSC prompt and trigger are two UDP datagrams and cannot be transactional. If the prompt datagram is sent but the trigger send raises an error, the engine records partial delivery and consumes that speech so it will not duplicate the prompt automatically. The operator can diagnose the missing trigger from the explicit partial-delivery log.

## Verification boundary

Existing automated tests cover timing classes, hard batch deadlines, context construction, 20-word enforcement, stale and shutdown-output rejection, capacity limits, malformed JSONL handling, partial OSC delivery, tail-from-end behavior, and real loopback OSC delivery. The legacy whole-request HTTP deadline test describes the previous policy and needs replacement with idle/progressive-stream coverage. No tests or model rehearsals were run for the idle-timeout change at the operator's request.
