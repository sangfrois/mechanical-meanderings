# Boundary-gated visual feedback (default)

Noninteractive and `--interactive` OSC launches now use **boundary mode**, not dynamic debounce.
It uses existing rt-stt `/trigger` packets; no new `/final` sender, TD edit, or
rt-stt edit is required. `/trigger 1` is already the intended VAD/phrase-timeout
boundary: `idle()` closes when **decided audio time** minus phrase end exceeds
`phrase_timeout`, not when subtitle text has been unchanged for a wall-clock
debounce. The previous default dynamic bridge ignored that boundary, including the interactive path. `--dynamic`
explicitly restores the old mid-speech refresh behavior; `--phrase-trigger`
retains the old FIFO. `--final-turn` remains the stricter authoritative-input
option documented below.

## Existing-protocol operator command

```bash
.venv/Scripts/python.exe -m mechanical_meanderings --model gemma4:12b --listen-host 127.0.0.1 --listen-port 9000 --output-host 127.0.0.1 --output-port 9001 --output-address /txt2img --forward-raw
```

Keep all existing `--num-ctx`, `--history-turns`, sampling, timeout and model
flags verbatim. No process was launched/restarted to install this change.
Raw forwarding is enabled unconditionally by the default CLI mode, including
duplicates, blank subtitle clears and trigger packets, independent of model
work. The optional activity packets are also passed through.

### Boundary mode contract and limits

- No debounce or refresh timer starts generation. `/trigger 0` opens/extends
  speech; `/trigger 1` completes the observed phrase. Early boundaries with no
  text are dropped, not latched. Repeated closures cannot create new work.
- If supplied, `/speech/activity 1` invalidates pending/in-flight visuals and
  blocks generation until `/speech/activity 0`. Silence only releases the veto;
  it does not manufacture a closure. `/trigger 1` is required; a closure cannot override
  an active veto. In channel mode use `/chN/speech/activity` alongside scoped
  transcription/trigger packets; aggregate duplicates are ignored.
- Growing/sliding snapshots use word-level suffix-prefix overlap deduplication;
  page flips append new words. Exact redraws cannot prove repeated speech.
  Empty clears preserve speech context. Accumulation is bounded to 8192
  characters per phrase (old pages are dropped on overflow with a drop event).
  Completed observed speech retains the existing chronological 128-entry
  ledger (`--speech-history`), including coalesced or failed generations.
  The Gemma model, token context and generated-history settings are unchanged.
- Pages arriving after closure enter a separate **ambiguous trailing buffer**:
  they may be old queued display pages or new previews. They never rewrite the
  previous completed phrase, never schedule regeneration, and are carried into
  the next `/trigger 0`/activity opening rather than cleared. Multiple chunk-0
  events preserve the candidate. Any changed nonempty snapshot invalidates
  pending/in-flight work even while closed; identical redraws/clears do not.
- An independent observation ledger retains every nonempty incoming subtitle
  snapshot verbatim (including redraws), bounded to `--speech-history` snapshots
  per channel (default 128), independently of the 128 completed-turn ledger.
  The **next actual Ollama request** includes retained distinct snapshots as
  exact user-message contents unless an identical message already represents
  them. A system annotation identifies these as overlapping provisional display
  observations with unknown ownership, not invented completed turns. This keeps
  ambiguous/gap-path speech in semantic context even if its visual request is
  skipped or coalesced. Capacity eviction emits a drop event; reset clears both
  lanes. Model/token context/history settings remain unchanged.
- An exact post-closure redraw stays in the raw/observation lanes but is not
  accumulated as a new trailing page. A genuinely repeated sentence after
  `/trigger 0` still counts as another completed turn. Older completed speech
  remains in context independently; it is not copied into a new phrase merely
  because its last screen page was redrawn.
- **This is phrase-gated, not a claim of perfect live VAD gating.** The checked
  out rt-stt uses VAD internally but sends no live `/speech/activity` start.
  Its closures are based on decided audio; physical speech may already have
  resumed before a delayed closure arrives. Its gap path may emit closure
  without a new `/trigger 0`, so following previews are indistinguishable from
  old page drains. Boundary mode retains these verbatim in model context and
  does not start a new request without an unambiguous opening signal. It may
  therefore skip a visual update rather than guess a new turn. There is no
  way to promise suppression during **all** physically ongoing speech from
  this existing wire alone. Supplied live activity improves the veto, but
  supersedes already dispatched work; its socket is closed and stale results
  are discarded. That does not prove the Ollama backend stopped computing.
- Subtitle previews may be frozen, corrected words may never reach OSC, pages
  may be dropped, and UDP has no IDs/order guarantee. Accumulation is useful
  observed context, **not** full authoritative final reconstruction. No sender,
  TD or transport changes are needed or included.

Offline tests cover real loopback OSC, default headless CLI with a local HTTP
fixture, supplied activity pauses longer than debounce, silence/phrase closure,
raw delivery during blocked inference, stale outputs and chronological page
context. An optional test imports the sibling rt-stt `PhraseTracker` read-only
and replays its actual queued-page/closure ordering; it ran on this checkout.

## Explicit authoritative completed-turn mode (opt-in)

`--final-turn` selects the strict noninteractive completed-input scheduler.
Interactive mediation/captions cannot be combined with `--final-turn`; default
interactive boundary mode nevertheless accepts the optional atomic `/final`
input through its ordinary boundary receiver. No
TouchDesigner project, callback, upstream sender, GPU placement, model, or
running process is changed by either implementation.

## Operator command

From `C:/Users/MAGen/git/mechanical-meanderings`, after you have provided the
final-input protocol below, the minimal command is:

```bash
.venv/Scripts/python.exe -m mechanical_meanderings --model gemma4:12b --ollama-url http://127.0.0.1:11434 --final-turn --forward-raw --listen-port 9000 --output-port 9001 --output-address /txt2img --speech-history 128
```

For your existing launch command, replace `--phrase-trigger` if present with
`--final-turn`, and keep/add `--forward-raw`. **Retain your existing
`--history-turns`, `--num-ctx`, sampling, timeout, host/port, and model flags
verbatim**; the minimal example above does not select a new token context.
Do not reduce context/model size to make this scheduling change. The Agent/CLI default
Ollama URL is now `http://127.0.0.1:11434`, avoiding the workstation's measured
localhost connection delay. Explicit user URLs (including remote hosts or
localhost) are preserved, never rewritten. You may
retain `--num-predict 96` if that was your chosen output-token maximum. This
is a maximum, not a statement that each request actually generated 96 tokens.
This document is not an instruction to restart the operator-owned runtime.

## Separate lanes: immediate subtitles and completed-turn generation

With `--forward-raw`, every accepted `/transcription` packet is forwarded as
`/transcription` immediately on receipt, with its original arguments,
including identical snapshots and empty subtitle clears. Neither debounce,
finalization, queueing, nor blocked HTTP inference gates forwarding. The
receiver does not wait for the agent lock when admitting transcript/final
input. In channel mode it forwards the accepted scoped input using the same
aggregate output addresses as the legacy bridge.

The **model lane** only runs on explicit completed text. It has one serial
HTTP request and at most one latest pending completed turn **per channel**
(maximum 64 channels), not a FIFO of obsolete visuals. A changed partial on a
channel invalidates that channel's running and pending visual requests,
without erasing any completed speech. A result whose speech revision changed
is discarded before OSC output or generated-history commit. Other channels
remain independent. Coalescing supersedes the real HTTP request and closes its socket. Default
requests also have a whole-request budget inheriting `--timeout` (default 120
seconds); `--generation-budget SECONDS` explicitly overrides it. A short budget
can reject healthy progressive output and is not a decode-stall repair. Legacy
injected blocking transports remain responsible for their own bound.
Continuous partial input can therefore hold visual updates until finalization;
raw transcription is still forwarded throughout.

Completed speech is retained chronologically in an independent bounded ledger
of `--speech-history` entries per channel (default 128, each at most 8192
characters). This retains intermediate completed inputs even when their visual
generations are coalesced or fail. Final-mode model messages are the system
instruction plus these completed speech inputs, oldest to newest. They do not
feed generated visual paraphrases back in place of actual speech. The normal
`--history-turns` setting is unchanged and bounds the generated-pair ledger;
only accepted/sent visual results enter that ledger. It is not the final-mode
speech-context limit. Setting it to zero does not disable the separate speech
ledger. The model's unchanged token window still determines what can fit;
the bridge does not claim unlimited memory, lossless incoming UDP, token-budget
truncation, or speech never supplied by the sender.

`/reset` invalidates running/pending requests and clears both ledgers.
In final mode `--reset-after` uses time between **completed speech admissions**,
not between generation starts. A new completed turn after that idle interval
starts a new speech context. Legacy modes retain their original reset timing.
Final-mode close has bounded waits (at most 0.75s shutdown-thread join, 0.1s
receiver join, 0.25s worker join, excluding OS scheduling). A blocked legacy injected-transport
worker may remain daemonized until its transport returns; it cannot emit a
late result or commit assistant history. No force-kill or retry cascade is used.

## Explicit final protocol and upstream prerequisite

Preferred atomic packet:

```text
/transcription  "live partial or subtitle page"   # optional, forwarded now
/final          "complete authoritative utterance text"  # exactly once per turn
```

`/final` is forwarded as `/transcription` when raw forwarding is enabled; it
is not a synthetic raw `/trigger`. The model's generated output still sends
`/txt2img` followed by `/txt2img/trigger 1` (or the configured output address
and its completion address). `/final` must contain the **whole final input**,
not just the last visible subtitle page. Repeated atomic finals represent
separate completed turns; this protocol has no message IDs/deduplication or
UDP reliability guarantee.

An explicitly supported two-packet alternative is:

```text
/transcription  "complete authoritative utterance text"
/trigger        1
```

It requires text-before-trigger arrival order. `/trigger 0` never generates;
an early `/trigger 1` with no stored text is dropped, **not latched onto a future
partial**. A trigger with stored text finalizes precisely that snapshot. There
is no safe way to infer from these untagged packets whether it belongs to a
future corrected final rather than an existing partial. Use atomic `/final`
if packet pairing/order is uncertain. Channel mode uses `/chN/final` or
`/chN/transcription` then `/chN/trigger 1` and ignores aggregate duplicates.
Manual/TUI submissions are explicit complete turns, not partial previews.

### What the checked-out rt-stt sender actually does

Read-only inspection of the sibling `../rt-stt/src/scarlett_live_whisper/osc.py`
found:

- `PhraseTracker.partial()` emits subtitle growth, **no trigger** (lines 166–189).
- `add()` extends phrases and emits `/trigger 0` for ordinary chunks (134–164).
- `_finalize()` writes full `p.text` through `sink.phrase(...)`, then sends
  `/trigger 1` (214–227). This is a **phrase** boundary (gap/timeout/flush),
  not necessarily each segmenter utterance boundary.
- `OscSink.phrase()` only calls the optional phrase writer (362–364); it does
  **not** publish the authoritative final text over OSC.
- `/transcription` carries display pages. Page refresh/rate limiting can leave
  final words queued when `/trigger 1` arrives, and frozen previews can differ
  from the authoritative corrected phrase.

Thus this checked-out sender does **not** establish the lossless final-text
protocol required here. Using its existing `/trigger 1` may generate from
whatever page was last displayed, not the full corrected speech. The bridge
cannot recover omitted words or identify an utterance boundary from that
stream alone. Before relying on final mode, an **upstream adapter outside TD**
must supply `/final FULL_TEXT` exactly once from the authoritative utterance
callback (utterance-level behavior), or from `PhraseTracker`'s `sink.phrase`
callback (phrase-level behavior). Leave the live subtitle stream unchanged.
Neither adapter nor upstream changes are included, and deployed sender/TD
compatibility is **unverified**. No live sender, TD, or Ollama was exercised.

## Attribution for multi-second or sporadic long requests

Noninteractive `TX` logs now include:

- actual `output_tokens` from `eval_count`, separately from `num_predict_max`;
- `output_eval` and `output_rate` from Ollama output evaluation telemetry;
- `input_tokens`, `input_eval`, `load`, and `ollama_total` from response fields;
- measured `http` round-trip time, `queue` wait, and `request` elapsed time.

Unavailable/invalid response metrics are `n/a`, never guessed. Final-mode
`DISCARD` logs/events retain timing even for completed obsolete requests.
`request` starts when the worker selects work; `http` surrounds the transport
call. `queue` is selection delay after completion admission, not speech
recognition time. Final mode has no debounce timer. Output remains capped at
20 words after HTTP completion; that cap does not stop model decoding early.
The time between `GENERATE` and `TX` is therefore not explained by the input
debounce. An observed 46.75s request needs its **own** input-eval/load/output/
HTTP telemetry; a later GPU-utilization snapshot cannot prove the cause of
that earlier spike. Operator-provided runner logs separately identify slow
requests with decoding durations of 43.527s for 27 tokens and 48.291s for
26 tokens, while normal decoding took about 0.3s. Those observed spikes were
in output decoding, not an inference from debounce or a later GPU snapshot.
Operator read-only `/api/version` probes also measured about 2s for localhost
versus about 0.001s for literal loopback, motivating the new local URL default.
These are distinct causes; loopback does not claim to cure runner decoding
stalls. No new speedup factor or GPU-placement cause is claimed.

## Offline verification

```bash
.venv/Scripts/python.exe -m pytest -q tests/test_boundary.py tests/test_final_turn.py tests/test_timing.py tests/test_agent.py tests/test_bridge.py tests/test_cli.py
.venv/Scripts/python.exe -m pytest -q tests/test_speech_pipeline_contract.py tests/test_interactive_boundary.py tests/test_mediation_control.py tests/test_generation_budget.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m mechanical_meanderings --help
```

Tests use isolated loopback OSC/HTTP and gated fake model transport, never a
live Ollama/TD endpoint. They cover partial/final ordering, in-flight and pending
supersession, completed context retention/bounds, immediate raw forwarding
while inference/agent locking blocks, independent channels, reset/idle reset,
failed-model context retention, bounded shutdown/no late commits, CLI wiring,
and timing telemetry/missing metrics.
