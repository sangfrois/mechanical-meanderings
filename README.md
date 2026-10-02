# Mechanical Meanderings

An interactive installation exploring self-perception and agency through a generative, deforming mirror. Participants speak and move; speech conditions the transformation of their live camera image. What returns can remain recognizable while departing from what they intended.

This repository implements the **text-to-visual-prompt service**, not the entire installation:

```text
microphone -> rt-stt -> OSC -> this bridge -> Ollama
                                 |
                                 v
                          generated prompt -> OSC -> TouchDesigner / StreamDiffusion
                                                       ^
                                                       |
                                                   live camera
```

The bridge receives text, not camera images or microphone audio. **rt-stt** provides speech recognition; Ollama serves the language model; the separately opened TouchDesigner patch handles camera input, image generation, and composition. None of those external services is launched automatically by the Python bridge.

## Conceptual framing

Mechanical Meanderings is a separate project from Monolith. Its framing here follows François Lespinasse's draft, *Mechanical Meanderings: Self-Perception and Agency in AI-Mediated Interactive Art*.

The draft places the installation in the lineage of Gordon Pask's **maverick machine**: an encounter involving partial autonomy, exploration, and negotiation rather than obedient execution alone. Speech recognition, linguistic reformulation, and image synthesis can each transform what survives from the participant's contribution. Their combined divergence is an artistic possibility, not a requirement that every output misunderstand the speaker.

The central distinction is between producing an artifact alongside a machine and encountering a transformed return of oneself. The draft calls the latter **hybrid subjectivity**, drawing on Tuer: the participant's embodied presence is implicated in what they perceive, rather than remaining wholly outside a shared artifact.

Its proposed **agency-coupling** framework distinguishes:

- **Intentionality:** whether participants consciously formulate the incoming signal. Spoken input provides a known contribution they can attempt to steer.
- **Traceability:** whether participants can follow how that contribution is transformed. Traceability may develop through repeated interaction; it is not synonymous with literal fidelity or deterministic output.

Mechanical Meanderings is situated as high-intentionality with initially low traceability. The comparison with the EEG-driven *Bio-Mechanical Poet* clarifies how input modality changes the encounter. Prospective anticipation and retrospective interpretation, and pre-reflective and reflective agency, provide distinct lenses for studying it.

These are **conceptual propositions, not demonstrated neurological outcomes** of this code. The service does not measure agency, flow, prediction errors, or self-perception. Semantic distance alone would not establish those experiences. Network failures, transcription errors, and excessive latency should not automatically be celebrated as artistic agency.

### Research description versus current implementation

The draft describes a version using Whisper, goofi-pipe, Llama 3.1-8B, and StreamDiffusion in TouchDesigner. This repository's current route uses **rt-stt and a direct OSC-to-Ollama bridge**, without requiring goofi-pipe. The default model is `gemma4:12b`; it is configurable and is not tied to the assistant model used to edit this repository.

The current [system instruction](mechanical_meanderings/system.txt) prioritizes the speaker's subject, corrections, and explicit aesthetic choices while allowing situated visual interpretation. It does not instruct the model to ignore core subjects. The bridge caps responses at **20 whitespace-separated words** (and 8192 characters), retains three successful input/output pairs per channel by default, and supports explicit sampling overrides. Temperature is left to the model unless set; elevated temperature is an exploration choice, not the default.

See [RESEARCH.md](RESEARCH.md) for additional prompt-design rationale. Neither document substitutes for evaluating the rendered installation with participants.

## Quick start and installation status

Requires Git, [uv](https://docs.astral.sh/uv/), an appropriate audio device/driver for rt-stt, and a running Ollama server with your chosen model. TouchDesigner and its diffusion setup are external prerequisites for the visual installation, not Python dependencies. GPU libraries and Whisper/Ollama model downloads are separate from installing this bridge.

**Current checkout:** rt-stt lives at `../rt-stt`; it is not yet a Git submodule or a declared dependency of this project. `uv sync` alone therefore does not install it. The local rt-stt OSC changes are uncommitted; its current commit `44f7fed788ccc3cc4da5a4b54bd0311c88f115e1` lacks `osc.py`. Pinning that commit would not reproduce the setup documented below.

For the existing sibling checkout, install both packages into this project's environment without moving or overwriting either repository:

```bash
# From mechanical-meanderings; use standalone CPython, not Conda.
uv sync --python 3.11 --managed-python
uv pip install --python .venv/Scripts/python.exe -e ../rt-stt
# On Linux/macOS use .venv/bin/python instead.
uv run --no-sync mechanical-meanderings --help
uv run --no-sync scarlett-transcribe --help
```

This is a **temporary local-development installation**, not a locked, portable release. `--no-sync` preserves the manually installed sibling dependency. On systems missing CUDA runtime libraries, rt-stt also provides a `cuda` extra; consult its requirements before downloading those large packages.

The intended portable layout is an `rt-stt/` Git submodule pinned to a published OSC-capable commit, with `scarlett-live-whisper` declared as a dependency and mapped to that path in `[tool.uv.sources]`. Git initializes the submodule (`git submodule update --init --recursive`); uv installs its Python package (`uv sync`). **uv does not initialize Git submodules.** That release wiring is pending publication of the required rt-stt changes.

To boot the bridge in the foreground using the local-development environment above:

```bash
MSYS_NO_PATHCONV=1 uv run --no-sync mechanical-meanderings \
  --listen-host 127.0.0.1 --listen-port 9000 \
  --output-host 127.0.0.1 --output-port 9001 --output-address /txt2img \
  --history-turns 3 --forward-raw --update-interval 2 --debounce 0.6
```

In a second terminal, from this same repository, identify inputs with `uv run --no-sync scarlett-probe`. For a Scarlett microphone on physical input 1:

```bash
uv run --no-sync scarlett-transcribe --asio --channels 8 --only 1 \
  --language auto --osc-ip 127.0.0.1 --osc-port 9000 --partial 0.7 --hold 0
```

Choose an actual supported device/channel configuration; alternatives follow below. Check `ollama list` first and run `ollama serve` separately only if no server is already running. Model installation is an explicit operator step, not an automatic download. Open the TouchDesigner patch separately when you want images. Ctrl+C stops each foreground process.

**There is currently no `boot_meanderings.sh` in this repository.** Earlier chat instructions referring to that launcher were incorrect. The commands above, and the detailed operating guide below, are the supported launch instructions.

## Live TUI: sampling, trajectory, and metrics

Install the optional panel and test dependencies into this project's environment:

```bash
uv sync --python 3.11 --managed-python --extra tui --extra dev
```

If you manually installed sibling rt-stt into this environment, sync may remove it; reinstall it with the temporary editable-install command above, then use `--no-sync`. The TUI itself needs neither rt-stt nor audio hardware.

Stop the old bridge before launching this replacement (do not bind two bridges to 9000):

```bash
MSYS_NO_PATHCONV=1 uv run --no-sync mechanical-meanderings --tui \
  --listen-host 127.0.0.1 --listen-port 9000 \
  --output-host 127.0.0.1 --output-port 9001 --output-address /txt2img \
  --model gemma4:12b --history-turns 3 --forward-raw
```

Use a wide terminal (130 columns recommended). The left panel scrolls; Tab moves between fields. Type manual text and press Enter or Send to explore without a microphone. This still emits real UDP prompts to the configured destination; it does not launch Ollama, capture, or TouchDesigner.

- **Ctrl+A / Apply:** validate and atomically apply the complete draft for the **next selected request**. In-flight inference retains its original settings, recorded alongside its output.
- **Temperature, top-p, top-k, repeat penalty, seed, output budget, context tokens:** blank means no Ollama override, not a known effective default. The playful presets stage four sampling fields only; Apply is still required. Context/budget/seed/history are preserved when selecting a preset, including “Model defaults”.
- **History:** successful input/output pairs per channel, distinct from the token window. Shrinking it takes effect on the next request; increasing it cannot recover discarded turns. Debounce and start interval can also be tuned live.
- **Ctrl+P / Pause:** hold new generation starts; raw forwarding and an in-flight request continue. **Ctrl+R / Reset:** clear history and pending work and suppress an obsolete in-flight output; settings remain unchanged.
- **Ctrl+Q:** stop the receiver and worker, close UDP and the journal. A running HTTP request cannot be cancelled immediately; shutdown can wait for its configured timeout (default 120 seconds).

Trajectory view: at most 300 lines, a 256-event display queue with visible eviction count, and the last 120 successful latency samples. Channel and input sizes are bounded. JSONL in Git-ignored `sessions/` records inputs, settings, starts, outputs, errors, resets, and runtime metadata before display, with flush + fsync per record. Display evictions do not delete disk records. Files contain private speech and grow until the session ends; archive/delete them according to your retention policy. Disk-write failure is latched visibly and subsequent records are **not guaranteed durable**. There is no replay/resume or automatic log rotation. Headless mode can opt in with `--record --session-dir PATH`.

Metrics are measured bridge generation-to-send time (excluding queue wait, which is recorded separately), session-average successful sends/minute, and Ollama-reported decode tokens/second when available. These are **not** microphone-to-image latency, acknowledged delivery, or rendered FPS. STT latency/audio levels and TD frames/FPS/delivery remain explicitly unavailable. Sampling variation and technical failures are displayed separately.

All sampling overrides also work without the TUI: `--temperature`, `--top-p`, `--top-k`, `--repeat-penalty`, `--seed`, `--num-predict`, and `--num-ctx`.

## 1. Routing and prerequisites

```text
rt-stt sends localhost:9000
          |
Mechanical Meanderings listens on 9000
          |  local Ollama /api/chat
          v
TD receives localhost:9001
    /txt2img          generated prompt
    /txt2img/trigger  1 after prompt delivery
    /transcription    original subtitle text, with --forward-raw
    /trigger          original STT trigger, with --forward-raw
```

Only the agent should listen on 9000. TD listens on 9001. Do not start duplicate service instances. These examples set ports explicitly because the original CLI defaults use the opposite ports and `/prompt`.

Requirements: Python >=3.10, python-osc, a running Ollama server with the requested model, and an OSC receiver. rt-stt is a separate speech-to-text project, not text-to-speech. Its own Whisper/audio requirements are documented in `../rt-stt/README.md`.

On this workstation use the already installed `../rt-stt/.venv/Scripts/python.exe`. For a separate environment on another machine, from this repository:

```bash
python -m venv .venv
# Activate the environment for your shell, then:
python -m pip install -e .
python -m mechanical_meanderings --help
ollama list
```

Install/download models only if needed. Do not start a second Ollama server if the desktop/server instance already runs. This guide uses Git Bash on Windows: `MSYS_NO_PATHCONV=1` prevents `/txt2img` being rewritten into a Windows path. A multiline command needs a backslash as the LAST character of every continued line, with no trailing spaces.

## 2. Start the agent (terminal 1)

```bash
cd ~/git/mechanical-meanderings
MSYS_NO_PATHCONV=1 ../rt-stt/.venv/Scripts/python.exe -u -m mechanical_meanderings \
  --listen-host 127.0.0.1 --listen-port 9000 \
  --output-host 127.0.0.1 --output-port 9001 --output-address /txt2img \
  --history-turns 3 --forward-raw \
  --update-interval 2 --debounce 0.6
```

Single-line equivalent, from the same directory:

```bash
MSYS_NO_PATHCONV=1 ../rt-stt/.venv/Scripts/python.exe -u -m mechanical_meanderings --listen-host 127.0.0.1 --listen-port 9000 --output-host 127.0.0.1 --output-port 9001 --output-address /txt2img --history-turns 3 --forward-raw --update-interval 2 --debounce 0.6
```

Normal logs show generation input, output, elapsed inference time, and errors. Add `2>&1 | tee -a dynamic-service.log` to save them as well. `--verbose` deliberately enables noisy per-packet logs; leave it off normally.

## 3. Select a capture scenario (terminal 2)

Run one suitable rt-stt command, not all of them. In each case keep OSC output on 9000. Identify your device and physical channel numbers before choosing inputs:

```bash
cd ~/git/rt-stt
uv run scarlett-probe
uv run scarlett-transcribe --help
```

### A. One live microphone on Scarlett input 1

```bash
cd ~/git/rt-stt
uv run scarlett-transcribe --asio --channels 8 --only 1 --language auto \
  --osc-ip 127.0.0.1 --osc-port 9000 --partial 0.7 --hold 0
```

Change `--only 1` to the connected input, e.g. `--only 3`. `--channels` is the number of hardware channels opened; `--only` selects which of those are transcribed. Channel numbering is 1-based.

Use `--language fr` or `--language en` for a known language; use `auto` for mixed/unknown language. Language detection on very short speech can be wrong. Verify representative speech before a performance.

### B. Several live microphones, isolated context per mic

Example: physical Scarlett inputs 1, 2, and 5.

```bash
cd ~/git/rt-stt
uv run scarlett-transcribe --asio --channels 8 --only 1,2,5 \
  --names Alice,Bob,,,Room --language auto --osc-per-channel \
  --osc-ip 127.0.0.1 --osc-port 9000 --partial 0.7 --hold 0
```

Start the agent with `--channel-mode` added to the section 2 command. Both flags are necessary:

- rt-stt `--osc-per-channel` emits `/ch1/transcription`, `/ch2/transcription`, `/ch5/transcription`, etc., in addition to aggregate messages.
- Agent `--channel-mode` consumes only `/chN/...` and ignores aggregate duplicates.
- Each channel gets its own history window and latest pending text. Names are console labels, not OSC address names.
- Do not feed multiple microphones into aggregate-only mode if context isolation matters: their snapshots would share one history and overwrite each other.

**Output limitation:** channel-mode isolates context, but currently all generated prompts still go to the same `/txt2img` output, and forwarded subtitles go to the same `/transcription`. There is no channel ID on those output messages. This is suitable for several microphones influencing one shared visual destination, not independent per-mic visual outputs. Separate destinations require separate routed agent pipelines or a future per-channel output feature. Do not assume `/chN/txt2img` exists.

Generation is serial across microphones. Two seconds is a scheduling target, not a guaranteed per-mic output rate; more active microphones share inference capacity. This is not diarization: two people using one mic share its channel history.

### C. Computer playback / Scarlett Loop 1

The workstation's observed Loop 1 channel is 9:

```bash
cd ~/git/rt-stt
uv run scarlett-transcribe --asio --channels 10 --only 9 --language fr \
  --osc-ip 127.0.0.1 --osc-port 9000 --partial 0.7 --hold 0
```

This captures routed playback, not a physical mic. Verify loopback numbering on other interfaces. To avoid transcribing computer audio during a live-mic installation, open only the analogue channels or explicitly exclude loopback with `--only`.

### D. Non-ASIO / ordinary Windows input device

First identify a device with `scarlett-probe`; substitute its actual name for the placeholder:

```bash
cd ~/git/rt-stt
uv run scarlett-transcribe --device "YOUR INPUT DEVICE NAME" --channels 1 --only 1 \
  --language auto --osc-ip 127.0.0.1 --osc-port 9000 --partial 0.7 --hold 0
```

Use a channel count the device really supports. For Scarlett's WDM endpoint, physical Analogue 1+2 are available; higher advertised channels may be padding. Use direct `--asio` for higher physical inputs. This generic-device example is a configuration template, not a claim that every Windows input was hardware-tested.

### Audio clock and capture tuning

Leave `--samplerate` off in ASIO mode to adopt the existing Scarlett clock. Keep Windows playback and the ASIO control-panel rate aligned. Do not retune the interface merely to change prompt cadence.

| rt-stt option | Purpose |
|---|---|
| `--threshold-db -42` | Default speech gate. Raise toward zero for noise; lower for quiet speech. Calibrate from real levels. |
| `--hangover 0.7` | Silence needed to finish an utterance; reducing it can split words/ideas more aggressively. |
| `--max-utt 20` | Maximum utterance duration before forced split; not the LLM refresh interval. |
| `--partial 0.7` | Re-read open speech periodically. Keep >0 for dynamic prompting before pauses. |
| `--phrase-timeout 3` | STT phrase boundary; does not gate the default dynamic LLM mode. |
| `--no-meter` | Hide the live audio meter. |

### Subtitle pages versus full transcripts

The current rt-stt parser uses `--line-chars` (default 42), `--lines` (default 2), and `--min-update` (default 0.5 seconds). It no longer accepts the older `--subtitle-chars` flag. `--hold 0` keeps the last page displayed; it does NOT disable paging or preserve the whole transcript.

The agent receives the actual `/transcription` page snapshots, not the complete JSONL transcript. Dynamic sampling and prior turns provide context, but words that scroll past between sampled snapshots can be missed. Increasing the display page dimensions changes subtitle layout too. There is currently no separate lossless full-transcript input stream in this bridge; don't describe its memory as an exhaustive recording of everything spoken.

## 4. Dynamic update rate

Default mode does NOT wait for `/trigger`.

- `--debounce 0.6`: a short pause in changing text makes a pending snapshot eligible.
- `--update-interval 2`: continuous changes become eligible after this interval; also spaces generation starts per channel.
- One request runs at a time. While it runs, incoming changes replace pending text instead of building a FIFO backlog.
- Identical text and blank subtitle clears do not generate another prompt.
- A running request can still finish with an older snapshot; the next request uses the latest text. This avoids endless cancellation during continuous speech.

Examples: `--update-interval 1 --debounce 0.3` for faster reactions, `--update-interval 4 --debounce 1` for less churn. These are tuning suggestions, not measured throughput guarantees. Whisper, Gemma, and diffusion may compete for GPU memory/time.

`--phrase-trigger` restores legacy phrase-end behavior using `/trigger 1`. Only that legacy mode uses `--queue-size` (default 8); it is not the dynamic mode's backlog control.

## 5. Context size and participant boundaries

### Conversation history: `--history-turns`

This is the maximum number of previous successful input/prompt pairs retained **per channel**:

| Setting | Use |
|---|---|
| `--history-turns 0` | Independent snapshots, no earlier turns. |
| `--history-turns 1` | Immediate continuity with little older influence. |
| `--history-turns 3` | Default, short rolling conversation. |
| `--history-turns 8` | Longer narrative continuity; more context cost and risk of old motifs persisting. |

The system instruction and current input are additional to the retained pairs. Oldest pairs fall out automatically. A turn means a successful LLM generation, not every OSC partial, sentence, person, or fixed number of seconds. Consecutive dynamic turns can contain overlapping text. History is not summarized or persisted across restarts. Increasing history cannot recover speech the bridge never sampled.

### Model token window: Ollama `num_ctx`

A token window is different from a turn count. It must accommodate the system instruction, selected history, current input, chat formatting, and generated response. More tokens generally consume more memory; more history does not automatically enlarge the token window.

Use `--num-ctx 8192` or the TUI's **Context tokens** field to send an explicit `options.num_ctx` override. Blank/omitted leaves Ollama's model/server default untouched. The bridge does not tokenize or truncate history to a token budget, so do not assume all requested history fits or that the advertised model maximum is the active window. Larger windows may compete with diffusion for VRAM.

Alternatively, to store a model-specific window without changing the original `gemma4:12b`, create a text file named `Modelfile.context` in this repository with:

```text
FROM gemma4:12b
PARAMETER num_ctx 8192
```

Then explicitly create a separate Ollama model configuration and select it in the agent:

```bash
ollama create mechanical-meanderings-context -f Modelfile.context
ollama show --modelfile mechanical-meanderings-context
```

Add `--model mechanical-meanderings-context` to the normal agent command. Change `8192` in the file and recreate that configuration if you need a different window. The base remains Gemma; the bridge still supplies system.txt. This is an optional documented configuration, not a change already applied to the running service. Choose a window that fits your GPU workload rather than automatically maximizing it.

Ollama reference: https://docs.ollama.com/modelfile (`PARAMETER num_ctx`).

### Resetting between people or scenes

- `--reset-after 30` clears a channel's history on its next generation when at least 30 seconds elapsed since its previous generation started. This is generation-input inactivity, not microphone silence detection or person recognition.
- Send OSC `/reset` to port 9000 to clear all histories and discard obsolete pending output.
- Restart also clears history.
- Saying 'start over' is interpreted by the model; it is not a guaranteed programmatic memory erase.

From this repository, reset using the existing Python environment:

```bash
../rt-stt/.venv/Scripts/python.exe -c "from pythonosc.udp_client import SimpleUDPClient; c=SimpleUDPClient('127.0.0.1',9000); c.send_message('/reset',1); c.close()"
```

## 6. Other models, instructions, and network layouts

- `--model gemma4:12b`: exact Ollama model tag.
- `--ollama-url http://127.0.0.1:11434`: Ollama server base URL.
- `--timeout 120`: maximum HTTP wait in seconds, not a cadence setting.
- `--system-prompt path/to/instruction.txt`: replace the packaged instruction at startup. Restart to load edits. The runtime still caps output at 20 words.
- `--output-address /txt2img`: any desired OSC output address. `/transcription` is a special drop-in choice whose completion address is `/trigger`; other choices append `/trigger`.
- Omit `--forward-raw` if the destination should receive only generated prompts.

Another OSC-capable application can replace TD or rt-stt. Dynamic input is a string at `/transcription`, or `/chN/transcription` with channel mode. No trigger is required.

For another machine, set the sender's destination to the receiver's reachable IP and the receiver's listen interface appropriately. `127.0.0.1` is same-machine only; `0.0.0.0` binds all local interfaces and is not a destination address. Keep OSC and Ollama on trusted networks: this bridge has no authentication, UDP delivery guarantee, or encryption. Sending to a remote Ollama server sends your speech text there.

## 7. TouchDesigner integration

Set OSC In DAT to 9001. In the installed patch, the callback preserves `/transcription` subtitles and assigns `/txt2img` to the sibling `StreamDiffusionTD.par.Promptdict0concept`. Other patches must map that address to their own prompt target; merely displaying OSC packets doesn't connect them to generation.

The workstation integration changed `/project1/oscin3` and its callback live. Original callback/port/prompt were backed up to Git-ignored `td-routing-backup.json`. Save your TD project to persist live edits; integration did not overwrite the original .toe.

## 8. Logs, lifecycle, and troubleshooting

The foreground command prints logs. Appending `2>&1 | tee -a dynamic-service.log` also saves them. Logs contain transcribed speech: keep them private and Git-ignored. Follow a running log without replaying its entire history:

```bash
powershell.exe -NoProfile -Command "Get-Content 'C:/Users/Admin/git/mechanical-meanderings/dynamic-service.log' -Tail 10 -Wait"
```

Ctrl+C stops a foreground service. Closing a log follower does not stop the underlying service. Stop any previous agent before relaunching; argument changes do not reconfigure an already running process.

| Symptom | Check |
|---|---|
| No prompts | rt-stt destination 9000, agent input 9000, TD input 9001; live partials enabled; speech above gate. |
| Waiting forever during speech | Do not use `--phrase-trigger` for dynamic use. |
| Log flood | Remove `--verbose`; don't follow the old packet-heavy service.log. |
| Address ValueError in Git Bash | Prefix launch with `MSYS_NO_PATHCONV=1`. |
| Bind/OSError | Another agent or TD may own the input port. Stop the conflicting receiver. |
| Generated prompt logged but no effect | Check TD address handler, receive port, and actual image-generator prompt parameter. |
| Missing words / context drift | Page snapshots may omit text; lower history if old motifs dominate, and inspect transcription quality first. |
| Multi-mic prompts mix | Use both `--osc-per-channel` and `--channel-mode`; output is still shared by design. |
| Slow output | Measure inference elapsed time; increasing refresh rate cannot make a busy GPU infer faster. |
| argparse rejects a capture flag | Check current rt-stt `--help`; its CLI can evolve independently. |

## 9. Tests and verification scope

From this repository:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m mechanical_meanderings --text "A blue glass forest, woodcut"
```

The first command tests local HTTP fixtures, real UDP routing, history, resets, dynamic coalescing, continuous updates without triggers, output limits, and CLI behavior. The second calls real Ollama without opening an OSC receiver.

With the `tui` and `dev` extras installed, the suite also exercises the Textual headless Pilot, atomic live controls, bounded journaling, failure reporting, and shutdown. An opt-in real-model panel check is `.venv/Scripts/python.exe scripts/smoke_tui.py`: it uses the existing `gemma4:12b` on localhost:11434 and isolated loopback UDP ports assigned by the OS, sends two inputs (OSC then manual), tunes controls during inference, verifies received prompts/triggers and shutdown, and saves `sessions/tui-smoke/evidence.json`, a session JSONL, and `tui-real.svg` exported from Textual. It does not launch a microphone or TD; it does consume Ollama inference resources. The evidence and screenshot are replaced on each run; journals have unique names.

Real local Gemma -> OSC -> TD prompt-parameter read-back has been verified, including dynamic updates from live playback transcription. Generic-device and multi-microphone recipes are configurations, not claims that those hardware scenarios were all performance-tested. Prompt delivery does not establish artistic quality of rendered imagery.

## 10. Optional mediated speech and selective engagement

`--interactive` keeps the existing serial, latest-text OSC route but separates three outputs:

- `/txt2img`: a visual prompt, at most 20 words, only when the model judges the words to be visual input.
- `/mediation`: structured intent category, direct-address confidence, visual specificity, transformation intent, preservation preference, and bounded morph amount/warp controls. These are model interpretations, not facts about a participant.
- `/engagement/message`: one packet containing machine text, kind (`invitation` or `reply`), and display lifetime. Machine text does not become a diffusion prompt. This installation displays captions; it does not synthesize spoken audio.

Explicit appearance requests can drive stronger central transformation. Scene descriptions and ordinary self-narration retain more source image. Incidental or uncertain speech emits no new visual prompt or morph controls. Addressed questions can receive a brief answer without changing the image. Replies are bounded, rate-limited and suppressed if newer words have superseded the input. Recognition of intent is fallible; a shared microphone is not speaker identification.

The TD installer is `scripts/install_td_interactivity.py`. Call its `install(root, scripts, backups)` in TD's Python environment, with the actual project root and filesystem folders. It validates the inspected workstation network, backs up callbacks before editing, creates the owned `interactivity` component and caption overlay, preserves forwarded raw subtitles, disables the unused competing port-9000 TD listener, and updates the center-only shader. It does not overwrite or save the original `.toe`. Save the live TD project deliberately when satisfied.

The central compositor samples the base source at its original coordinates. Its crop mask uses unwarped coordinates, and only the generated branch is displaced. Source alpha is retained outside the crop. A compiling shader and static regression tests are not a full live, pixel-by-pixel GPU comparison; avoid force-cook/synchronous full-resolution GPU readbacks on this streaming installation.

### Gating and camera lifecycle

TD sends a small YOLO summary once per second, not a camera frame: fresh detector sequence, person count and confident object labels. Object and pose person counts are combined by maximum, not added. Tracker IDs are excluded. Repeated/frozen detector sequences cannot refresh presence.

Default invitation conditions are 20 seconds without new transcript activity, 8 seconds of sustained fresh presence, and a 90-second invitation cooldown. The budget is two invitations per occupancy, reset after 5 seconds of an empty scene. Summaries older than 3 seconds are rejected. Speech and queued visual work take priority; pause, reset, departure and renewed speech invalidate in-flight invitations. `/speech/activity` with 0/1 can provide a true VAD signal; without it, silence is inferred from transcript changes, not measured microphone audio.

Live invitations use YOLO metadata only: no `/camera/request`, JPEG or vision inference. The model receives fresh scene labels, person count, recent speech and recent invitations. Known object labels at confidence >= 0.60 become concrete transformation suggestions; e.g. a detected chair can produce `Say “floating chair” to the mic.` or `Dis «chaise en or» au micro.` Detector spelling is preserved, with a compact French display-name mapping. If the model ignores the detector, repeats itself or rambles, an authored short suggestion keeps an actually detected noun. The object list is rechecked immediately before sending, so an object that left during inference is not retained in the invitation. With no reliable object, the instruction is generic. Detector-only data is never treated as clothing, identity, ownership or a gesture.

The loopback port-9012 frame inbox remains for explicit legacy camera diagnostics, not routine invitations. TD's optional camera callback copies the cached YOLO CPU buffer and JPEG-encodes/uploads off its main thread; no new TOP readback is introduced and no camera image is saved in scene/history state.

### Workstation launch and controls

Stop the old bridge only before starting this one; leave rt-stt, Ollama and TD running. From this repository in Git Bash:

```bash
MSYS_NO_PATHCONV=1 .venv/Scripts/python.exe -u -m mechanical_meanderings \
  --interactive --listen-host 127.0.0.1 --listen-port 9000 \
  --output-host 127.0.0.1 --output-port 9001 --output-address /txt2img \
  --forward-raw --history-turns 60 --update-interval 2 --debounce 0.6 \
  --record --session-dir G:/mechanical-meanderings/runtime/sessions
```

The currently handed-off background copy, if running, already owns port 9000; do not launch a duplicate. Its private log is `G:/mechanical-meanderings/runtime/interactive-service.log`. Recorded events contain speech/settings/decisions but not JPEGs; keep them private. For a portable installation choose your own private session/log location.

At `/project1/interactivity`, `Enabled` disables invitations/replies; `Controls` disables LLM overrides of morph amount/warp; `Messagevisible` hides captions; `Confidence` adjusts detection filtering; `Fontsize` changes the caption. `Flipframe` corrects TD's bottom-up cached camera rows. The caption has a text-only shadow, not a full-frame background dimmer. `Frameport` must match `--frame-port`; `Agentport` must match the bridge input.

Timing flags are `--idle-after`, `--presence-dwell`, `--invite-cooldown`, `--scene-timeout`, `--absence-reset`, `--max-invitations`, `--reply-cooldown`, and `--message-ttl` (positive, at most 60 seconds). `--invitation-language auto` mixes English with French (one French request per three invitation requests); an explicit language locks the wording. Send `/reset` to port 9000 to clear histories, pending work, invitation reservations/budget and the machine caption.

Invitations are direct microphone instructions, 3–7 words and at most 48 characters, not poetic riddles. Object-grounded invitations explicitly suggest the words to say. Overlong, repeated or unclear model text is replaced by a complete authored instruction, not cut into a fragment. The TD caption receiver independently enforces the same size limit. Each new invitation advances one of six safe caption positions and type treatments: Cooper Black, Bauhaus, Impact, Comic Sans, Showcard and Agency, with varied size, casing, tracking, colour and deliberate line breaks. The position stays still while the invitation is visible. French accents are preserved; there is no `MIRROR /` prefix. Replies retain a quieter centred style. No new camera readback or whole-frame dimmer is added.

The YOLO-grounded workstation service writes `G:/mechanical-meanderings/runtime/interactive-service-yolo.log`; previous logs are preserved. `python -m scripts.yolo_invitation_smoke --instance <live-TD-instance-id> --output <private-json-path> --show` reads actual live detector metadata, checks that its sequence advances, generates real English/French local-model outputs and optionally displays them over OSC. It requests no image and opens no competing production listener. Without a reliable non-person object in view, this check correctly produces generic wording; synthetic object fixtures in the unit tests verify object-specific behavior separately. The older `scripts/invitation_smoke.py` is an explicit camera diagnostic; run that one only while its callback port is free.

The 2-second scheduling interval is not guaranteed model throughput. A real workstation smoke measured speech decisions around 9–25 seconds per request, including cold/context initialization, and a camera-grounded invitation around 5.5 seconds. Actual load and retained context affect latency.

### Reproducible verification

The full test suite covers mediation policy, finite controls, scheduler bounds/staleness, request correlation, real OSC transport, pause/reset/in-flight cancellation and the TD helper's thread boundary. Its synthetic model/JPEG fixtures are not evidence of real vision inference.

For an explicit real local-model check, with TD's installed camera route available and port 9012 free:

```bash
.venv/Scripts/python.exe -m scripts.interactive_smoke \
  --output G:/mechanical-meanderings/runtime/real-interactive-smoke.json --vision
```

This sends declared example utterances, requests one actual fresh cached camera frame, and records structured decisions/timings plus invitation text, never JPEG data. It must not run alongside the production bridge on the same frame port. Omitting `--vision` checks speech only. The recorded smoke successfully distinguished appearance requests, Batman self-narration, New York scene descriptions, incidental logistics and an addressed face question; it also received an actual TD JPEG and a real Gemma invitation. Visual quality, humour and audience response still require watching people use the installation.
