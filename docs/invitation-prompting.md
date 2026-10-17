# Editing the invitation voice

The invitation's entire system message is now one UTF-8 file:

`mechanical_meanderings/prompts/invite.md`

Edit it in your usual editor and save. The updated bridge reads it before each invitation. Nothing from `PERSONALITY`, `NUDGE`, `EMBODIED` or the visual prompt is appended afterward. The same file controls first contact, continuing play, legacy short mic invitations and the explicit aside mode. A separate JSON context identifies which mode is being requested.

This requires one operator-owned relaunch to load the new Python code. After that, prompt edits do not require restarting the bridge, clearing speech history, reloading a model or changing TouchDesigner. This change does not start or stop any runtime.

## What to change

Each heading in the file controls a different part of the invitation:

- Job: what a newcomer needs to understand and do.
- Character and attitude: the snark, target of the joke, and unwanted poetic/customer-service voice.
- Language, register and profanity: Quebec French, English, tutoiement, and whether swearing belongs.
- What the participant actually controls: speech, camera input, optional movement and honest expectations.
- Context and repetition: how recent speech and previous lines should affect the next one.
- Modes and display budgets: first contact versus another invitation; how much text fits.
- Output protocol: the hidden joke-angle field versus the displayed message.
- Typography: font, placement, size and decoration.

You can rewrite any section or replace the entire file. These are editable instructions, not locked personality sliders. The default asks for basic speech-to-image onboarding with a dry, snarky machine attitude. It does not demand impossible worlds, diagnose shyness, personify an abandoned microphone, or append a compulsory gesture. It contains no sample invitation lines for the model to copy.

To keep distinct versions, copy the file and select one by adding this to your existing interactive launch command:

```bash
--invite-prompt "C:/Users/MAGen/git/mechanical-meanderings/my-invite.md"
```

Do not run another production bridge alongside the current one. A path change needs a relaunch; edits to the selected file do not. The default path is resolved from the installed package, not the working directory; the startup log prints its actual location. In an editable checkout it is the file above. An installed wheel has its own packaged copy, so choose an explicit external file if you do not want to edit an installation directory.

`--system-prompt` still controls visual interpretation, not invitations. It is deliberately independent from `--invite-prompt`.

## Inspect exactly what the model gets

From the repository:

```bash
.venv/Scripts/python.exe -m mechanical_meanderings --print-invite-prompt
```

For a variant:

```bash
.venv/Scripts/python.exe -m mechanical_meanderings --print-invite-prompt --invite-prompt my-invite.md
```

stdout is the actual system message, with only a UTF-8 BOM and outer whitespace removed. stderr prints its source and SHA-256 fingerprint. Inspection does not create an agent, contact Ollama or open OSC/camera sockets.

Each invitation also records the source and fingerprint in telemetry and an `INVITE PROMPT` log line. These let you identify the prompt revision behind a delivered caption without dumping prompt text into every log entry.

## Try wording without sending it to the installation

This is an explicit inference request, not an offline check. Get the runtime operator's permission before consuming the shared Ollama/GPU resources.

```bash
.venv/Scripts/python.exe -m mechanical_meanderings --preview-invite --invitation-language "French (Quebec)" --invite-kind nudge --invite-count 4
```

For a variant and a repeatable sampling seed:

```bash
.venv/Scripts/python.exe -m mechanical_meanderings --preview-invite --invite-prompt my-invite.md --invitation-language English --invite-kind embodied --invite-count 4 --seed 731
```

Use the operator's actual `--model` and `--ollama-url` if they differ from the normal CLI defaults. The preview requires that exact model to be already resident and uses its currently reported context length. An explicit `--num-ctx` must match it. Before every chat, residency and context are rechecked; a change aborts the batch. The check and request are not atomic, so eviction between them can still make Ollama reload. The tool does not offer an absolute no-reload guarantee.

No OSC listener, installation packet or camera request is created. The output is JSON containing every accepted or rejected sample, the production HTTP wire payload (including its effective streaming flag), the logical pre-transport payload, raw model response, typography and timings. Custom injected transports are labelled `logical` because their serialization cannot be assumed. A timeout, prompt edit/error, network failure or residency change aborts safely but still returns the accumulated samples and an explicit `terminal_error`; report `status` is `complete` or `aborted`. Exit status is 1 on abort or if any sample was rejected, rather than quietly treating a repeated line as a success. A repeated sample is retained in the report; only accepted lines enter the next sample's repetition context. `--invite-count` accepts 1–12; with `--seed`, samples increment it. The whole seed sequence is validated before any network request; an out-of-range final seed is rejected without wrapping or clamping. These timings exclude STT, installation delivery and rendering.

The preview's empty scene represents onboarding, not a live detector read. For object-grounded legacy invitations, use the existing YOLO diagnostic or tests instead of interpreting this empty-scene preview as camera evidence.

A useful refinement loop:

1. Change one section of the prompt.
2. Print the effective prompt to confirm that you selected the intended file.
3. With permission, preview first-contact and continuing-play batches in the desired language.
4. Reject wording that fails to tell a newcomer what to do, even when the joke is good.
5. Watch actual participants before calling the invitation successful.

## Read invitation rejections

A completed model response can still fail the invitation validator. These are now logged as `INVITE REJECT`, with the engagement kind, a stable reason code, whitespace-word and character counts, the intended displayed message, and timing metrics. The same details appear as `invitation_rejection` in telemetry and `rejection` on the error event, so recorded sessions and the preview retain them.

Codes distinguish `word_limit`, `character_limit`, `missing_microphone`, `repeated`, malformed JSON/content, and invalid `message`, `visible_detail` or `approach` fields. Generated candidates longer than 512 characters are explicitly marked truncated in diagnostics; that abbreviation is never sent as a fallback caption. Unparseable response bodies and arbitrary provider errors are not copied into these logs. Candidate logging can include a model callback to recent participant speech, so keep these logs private as you already would for captions.

A generic `ValueError` from an older log cannot reveal which condition failed or reconstruct text that was not recorded. Token counts alone are not a rejection reason. `Discarded superseded invitation` is a freshness cancellation, separate from output validation. New diagnostic Python code requires an operator-owned relaunch; editing the selected prompt alone still does not.

## Limits that are not prompt wording

The existing runtime/display contract remains in force:

- `nudge` and `embodied` are displayed as `challenge`: 3–24 whitespace-separated words, at most 160 characters, and a mic/micro/microphone mention is required. The default writing target is 10–20 words. Overlong output is rejected, not silently sliced.
- Legacy `invitation`: 3–7 words and at most 48 characters. The existing action/object validator can replace invalid output with a complete mic instruction; this older lane cannot preserve arbitrary long snark. The normal Asides-enabled route uses `challenge`, not this fallback lane.
- `aside`: at most 16 words / 120 characters; existing parsing remains unchanged. The current live Asides control selects first-contact/continuing-play challenges, not this diagnostic aside mode.
- The JSON response schema and safe typography bounds are enforced in Python and the TD adapter. Editing their description in the prompt does not change that schema.
- The repetition guard rejects exact/cosmetic rewrites and reused punchlines. When both lines contain a separable, explicit mic-instruction clause and another nonempty clause, it compares the remainder rather than treating the required onboarding words as the joke. Different jokes may reuse the same practical instruction. This is a lexical heuristic, not proof of humour, semantic novelty or absence of poetry.
- `--invitation-language`, `--idle-after`, `--presence-dwell`, `--invite-cooldown`, `--max-invitations`, `--message-ttl`, `--no-invitations`, `--reply-mode` and sampling settings remain separate operator controls. None was silently retuned here.

A missing, empty, invalid-UTF-8 or oversized prompt fails visibly and skips the invitation; it never secretly restores a different voice. A corrected file is read on the next eligible attempt. An edit observed during generation or the pre-send check discards the old draft. This is not an atomic transaction with an editor save at the exact instant UDP is sent. Prompt file IO stays outside the raw-transcription receive lock. Speech context and visual prompts are unaffected by invitation edits.

Automated tests use explicitly synthetic model wording and real isolated HTTP/OSC transport. They verify the controls and separation of lanes, not whether the local model is funny. The opt-in real-model preview and participant observation are separate checks.
