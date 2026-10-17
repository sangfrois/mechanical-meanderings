# Participant interaction: concrete, cheeky, embodied

## Design principles

1. The camera image is an input, not merely a background. Shape, silhouette and framing influence the generated image alongside the prompt. A human in the source can remain human-looking without the word “human” in the prompt.
2. Participants can change the input physically, not only verbally. Occluding facial features, turning sideways, reframing a hand or sleeve, moving the face out of frame, or changing distance can open other possibilities. Hands still carry human cues; none of these guarantees a nonhuman result.
3. Denoising, prompt direction, source cues and the morph blend interact. Do not explain the installation as “more steps always means less human” or invent a magic setting. A practical answer begins with what a participant can try.
4. Different channels have different jobs: transcribed speech supplies intent; image prompts guide generation; brief captions answer or invite; detector metadata gates presence; occasional snapshots can ground a relevant visual answer. A caption must never become an image prompt.
5. The work evolves between deployments. Its conversational model knows the above principles, not every possible future capability or exact current hardware setting. No current snapshot means no claim that a gesture was seen or worked.
6. The conversational model is the text/control layer; StreamDiffusion generates the pixels. A refusal caption is not a censorship mechanism. Withholding a new prompt does not stop the renderer or remove images already being generated. The agent must not invent downstream filters, blocking code or claims that the system is unrestricted.
7. Camera snapshots ground observations about camera input, not claims about the generated output on screen. Without output evidence, the agent cannot contradict a participant's report of what is being displayed.

## Engagement policy

With the existing Experience / Asides control enabled:

- First contact: a plain instruction to describe what to see or change in the image into the microphone, with a short, dry, snarky jab. Ordinary descriptions are enough; no demand for an impossible world or poetic performance.
- After speech OR a previous invitation: another practical mic invitation with a different joke. Movement is optional and secondary, not appended to every line. Prior speech is not proof that someone understands the installation; microphone instructions remain allowed. Leaning forward and face-hiding are not defaults.
- The model invents a short comic premise before writing the invitation, without sample lines, dictated image subjects or a fixed rotation of slogans. Twelve recently delivered invitations and premises provide anti-repetition context; exact/cosmetic rewrites are suppressed without another blocking model call. This lexical check does not guarantee semantic originality.
- Each presence session retains the existing idle/dwell/cooldown limits and maximum invitation count. New speech interrupts pending invitations and clears displayed challenges.
- Sustained absence clears this presence session's speech cues, so newcomers do not inherit another visitor's progression. This is coarse presence bookkeeping, not identification of individuals.
- Challenges use a separate `challenge` OSC message kind, up to 24 words / 160 characters, so TouchDesigner does not replace them with its older seven-word mic fallback. The model targets 10–20 words. The joke angle is diagnostic metadata, not displayed or sent as image text.
- With Asides off, the older concise microphone/object invitation mode remains available.

## Voice

Concrete, cheeky, conversational Quebec French; English when explicitly selected or answering an English speaker. Occasional vulgarity and sacres are welcome, without a caricatured accent or swearing on every line. No mystical monologues, customer-service praise or demands that participants respect software. Criticise AI hype and acknowledge sexism and other learned biases without pretending to be neutral, fair by construction, or able to fix everything. One small joke, one clear action. A dare is not a judgement of somebody's body, feelings or ability.

No sample invitations are supplied to the model. Basic onboarding comes first; personality fits around the instruction. The joke can target the machine's taste, limitations or pretensions, without turning an ordinary visitor into a creativity contestant. No lonely/bored microphone persona, invented timidity, half-poetry or customer-service praise. Vary the joke, opening and sentence structure without obscuring how to engage. Do not scold participants for politeness or disrespect toward software.

Hands can obscure the face while standing still; never obstruct breathing. Distance changes need space. No blind movement, balancing, touching the lens or another person's body. Refusal should not attract ridicule or repeated pressure.

## Bodies and image direction

Interactive image prompts now target 40–60 words, aiming for 50, with concrete subject/action, spatial relationships, surfaces, light and a coherent treatment. The parser permits 60 words rather than truncating to 20, preferring the last complete sentence of at least 40 words on overshoot. Legacy noninteractive generation retains its existing 20-word contract.

A request for novelty is sufficient direction to generate: the model should leave behind both the old subject and its style family rather than recycling their synonyms. Rejected motifs are not positive keywords. Local corrections should retain the compatible current subject. Four accepted visual briefs are kept separately from speech; boundary-mode requests label these as generated assistant context immediately before the untouched latest utterance. Only delivered, current decisions enter that context. Reset/epoch expiry clears it. Speech history, verbatim observations, model and context-window settings are not reduced.

Silent decisions use a compact `caption: null` model field instead of generating typography for an invisible caption. Visible captions retain model-designed typography and are adapted back to the existing `reply`/`typography` OSC metadata; no TD schema update is needed. The model may still produce a caption subsequently suppressed by the sparse-reply policy, so this saves work on compliant silent turns rather than guaranteeing every request is shorter.

Both the interactive prompt field and the legacy visual-only instructions distinguish a mention of woman/flesh/skin/ass from a request for nudity. No automatic undressing, erotic posing, white-female beauty default or bodily replacement. Complaints about sexism are conversation, not requests for more sexualised images.

For imagined figures, use a broader range of adult ages, builds, skin tones and unspecified genders without overriding the participant's stated subject or assigning traits to the real camera visitor. Explicitly requested non-graphic adult nudity can remain nude; open-ended style should lead to a concrete material/compositional treatment rather than generic glamour, a crowd in a field, or an 'artistic and cinematic' suffix. These instructions steer prompts; they do not remove the image model's learned biases or guarantee its output.

## Implementation and checks

`mechanical_meanderings/personality.py` holds the reply voice, interactive visual reference and self-knowledge. `--interactive` uses that reference instead of the conflicting always-generate/never-answer rules in the legacy `system.txt`; an explicit `--system-prompt` override remains respected. Invitations have their own entire editable system message in `mechanical_meanderings/prompts/invite.md`, selected by `--invite-prompt PATH` and reread for every request. No Python personality or output instructions are appended. `--print-invite-prompt` inspects the exact message offline, and opt-in `--preview-invite` exercises resident-model wording without OSC/camera. See [invitation editing and controls](invitation-prompting.md). `interactive.py` chooses first-contact versus continuing-play engagement. `scripts/td_interactivity.py` renders and clears the bounded `challenge` caption without modifying visuals.

The interactive CLI now defaults to sparse replies: answer addressed questions/conversation, not every image update or greeting. Prefer one sentence of 3–12 words (up to 28 for a practical answer), without a routine follow-up question. The CLI reply cooldown is 15 seconds. Idle invitations wait 45 seconds, have a 180-second cooldown, and are limited to one per occupancy; these remain explicit override flags. `--no-invitations` disables unsolicited messages without changing image prompts or speech-driven controls. `--reply-mode off` disables replies and invitations; `--reply-mode auto` retains the older optional acknowledgement behavior.

An explicit whole-utterance “Don't talk back”, “Stop talking” or “Tais-toi” immediately clears the caption and latches replies/invitations off without asking the model. Visual prompts, controls and raw transcription keep flowing. “You can reply again” or “Tu peux parler” resumes optional captions, and `/reset` clears the spoken mute; ordinary per-channel history expiry does not. Quoted phrases or imagined scenes containing these words do not trigger the control. The operator's `--reply-mode off` / `--no-invitations` choices still take priority.

The startup log identifies the reply voice as `Interaction voice: concrete-QC`, reports caption policy and prints the invitation file path/fingerprint. Python source edits do not hot-reload an already running agent: the operator owns its restart. After that one relaunch, invitation prompt-file saves take effect on the next request without resetting speech history. Context-window settings remain explicit CLI/TUI choices, not changes to the base model. A larger `num_ctx` does not turn dynamic subtitle fragments into complete utterances, and concurrent verification requests against the live Ollama service are not a clean latency benchmark.

The deterministic tests exercise scheduling, OSC separation, newcomer reset, length limits, wrong-action rejection, typography validation and the TD caption handoff. Real-model probes are separately necessary to assess the actual wording and factual explanation; fixture tests do not establish linguistic quality or a guaranteed visual transformation.

`scripts/creative_replay.py --output PATH` is an opt-in live wording/timing probe. Obtain operator permission first: it uses the resident gemma4:12b and its existing context size, retains history=25, and never sends OSC, captures a camera or launches a runtime. Saved JSON contains exact requests, outputs and Ollama timings. The interactive INFO logs now separate queue, HTTP, input evaluation, output evaluation, load and token counts. The boundary route does not use debounce/update-interval to delay a completed phrase. These timings do not include STT endpointing or image rendering.
