# Creative direction and latency verification

## Scope and runtime ownership

Changed the Python interactive path in `C:/Users/MAGen/git/mechanical-meanderings`; preserved the existing dirty working tree and left TouchDesigner source/runtime untouched. No installation OSC traffic, camera capture, model switch, context reduction, or runtime restart was performed. The operator explicitly authorized isolated requests to the already-resident gemma4:12b.

The reported launch selects boundary/latest-completed scheduling. `--debounce .6` and `--update-interval 3` do not add those waits to completed-phrase mediation. Microphone endpointing, STT decoding and image-renderer latency are outside the measured model-request times below.

## Changes

- Interactive visual prompts target 40–60 words, aiming for 50, rather than the previous 20-word instruction and parser truncation. On overshoot, the parser prefers a complete sentence of at least 40 words within the cap.
- Explicit novelty/repetition complaints request a new subject and representational direction. Rejected or superseded speech is not a positive keyword bag. The instructions distinguish global novelty, local edits and recall without a fixed catalogue of visual examples.
- Four accepted render briefs supply explicitly generated assistant context immediately before the latest untouched speech turn. They are not substituted for chronological speech. Only accepted/delivered visuals enter this context in boundary mode; reset/epoch boundaries clear it.
- Invitations describe visually evocative speech into the microphone, optionally paired with gentle motion/framing. After-speech invitations no longer forbid mentioning the mic or force a face/body experiment. The active challenge instructions contain no sample invitation lines or mandated image subjects.
- The model invents a short comic premise, with twelve accepted recent lines/premises as anti-repetition context. Cosmetic rewrites are suppressed without another blocking inference request. This is not a guarantee of semantic originality.
- Silent decisions use `caption: null` in the model schema. Visible captions retain typography; the parser adapts both to the unchanged `reply`/`typography` OSC metadata. This avoids generating invisible design data when the model follows the silent-caption policy.
- Interactive INFO logs now include queue, HTTP, load, input/output evaluation, token counts and the effective output cap. Camera refinement remains separate: top-level model counters describe the text pass, `vision_pass` holds refinement counters, and HTTP time spans requests.

## Real-model replay

Evidence: `creative-direction-before.json` and `creative-direction-accepted.json` in this directory. Each contains six sequential speech requests and four invitation attempts. Both use gemma4:12b, the already-loaded 32768-token context, history=25 and matched speech seed 731 / invitation seeds 731–734. Sampling defaults remain unchanged. These A/B calls were nonstreaming isolated HTTP; they are a small local replay, not a controlled hardware benchmark or end-to-end installation rehearsal.

| Measurement | Before | Revised |
|---|---:|---:|
| Median speech request (6 requests) | 4.969 s | 4.051 s |
| Median speech output tokens | 274.5 | 174 |
| Visual prompt word counts | 17, 17, 16, 15, 17, 18 | 53, 50, 56, 56, 51, 50 |
| Median invitation attempt (4 attempts, including suppression) | 4.661 s | 4.153 s |

Observed speech-request median reduction: 18.47%, despite much longer image descriptions. Five of six revised speech responses used null captions. One still generated a full caption later suppressed by the sparse policy, so the optimization does not shorten every turn. The invitation median includes one rejected near-repeat and is not four delivered invitations.

Behavior observed and programmatically checked:

- The baseline's first explicit request for a completely new visual returned another cubist disco forest.
- The revised first novelty request left that subject/style for a cavern. Subsequent prompts also dropped cubism/disco/kaleidoscopic vocabulary.
- All six revised descriptions contained 40–60 words, specifically 50–56 in this replay.
- The final local-edit request kept the immediately preceding underwater-city subject while changing surface texture and lighting.
- Three revised invitation attempts were accepted. One cosmetic repeat was rejected by the guard instead of published. The model still has stylistic habits; this result supports improved direction and a working suppression mechanism, not unlimited diversity or consistently excellent jokes.
- Generated image pixels were not inspected. Prompt compliance does not prove StreamDiffusion will remove human silhouettes or render a specified scene faithfully.

A separate real bounded-streaming request verified that the compact caption schema still produces a visible answer without a visual update: “Your voice feeds the prompt; your body shapes the pixels.” It completed in 6.570 s with `apply_visual=false`, an empty image prompt, one pass and no camera attachment. See `creative-caption-check.json`. This is a compatibility check, not an additional speedup sample.

## Deterministic tests

Before source edits: 313 passed, 5 failed in the full suite.

After final source edits:

- Combined affected-path suite: 197 passed in 69.57 s.
- Full suite: 331 passed, 5 failed in 118.58 s.
- The same five failures remain in `tests/test_td_speech_controls.py`, concerning existing TD step-target bounds. Those controls were not changed.
- New tests cover 50-word UDP delivery, null and visible captions, bounded accepted-only render memory, verbatim speech preservation, cancellation/reset, example-free invitation requests, twelve-line anti-repeat context, cosmetic duplicate suppression, and detailed timing logs.
- A Windows HTTP-fixture cancellation race was corrected to recognize connection abort/reset as disconnect without treating a timeout as success.
- Relevant source diff whitespace checks and scoped dynamic-execution/shell-injection/unsafe-deserialization scans passed.

## Deployment and limitations

Source changes are not hot-loaded into an already-running installation process. The operator owns the restart; the existing command can be reused. No CLI flags, model/context choices or speech-memory sizes were silently changed.

Runtime language remains probabilistic. Near-repeat suppression can leave a quiet interval rather than forcing a bad invitation. The legacy short-object invitation mode with Experience/Asides disabled retains its older wire limits. The 20-word noninteractive visual-only path is unchanged.
