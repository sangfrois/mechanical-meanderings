# Conservative StreamDiffusion performance profile

Keep the visual contract: SDXL Turbo, 512×512, two denoising stages, depth ControlNet,
existing prompt/seed blending, cached attention, speech morph controls and OSC routes.
Do not switch to SD-Turbo merely to improve speed: model families require compatible
ControlNet weights and engines. An error in a different family is not a reason to
change the current working aesthetic.

## Prepared FP16 TensorRT copy

The standalone TouchDesigner copy is:

`D:/Documents/Touch/mechanical_meanderings_accelerated_fp16.toe`

It is derived from a newly saved live snapshot, including the in-memory work, not
an older disk-only project. It is not automatically loaded or launched.

Only acceleration/build settings change:

- Acceleration: `tensorrt` rather than `none`.
- TRT Profile: `quality` rather than `performance`.
- Static shapes: enabled at the existing 512×512 resolution.
- FP8: disabled; use FP16, avoiding an additional quantization/calibration change.
- Build Engines If Missing: remains enabled; Compile Engines Only remains disabled.
- Quality profile pins the allocated attention-cache axis without removing the
  existing 1–4-frame live logical-cache range.

The ControlNet input CUDA-IPC sender is gated by server readiness (and enabled
ControlNet), not by successful output streaming. This avoids waiting for the first
generated frame before making its required depth input available. IPC names and
physical-GPU selection are unchanged. The parameter callback accepts both two- and
three-argument value-change dispatch; it does not reload the diffusion extension.

On this installation, only the depth-preprocessor TensorRT engine was already built.
The operator must open the prepared copy and explicitly start StreamDiffusion to
build/load matching diffusion and ControlNet engines. The first build can take
several minutes and use substantial memory. Keep other projects saved; do not start
a competing second diffusion backend. Do not overwrite this prepared copy with the
currently open unaccelerated snapshot.

## Verification and limits

The copy is verified through expand → targeted edit → collapse → expand. Its embedded
parameters and callback source are checked again after collapse. The source model,
resolution, schedule, depth weight/preprocessor and OSC connections are preserved.
TensorRT, ONNX, graph-surgeon, Polygraphy and PEFT imports were checked in the installed
StreamDiffusion interpreter. Engines were not built and the new project was not run;
10 FPS is a target, not a measured result.

A live cache-only trial kept the two-frame ControlNet interval and temporarily set
CN Cache Decay to 1 (no added EMA smoothing). It did not establish a throughput gain
and was reverted to the original 0. A missing ControlNet IPC mapping appeared during
the observation window; that trial is not a clean latency benchmark. Retain the
original cache parameters in this conservative copy.

For a valid future FPS comparison, count generated frames over equal stable windows,
check unchanged scene/settings and live input availability, and keep the transcription
and LLM workload consistent. TouchDesigner's editor FPS is not diffusion FPS.

## Less chatter without removing image control

The interactive CLI defaults to `--reply-mode sparse`, a 15-second reply cooldown,
and at most one idle invitation after 45 seconds. Use `--no-invitations` to remove
unsolicited captions entirely. Explicit spoken silence commands are handled locally
without LLM inference; images, raw transcription and speech controls continue.
Source updates take effect when the operator restarts the agent, not in an already
running Python process. The model, history and `num_ctx` remain explicit user choices.
