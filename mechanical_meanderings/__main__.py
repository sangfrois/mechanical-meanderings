"""Run with python -m mechanical_meanderings."""
import argparse
import json
from importlib.resources import files
import logging
from pathlib import Path
import sys
import time
from . import Agent, GenerationControl
from .bridge import Bridge
from .live import Settings, Session


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mechanical Meanderings OSC → Ollama → OSC; default noninteractive scheduling waits for rt-stt phrase closure')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--text', help='One-shot input; print prompt without opening OSC sockets')
    mode.add_argument('--tui', action='store_true', help='Live Textual control panel; owns the OSC bridge')
    mode.add_argument('--print-invite-prompt', action='store_true',
        help='Print the exact invitation system prompt and exit; no model request or sockets')
    mode.add_argument('--preview-invite', action='store_true',
        help='Opt-in invitation wording/timing JSON from the already resident model; no OSC or camera')
    parser.add_argument('--record', action='store_true', help='Also record headless service events to JSONL')
    parser.add_argument('--session-dir', type=Path, default=Path('sessions'))
    for option in ('temperature', 'top-p', 'repeat-penalty'):
        parser.add_argument('--' + option, type=float, default=None, help='Ollama override; omitted = model default')
    for option in ('num-ctx', 'top-k', 'seed', 'num-predict'):
        parser.add_argument('--' + option, type=int, default=None, help='Ollama override; omitted = model default')
    parser.add_argument('--system-prompt', type=Path, help='Override the packaged visual reference (system.txt in legacy mode)')
    parser.add_argument('--invite-prompt', type=Path,
        help='FULL invitation system prompt replacement; UTF-8 file reloaded on every invitation')
    parser.add_argument('--invite-kind', choices=('nudge', 'embodied', 'invitation', 'aside'), default='nudge',
        help='Preview mode only; nudge = first contact, embodied = continuing play')
    parser.add_argument('--invite-count', type=int, default=1,
        help='Preview mode only: generate 1..12 samples with repetition context, keeping rejected outputs')
    parser.add_argument('--model', default='gemma4:12b')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--generation-budget', type=float, default=None,
        help='Whole realtime request budget in seconds (default: --timeout); no effect on explicit legacy dynamic/FIFO or one-shot input')
    parser.add_argument('--history-turns', type=int, default=3)
    parser.add_argument('--reset-after', type=float, default=None)
    parser.add_argument('--listen-host', default='localhost')
    parser.add_argument('--listen-port', type=int, default=9001)
    parser.add_argument('--output-host', default='localhost')
    parser.add_argument('--output-port', type=int, default=9000)
    parser.add_argument('--output-address', default='/prompt')
    parser.add_argument('--queue-size', type=int, default=8)
    parser.add_argument('--debounce', type=float, default=0.6, help='Legacy dynamic/interactive only: generate after this pause in changing text')
    parser.add_argument('--update-interval', type=float, default=2.0, help='Dynamic refresh interval during continuous speech; inference speed also limits cadence')
    scheduling = parser.add_mutually_exclusive_group()
    scheduling.add_argument('--dynamic', action='store_true', help='Legacy opt-in: generate during speech using text debounce/refresh (not VAD gated)')
    scheduling.add_argument('--phrase-trigger', action='store_true', help='Legacy FIFO: wait for /trigger 1; default is latest-only boundary mode')
    scheduling.add_argument('--final-turn', action='store_true',
        help='Noninteractive latest-only completed speech: /final TEXT or text then /trigger 1; partials never generate')
    parser.add_argument('--speech-history', type=int, default=128,
        help='Boundary/final/interactive speech ledger size, independent of generated --history-turns (default 128)')
    parser.add_argument('--verbose', action='store_true', help='Log every incoming OSC packet; normally disabled')
    parser.add_argument('--channel-mode', action='store_true', help='Only /chN/transcription and /chN/trigger (also /chN/final in final-turn mode); ignore aggregate duplicates')
    parser.add_argument('--forward-raw', action='store_true', help='Forward raw transcription/trigger immediately; always enabled in default boundary mode')
    parser.add_argument('--interactive', action='store_true', help='Mediate intent/identity and enable bounded YOLO-gated invitations')
    parser.add_argument('--reply-mode', choices=('sparse', 'auto', 'off'), default='sparse',
        help='Sparse: answer questions/conversation, not greetings or visual updates; off: no captions')
    parser.add_argument('--no-invitations', action='store_true', help='Disable unsolicited invitations; keep speech/visual controls')
    parser.add_argument('--idle-after', type=float, default=45., help='Quiet transcript/VAD seconds before inviting')
    parser.add_argument('--presence-dwell', type=float, default=8., help='Fresh sustained person presence before inviting')
    parser.add_argument('--invite-cooldown', type=float, default=180.)
    parser.add_argument('--scene-timeout', type=float, default=3., help='Reject stale YOLO summaries')
    parser.add_argument('--absence-reset', type=float, default=5., help='Sustained empty scene resets occupancy invitation budget')
    parser.add_argument('--max-invitations', type=int, default=1, help='Maximum invitations per occupancy')
    parser.add_argument('--frame-port', type=int, default=9012, help='Loopback-only temporary JPEG callback port')
    parser.add_argument('--frame-timeout', type=float, default=3.)
    parser.add_argument('--camera-mode', choices=('auto', 'off'), default='auto',
        help='Auto: one model-requested frame for addressed visual questions; off: text/YOLO only')
    parser.add_argument('--camera-cooldown', type=float, default=20.,
        help='Minimum seconds between speech camera attempts, including timeouts')
    parser.add_argument('--reply-cooldown', type=float, default=15.)
    parser.add_argument('--message-ttl', type=float, default=12.)
    parser.add_argument('--invitation-language', default='auto', help='Auto: Quebec French challenges; legacy mic mode mixes English/French; explicit language locks invitations')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s: %(message)s',
                        handlers=[logging.NullHandler()] if args.tui else None)
    bridge = None
    session = None
    try:
        if args.print_invite_prompt:
            from .invitation import InvitationPrompt
            snapshot = InvitationPrompt(args.invite_prompt).read()
            print(snapshot.text)
            print(f'Invitation prompt: {snapshot.source}; sha256={snapshot.sha256}', file=sys.stderr)
            return 0
        if args.invite_prompt is not None and not (args.interactive or args.preview_invite):
            raise ValueError('--invite-prompt requires --interactive, --preview-invite or --print-invite-prompt')
        if not 1 <= args.invite_count <= 12:
            raise ValueError('invite-count must be between 1 and 12')
        if not args.preview_invite and (args.invite_count != 1 or args.invite_kind != 'nudge'):
            raise ValueError('--invite-count and --invite-kind are preview-only options')
        if args.timeout <= 0 or args.queue_size < 1:
            raise ValueError('timeout and queue-size must be positive')
        if args.generation_budget is None:
            args.generation_budget = args.timeout
        GenerationControl(args.generation_budget)
        if args.final_turn and (args.interactive or args.text is not None or args.preview_invite):
            raise ValueError('--final-turn is for the noninteractive OSC bridge, not --interactive or --text')
        if args.speech_history < 1:
            raise ValueError('speech-history must be positive')
        settings = Settings(temperature=args.temperature, num_ctx=args.num_ctx,
                            top_p=args.top_p, top_k=args.top_k, repeat_penalty=args.repeat_penalty,
                            seed=args.seed, num_predict=args.num_predict, history=args.history_turns,
                            debounce=args.debounce, interval=args.update_interval)
        if args.preview_invite:
            from .mediation import MediatingAgent
            from .invite_preview import preview_invitations
            # Invitations own their full prompt; no visual prompt or bridge state is needed.
            agent = MediatingAgent(system_prompt='', model=args.model, ollama_url=args.ollama_url,
                timeout=args.timeout, history_turns=args.history_turns, reset_after=args.reset_after,
                reply_mode=args.reply_mode, invitation_prompt=args.invite_prompt)
            result = preview_invitations(agent, settings, kind=args.invite_kind,
                language=args.invitation_language, count=args.invite_count, budget=args.generation_budget)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
            return 0 if result['status'] == 'complete' and all(
                sample['accepted'] for sample in result['samples']) else 1
        source = args.system_prompt or files('mechanical_meanderings').joinpath('system.txt')
        system = source.read_text(encoding='utf-8').strip()
        if not system:
            raise ValueError('system prompt is empty')
        agent_class = Agent
        bridge_class = Bridge
        interactive_options = {}
        agent_options = {}
        boundary_mode = not (args.dynamic or args.phrase_trigger or args.final_turn)
        if boundary_mode:
            from .boundary import BoundaryBridge
            bridge_class = BoundaryBridge
            interactive_options = dict(speech_history=args.speech_history, generation_budget=args.generation_budget)
        if args.final_turn:
            from .final_turn import FinalTurnBridge
            bridge_class = FinalTurnBridge
            interactive_options = dict(speech_history=args.speech_history, generation_budget=args.generation_budget)
        if args.interactive:
            from .mediation import MediatingAgent
            from .interactive import InteractiveBridge
            from .engagement import EngagementState
            from .personality import INTERACTIVE_REFERENCE
            if args.system_prompt is None:
                # The legacy system.txt explicitly says never answer and always turn fragments
                # into visuals. Don't make the interactive model fight that conflicting contract.
                system = INTERACTIVE_REFERENCE
            agent_class, bridge_class = MediatingAgent, InteractiveBridge
            agent_options = dict(reply_mode=args.reply_mode, invitation_prompt=args.invite_prompt)
            state = EngagementState(idle_after=args.idle_after, presence_dwell=args.presence_dwell,
                invite_cooldown=args.invite_cooldown, scene_timeout=args.scene_timeout,
                absence_reset=args.absence_reset, max_invitations=args.max_invitations)
            interactive_options = dict(engagement=state, frame_port=args.frame_port,
                frame_timeout=args.frame_timeout, reply_cooldown=args.reply_cooldown,
                message_ttl=args.message_ttl, invitation_language=args.invitation_language,
                camera_enabled=args.camera_mode == 'auto', camera_cooldown=args.camera_cooldown,
                invitations_enabled=not args.no_invitations,
                boundary_speech=not (args.dynamic or args.phrase_trigger),
                speech_history=args.speech_history, generation_budget=args.generation_budget)
        agent = agent_class(system_prompt=system, model=args.model, ollama_url=args.ollama_url,
                      timeout=args.timeout, history_turns=args.history_turns, reset_after=args.reset_after, **agent_options)
        if args.interactive:
            invite_snapshot = agent.invitation_prompt.read()  # Validate before any bridge socket opens.
        if args.text is not None:
            if not args.text.strip():
                raise ValueError('text must not be empty')
            if args.interactive:
                print(json.dumps(agent.interpret(args.text.strip(), settings=settings).metadata(),
                    ensure_ascii=False, allow_nan=False))
            else:
                print(agent.generate(args.text.strip(), settings=settings))
            return 0
        bridge = bridge_class(agent, listen_host=args.listen_host, listen_port=args.listen_port,
                        output_host=args.output_host, output_port=args.output_port,
                        output_address=args.output_address, queue_size=args.queue_size,
                        channel_mode=args.channel_mode, forward_raw=args.forward_raw or boundary_mode,
                        dynamic=not args.phrase_trigger, debounce=args.debounce,
                        update_interval=args.update_interval, **interactive_options)
        bridge.configure(**settings.snapshot())
        if args.tui or args.record:
            session = Session(args.session_dir)
            bridge.on_event = session.record
        if args.tui:
            try:
                from .tui import MeanderingsApp
            except ImportError as exc:
                raise RuntimeError('Install the tui extra: uv sync --extra tui') from exc
            MeanderingsApp(bridge, session).run()
            return 0
        bridge.start()
        logging.info('Listening on %s:%s; output %s:%s%s', *bridge.address,
                     args.output_host, args.output_port, args.output_address)
        if boundary_mode:
            logging.info('Mode: boundary latest-completed; no timer-generated turns; raw=%s; speech-history=%d',
                         bridge.forward_raw, args.speech_history)
            logging.info('Realtime whole-request budget: %.2fs; transcription is a separate immediate lane',
                         args.generation_budget)
        else:
            logging.info('Mode: %s; interval=%.2fs debounce=%.2fs',
                         'final-turn latest-completed' if args.final_turn else
                         'phrase-trigger' if args.phrase_trigger else 'dynamic latest-text',
                         args.update_interval, args.debounce)
        if args.interactive:
            logging.info('Interaction voice: concrete-QC; replies=%s cooldown=%.1fs; invitations=%s; context=%s; history=%d turns',
                         args.reply_mode, args.reply_cooldown, not args.no_invitations,
                         args.num_ctx if args.num_ctx is not None else 'model default', args.history_turns)
            logging.info('Invitation prompt: %s; sha256=%s; reloaded for each invitation; full replacement',
                         invite_snapshot.source, invite_snapshot.sha256)
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        # Do not echo response bodies or reasoning into logs.
        print(f'ERROR: {type(exc).__name__}; no prompt emitted', file=sys.stderr)
        if isinstance(exc, (ValueError, RuntimeError)):
            print(str(exc), file=sys.stderr)
        return 1
    finally:
        if bridge is not None:
            bridge.close()
        if session is not None:
            session.close()


if __name__ == '__main__':
    raise SystemExit(main())
