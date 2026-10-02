"""Run with python -m mechanical_meanderings."""
import argparse
import json
from importlib.resources import files
import logging
from pathlib import Path
import sys
import time
from . import Agent
from .bridge import Bridge
from .live import Settings, Session


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mechanical Meanderings OSC → Ollama → OSC')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--text', help='One-shot input; print prompt without opening OSC sockets')
    mode.add_argument('--tui', action='store_true', help='Live Textual control panel; owns the OSC bridge')
    parser.add_argument('--record', action='store_true', help='Also record headless service events to JSONL')
    parser.add_argument('--session-dir', type=Path, default=Path('sessions'))
    for option in ('temperature', 'top-p', 'repeat-penalty'):
        parser.add_argument('--' + option, type=float, default=None, help='Ollama override; omitted = model default')
    for option in ('num-ctx', 'top-k', 'seed', 'num-predict'):
        parser.add_argument('--' + option, type=int, default=None, help='Ollama override; omitted = model default')
    parser.add_argument('--system-prompt', type=Path, help='Override packaged system.txt')
    parser.add_argument('--model', default='gemma4:12b')
    parser.add_argument('--ollama-url', default='http://localhost:11434')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--history-turns', type=int, default=3)
    parser.add_argument('--reset-after', type=float, default=None)
    parser.add_argument('--listen-host', default='localhost')
    parser.add_argument('--listen-port', type=int, default=9001)
    parser.add_argument('--output-host', default='localhost')
    parser.add_argument('--output-port', type=int, default=9000)
    parser.add_argument('--output-address', default='/prompt')
    parser.add_argument('--queue-size', type=int, default=8)
    parser.add_argument('--debounce', type=float, default=0.6, help='Generate after this pause in changing text')
    parser.add_argument('--update-interval', type=float, default=2.0, help='Dynamic refresh interval during continuous speech; inference speed also limits cadence')
    parser.add_argument('--phrase-trigger', action='store_true', help='Legacy mode: wait for /trigger 1 instead of dynamic updates')
    parser.add_argument('--verbose', action='store_true', help='Log every incoming OSC packet; normally disabled')
    parser.add_argument('--channel-mode', action='store_true', help='Only /chN/transcription and /chN/trigger; ignore aggregate duplicates')
    parser.add_argument('--forward-raw', action='store_true', help='Forward accepted raw input as /transcription and /trigger')
    parser.add_argument('--interactive', action='store_true', help='Mediate intent/identity and enable bounded YOLO-gated invitations')
    parser.add_argument('--idle-after', type=float, default=20., help='Quiet transcript/VAD seconds before inviting')
    parser.add_argument('--presence-dwell', type=float, default=8., help='Fresh sustained person presence before inviting')
    parser.add_argument('--invite-cooldown', type=float, default=90.)
    parser.add_argument('--scene-timeout', type=float, default=3., help='Reject stale YOLO summaries')
    parser.add_argument('--absence-reset', type=float, default=5., help='Sustained empty scene resets occupancy invitation budget')
    parser.add_argument('--max-invitations', type=int, default=2, help='Maximum invitations per occupancy')
    parser.add_argument('--frame-port', type=int, default=9012, help='Loopback-only temporary JPEG callback port')
    parser.add_argument('--frame-timeout', type=float, default=3.)
    parser.add_argument('--reply-cooldown', type=float, default=12.)
    parser.add_argument('--message-ttl', type=float, default=12.)
    parser.add_argument('--invitation-language', default='auto', help='English/French mix when auto; explicit language locks invitations')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s: %(message)s',
                        handlers=[logging.NullHandler()] if args.tui else None)
    bridge = None
    session = None
    try:
        if args.timeout <= 0 or args.queue_size < 1:
            raise ValueError('timeout and queue-size must be positive')
        settings = Settings(temperature=args.temperature, num_ctx=args.num_ctx,
                            top_p=args.top_p, top_k=args.top_k, repeat_penalty=args.repeat_penalty,
                            seed=args.seed, num_predict=args.num_predict, history=args.history_turns,
                            debounce=args.debounce, interval=args.update_interval)
        source = args.system_prompt or files('mechanical_meanderings').joinpath('system.txt')
        system = source.read_text(encoding='utf-8').strip()
        if not system:
            raise ValueError('system prompt is empty')
        agent_class = Agent
        bridge_class = Bridge
        interactive_options = {}
        if args.interactive:
            from .mediation import MediatingAgent
            from .interactive import InteractiveBridge
            from .engagement import EngagementState
            agent_class, bridge_class = MediatingAgent, InteractiveBridge
            state = EngagementState(idle_after=args.idle_after, presence_dwell=args.presence_dwell,
                invite_cooldown=args.invite_cooldown, scene_timeout=args.scene_timeout,
                absence_reset=args.absence_reset, max_invitations=args.max_invitations)
            interactive_options = dict(engagement=state, frame_port=args.frame_port,
                frame_timeout=args.frame_timeout, reply_cooldown=args.reply_cooldown,
                message_ttl=args.message_ttl, invitation_language=args.invitation_language)
        agent = agent_class(system_prompt=system, model=args.model, ollama_url=args.ollama_url,
                      timeout=args.timeout, history_turns=args.history_turns, reset_after=args.reset_after)
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
                        channel_mode=args.channel_mode, forward_raw=args.forward_raw,
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
        logging.info('Mode: %s; interval=%.2fs debounce=%.2fs', 'phrase-trigger' if args.phrase_trigger else 'dynamic latest-text', args.update_interval, args.debounce)
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
