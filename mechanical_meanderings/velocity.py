"""High-velocity rt-stt JSONL -> Ollama -> OSC prompt engine.

This is deliberately separate from the legacy subtitle/trigger bridge. rt-stt's
JSONL records are authoritative completed utterances; subtitle OSC packets are
presentation snapshots and can arrive in an ambiguous order around phrase
closure. Tailing JSONL also lets an already-running rt-stt continue sending its
subtitles directly to TouchDesigner while this process emits only image prompts.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import re
import http.client
import socket
import threading
import time
from typing import Callable, Iterable
from urllib.parse import urlsplit, urlunsplit

from pythonosc.udp_client import SimpleUDPClient

log = logging.getLogger("mechanical_meanderings.velocity")

PROMPT_PATH = Path(__file__).with_name("prompts") / "velocity.txt2img.md"
DEFAULT_TRANSCRIPT = Path(__file__).resolve().parents[2] / "rt-stt" / "transcript.jsonl"
TERMINAL_RE = re.compile(r"[.!?…][\"'’”»)]*$")
CONTINUATION_RE = re.compile(r"[,;:—-][\"'’”»)]*$")
LABEL_RE = re.compile(r"^\s*(?:text\s*[- ]?to\s*[- ]?image|visual|image)?\s*prompt\s*:\s*", re.I)
THINK_RE = re.compile(r"<think\b[^>]*>.*?(?:</think\s*>|$)", re.I | re.S)
MAX_TEXT = 8192
MAX_PENDING_CHARS = 32768
MAX_CHANNELS = 64
MAX_JSONL_LINE = 65536
MAX_HTTP_RESPONSE = 65536
MAX_PROMPT_CHARS = 2048


class PromptDeliveryError(OSError):
    def __init__(self, message: str, prompt_sent: bool = False):
        super().__init__(message)
        self.prompt_sent = prompt_sent
        self.accepted = False


def normalize_prompt(value: str, max_words: int = 20) -> str:
    """Enforce the renderer contract even when a small model ignores it."""
    if not isinstance(value, str):
        raise ValueError("Ollama returned no text")
    value = THINK_RE.sub("", value)
    value = re.sub(r"^\s*```[^\n]*\n|\n?```\s*$", "", value)
    value = LABEL_RE.sub("", value)
    value = " ".join(value.strip().strip("\"'`“”").split())
    value = value[:MAX_PROMPT_CHARS]
    words = value.split()
    value = " ".join(words[:max_words]).strip()
    if not value:
        raise ValueError("Ollama returned an empty prompt")
    return value


def clean_speech(value: str) -> str:
    return " ".join(value.split())


@dataclass(frozen=True)
class TimingPolicy:
    """Sentence-aware delays measured after rt-stt has completed an utterance.

    The ASR hangover has already supplied a real acoustic pause. Terminal
    punctuation can therefore fire quickly. Commas, conjunctions and max-length
    forced cuts wait for likely continuation. max_batch_wait prevents an endless
    monologue from starving visual updates.
    """

    terminal_delay: float = 0.10
    complete_delay: float = 0.35
    continuation_delay: float = 0.85
    forced_cut_delay: float = 1.20
    max_batch_wait: float = 2.50

    def __post_init__(self):
        values = (self.terminal_delay, self.complete_delay,
                  self.continuation_delay, self.forced_cut_delay,
                  self.max_batch_wait)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("timing values must be finite and nonnegative")
        if self.max_batch_wait <= 0:
            raise ValueError("max_batch_wait must be positive")

    def delay(self, text: str, cut: str = "hangover") -> float:
        text = text.rstrip()
        if cut == "maxlen":
            return self.forced_cut_delay
        if TERMINAL_RE.search(text):
            return self.terminal_delay
        if CONTINUATION_RE.search(text):
            return self.continuation_delay
        final_word = re.sub(r"[^\wÀ-ÿ'-]", "", text.split()[-1].lower()) if text else ""
        if final_word in {
            "and", "or", "but", "because", "with", "then", "that",
            "et", "ou", "mais", "parce", "avec", "puis", "que", "qui",
        }:
            return self.continuation_delay
        return self.complete_delay


@dataclass(frozen=True)
class Snapshot:
    channel: str
    revision: int
    count: int
    text: str
    messages: tuple[dict[str, str], ...]


@dataclass
class _Channel:
    unrendered: deque[str] = field(default_factory=deque)
    history: deque[tuple[str, str]] = field(default_factory=deque)
    revision: int = 0
    due_at: float | None = None
    hard_due_at: float | None = None


class SentenceScheduler:
    """Lossless speech ledger with latest-only visual work per channel."""

    def __init__(self, policy: TimingPolicy | None = None, history_turns: int = 6,
                 min_interval: float = 0.75, clock: Callable[[], float] = time.monotonic):
        if not 0 <= history_turns <= 30:
            raise ValueError("history_turns must be between 0 and 30")
        if not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError("min_interval must be finite and nonnegative")
        self.policy = policy or TimingPolicy()
        self.history_turns = history_turns
        self.min_interval = min_interval
        self.clock = clock
        self.channels: dict[str, _Channel] = {}
        self.last_started = -float("inf")
        self.condition = threading.Condition(threading.RLock())
        self.stopped = False

    def ingest(self, channel: str, text: str, cut: str = "hangover", now: float | None = None) -> bool:
        text = clean_speech(text)
        if not text:
            return False
        if len(text) > MAX_TEXT:
            log.warning("drop oversized speech record channel=%s chars=%d", channel, len(text))
            return False
        channel = str(channel)
        if len(channel) > 64:
            log.warning("drop oversized channel identifier")
            return False
        now = self.clock() if now is None else now
        with self.condition:
            if channel not in self.channels and len(self.channels) >= MAX_CHANNELS:
                log.warning("drop speech record: channel capacity reached")
                return False
            state = self.channels.setdefault(channel, _Channel())
            pending_chars = sum(len(value) for value in state.unrendered)
            if pending_chars + len(text) + len(state.unrendered) > MAX_PENDING_CHARS:
                log.warning("drop speech record channel=%s: pending capacity reached", channel)
                return False
            if not state.unrendered:
                state.hard_due_at = now + self.policy.max_batch_wait
            state.unrendered.append(text)
            state.revision += 1
            natural_due = now + self.policy.delay(text, cut)
            state.due_at = min(natural_due, state.hard_due_at)
            self.condition.notify_all()
        return True

    def _messages(self, state: _Channel, current: str) -> tuple[dict[str, str], ...]:
        messages: list[dict[str, str]] = []
        if self.history_turns:
            for speech, prompt in list(state.history)[-self.history_turns:]:
                messages.append({"role": "user", "content": speech})
                messages.append({"role": "assistant", "content": prompt})
        messages.append({"role": "user", "content": current})
        return tuple(messages)

    def next_ready(self, now: float | None = None) -> tuple[Snapshot | None, float | None]:
        now = self.clock() if now is None else now
        with self.condition:
            earliest = None
            chosen: tuple[str, _Channel] | None = None
            interval_due = self.last_started + self.min_interval
            for channel, state in self.channels.items():
                if not state.unrendered or state.due_at is None:
                    continue
                due = max(state.due_at, interval_due)
                if due <= now and (chosen is None or state.due_at < chosen[1].due_at):
                    chosen = (channel, state)
                earliest = due if earliest is None else min(earliest, due)
            if chosen is None:
                return None, earliest
            channel, state = chosen
            text = clean_speech(" ".join(state.unrendered))
            self.last_started = now
            return Snapshot(channel, state.revision, len(state.unrendered), text,
                            self._messages(state, text)), earliest

    def wait_ready(self) -> Snapshot | None:
        with self.condition:
            while not self.stopped:
                snapshot, due = self.next_ready()
                if snapshot is not None:
                    return snapshot
                timeout = None if due is None else max(0.0, due - self.clock())
                self.condition.wait(timeout)
        return None

    def is_fresh(self, snapshot: Snapshot) -> bool:
        with self.condition:
            state = self.channels.get(snapshot.channel)
            return state is not None and state.revision == snapshot.revision

    def accept(self, snapshot: Snapshot, prompt: str) -> bool:
        with self.condition:
            state = self.channels.get(snapshot.channel)
            if state is None or state.revision != snapshot.revision:
                return False
            return self._accept_locked(state, snapshot, prompt)

    def _accept_locked(self, state: _Channel, snapshot: Snapshot, prompt: str) -> bool:
        for _ in range(snapshot.count):
            state.unrendered.popleft()
        state.history.append((snapshot.text, prompt))
        while len(state.history) > self.history_turns:
            state.history.popleft()
        state.due_at = None
        state.hard_due_at = None
        return True

    def deliver(self, snapshot: Snapshot, prompt: str, send: Callable[[str], None]) -> bool:
        """Freshness check, UDP delivery and history commit are one decision."""
        with self.condition:
            state = self.channels.get(snapshot.channel)
            if self.stopped or state is None or state.revision != snapshot.revision:
                return False
            try:
                send(prompt)
            except PromptDeliveryError as exc:
                if exc.prompt_sent:
                    exc.accepted = self._accept_locked(state, snapshot, prompt)
                raise
            return self._accept_locked(state, snapshot, prompt)

    def stop(self):
        with self.condition:
            self.stopped = True
            for state in self.channels.values():
                state.revision += 1
            self.condition.notify_all()


class OllamaClient:
    def __init__(self, model: str = "gemma3:4b", url: str = "http://127.0.0.1:11434",
                 timeout: float = 120.0, num_ctx: int = 4096, num_predict: int = 64,
                 temperature: float = 1.3, top_p: float = 0.9,
                 system_prompt: str | None = None,
                 transport: Callable[[str, dict, float], dict] | None = None,
                 allow_remote: bool = False):
        if (not math.isfinite(timeout) or timeout <= 0 or num_ctx < 512
                or not 1 <= num_predict <= 128
                or not math.isfinite(temperature) or temperature < 0
                or not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("invalid Ollama compute budget")
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
            raise ValueError("Ollama URL must be an HTTP(S) URL without embedded credentials")
        hostname = parsed.hostname.lower()
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            if hostname != "localhost":
                raise ValueError("Ollama URL host must be an IP literal or localhost; DNS names are not accepted")
            address = ipaddress.ip_address("127.0.0.1")
        loopback = address.is_loopback
        if not loopback and not allow_remote:
            raise ValueError("remote Ollama would receive private speech; pass --allow-remote-ollama explicitly")
        # Use the validated numeric address so DNS cannot escape the request deadline.
        numeric_host = f"[{address}]" if address.version == 6 else str(address)
        netloc = numeric_host + (f":{parsed.port}" if parsed.port is not None else "")
        url = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
        self.model = model
        self.url = url.rstrip("/") + "/api/chat"
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = (system_prompt if system_prompt is not None
                              else PROMPT_PATH.read_text(encoding="utf-8").strip())
        self.transport = transport or self._http

    @staticmethod
    def _http(url: str, payload: dict, timeout: float) -> dict:
        """Stream to completion; timeout bounds socket inactivity, not total generation.

        Model loading and a healthy progressive decode must not be killed by an
        eight-second whole-request watchdog. Each blocking socket operation has
        the configured no-progress timeout; response capacity remains bounded.
        """
        parsed = urlsplit(url)
        connection_type = (http.client.HTTPSConnection if parsed.scheme == "https"
                           else http.client.HTTPConnection)
        connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
        response = None
        stage = "request"
        try:
            controlled = dict(payload, stream=True)
            body = json.dumps(controlled).encode("utf-8")
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            connection.request("POST", path, body,
                               {"Content-Type": "application/json", "Connection": "close"})
            stage = "response headers / model loading"
            response = connection.getresponse()
            if response.status >= 400:
                raise ValueError(f"Ollama HTTP {response.status}")
            pending = b""
            total = 0
            content: list[str] = []
            final = None
            stage = "response stream"
            while final is None:
                block = response.read1(16384)
                if not block:
                    break
                total += len(block)
                if total > MAX_HTTP_RESPONSE:
                    raise ValueError("Ollama response exceeds capacity")
                pending += block
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("error"):
                        raise ValueError("Ollama returned an error")
                    content.append(record.get("message", {}).get("content", ""))
                    if record.get("done"):
                        final = record
            if final is None:
                raise ValueError("Ollama stream ended before completion")
            return dict(final, message={"content": "".join(content)})
        except (TimeoutError, socket.timeout) as exc:
            raise TimeoutError(
                f"Ollama made no socket progress for {timeout:g}s during {stage}"
            ) from exc
        finally:
            if response is not None:
                response.close()
            connection.close()

    def payload(self, messages: Iterable[dict[str, str]]) -> dict:
        return {
            "model": self.model,
            "think": False,
            "stream": False,
            "keep_alive": -1,
            "messages": [{"role": "system", "content": self.system_prompt}, *messages],
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }

    def generate(self, messages: Iterable[dict[str, str]]) -> tuple[str, dict]:
        response = self.transport(self.url, self.payload(messages), self.timeout)
        prompt = normalize_prompt(response.get("message", {}).get("content", ""))
        metrics = {key: response.get(key) for key in (
            "load_duration", "prompt_eval_count", "prompt_eval_duration",
            "eval_count", "eval_duration", "total_duration") if key in response}
        return prompt, metrics


class PromptSink:
    def __init__(self, host: str | None = "127.0.0.1", port: int = 9000,
                 address: str = "/txt2img", on_send: Callable[[str], None] | None = None):
        if not address.startswith("/") or address == "/":
            raise ValueError("OSC output address must be a non-root OSC path")
        if not 1 <= port <= 65535:
            raise ValueError("OSC output port must be between 1 and 65535")
        self.address = address.rstrip("/")
        self.trigger_address = self.address + "/trigger"
        self.client = SimpleUDPClient(host, port) if host else None
        self.on_send = on_send

    def send(self, prompt: str):
        if self.client is not None:
            try:
                self.client.send_message(self.address, prompt)
            except OSError as exc:
                raise PromptDeliveryError("OSC prompt delivery failed") from exc
            try:
                self.client.send_message(self.trigger_address, 1)
            except OSError as exc:
                raise PromptDeliveryError("OSC trigger delivery failed", prompt_sent=True) from exc
        if self.on_send:
            self.on_send(prompt)

    def close(self):
        if self.client is not None:
            self.client.close()


class PromptWorker:
    def __init__(self, scheduler: SentenceScheduler, client: OllamaClient, sink: PromptSink):
        self.scheduler = scheduler
        self.client = client
        self.sink = sink
        self.thread = threading.Thread(target=self.run, name="velocity-prompt-worker", daemon=True)
        self.generated = 0
        self.discarded = 0
        self.errors = 0

    def start(self):
        self.thread.start()

    def run(self):
        while True:
            snapshot = self.scheduler.wait_ready()
            if snapshot is None:
                return
            started = time.perf_counter()
            try:
                prompt, metrics = self.client.generate(snapshot.messages)
                if not self.scheduler.is_fresh(snapshot):
                    self.discarded += 1
                    log.info("discard stale channel=%s %.3fs", snapshot.channel,
                             time.perf_counter() - started)
                    continue
                if not self.scheduler.deliver(snapshot, prompt, self.sink.send):
                    self.discarded += 1
                    log.info("discard raced channel=%s before send", snapshot.channel)
                    continue
                self.generated += 1
                eval_ns = metrics.get("eval_duration") or 0
                eval_count = metrics.get("eval_count") or 0
                rate = eval_count * 1e9 / eval_ns if eval_ns else None
                log.info("prompt channel=%s words=%d elapsed=%.3fs decode_tps=%s %r",
                         snapshot.channel, len(prompt.split()), time.perf_counter() - started,
                         f"{rate:.1f}" if rate is not None else "n/a", prompt)
            except Exception as exc:
                if isinstance(exc, PromptDeliveryError) and exc.prompt_sent:
                    self.generated += int(exc.accepted)
                    self.errors += 1
                    log.error("partial OSC delivery channel=%s prompt_sent=true trigger_sent=false; "
                              "speech_consumed=%s", snapshot.channel, exc.accepted)
                    continue
                self.errors += 1
                log.error("generation failed channel=%s error=%s elapsed=%.3fs "
                          "idle_timeout=%s; speech retained, no OSC prompt sent",
                          snapshot.channel, type(exc).__name__,
                          time.perf_counter() - started,
                          getattr(self.client, "timeout", "n/a"))
                # Do not consume speech. Retry it after one minimum interval.
                with self.scheduler.condition:
                    state = self.scheduler.channels.get(snapshot.channel)
                    if state and state.revision == snapshot.revision:
                        state.due_at = self.scheduler.clock() + max(
                            self.scheduler.min_interval, 0.25)
                        self.scheduler.condition.notify_all()

    def close(self, timeout: float = 10.0):
        self.scheduler.stop()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError("prompt worker did not stop; OSC sink left open")
        self.sink.close()


class JsonlTailer:
    """Follow newly appended rt-stt records; existing transcript is skipped by default."""

    def __init__(self, path: Path, callback: Callable[[dict], None], from_start: bool = False,
                 poll: float = 0.05):
        self.path = Path(path)
        self.callback = callback
        self.from_start = from_start
        self.poll = poll
        self.stop = threading.Event()

    def run(self):
        def opened(start_at_end: bool):
            handle = self.path.open("rb")
            if start_at_end:
                handle.seek(0, 2)
            return handle, os.fstat(handle.fileno())

        handle, identity = opened(not self.from_start)
        discarding = False
        try:
            while not self.stop.is_set():
                position = handle.tell()
                raw = handle.readline(MAX_JSONL_LINE + 2)
                if not raw:
                    try:
                        current = self.path.stat()
                        replaced = not os.path.samestat(identity, current)
                        truncated = current.st_size < position
                    except FileNotFoundError:
                        replaced = truncated = False
                    if replaced or truncated:
                        try:
                            new_handle, new_identity = opened(False)
                        except FileNotFoundError:
                            self.stop.wait(self.poll)
                            continue
                        handle.close()
                        handle, identity = new_handle, new_identity
                        discarding = False
                        continue
                    self.stop.wait(self.poll)
                    continue
                if discarding:
                    if raw.endswith(b"\n"):
                        discarding = False
                    continue
                if len(raw) > MAX_JSONL_LINE:
                    log.warning("ignored oversized JSONL record")
                    discarding = not raw.endswith(b"\n")
                    continue
                if not raw.endswith(b"\n"):
                    handle.seek(position)
                    self.stop.wait(self.poll)
                    continue
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    log.warning("ignored invalid UTF-8 JSONL record")
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("ignored incomplete JSONL record")
                    continue
                if not isinstance(record, dict):
                    log.warning("ignored non-object JSONL record")
                    continue
                try:
                    self.callback(record)
                except Exception as exc:
                    log.error("ignored JSONL callback failure error=%s", type(exc).__name__)
        finally:
            handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="High-velocity authoritative rt-stt JSONL -> Gemma 3 4B -> OSC prompts")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-system-prompt", action="store_true")
    mode.add_argument("--text", help="one-shot model request; prints prompt, sends no OSC")
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--from-start", action="store_true",
                        help="process existing JSONL; default follows only new records")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--allow-remote-ollama", action="store_true",
                        help="explicitly permit sending private speech to a non-loopback Ollama URL")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Socket no-progress timeout in seconds; not a total generation deadline (default 120)")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--min-interval", type=float, default=0.75)
    parser.add_argument("--terminal-delay", type=float, default=0.10)
    parser.add_argument("--complete-delay", type=float, default=0.35)
    parser.add_argument("--continuation-delay", type=float, default=0.85)
    parser.add_argument("--forced-cut-delay", type=float, default=1.20)
    parser.add_argument("--max-batch-wait", type=float, default=2.50)
    parser.add_argument("--output-host", default="127.0.0.1")
    # Match the installation route used by the other prompt bridges: the
    # speech/agent side owns 9000 and TouchDesigner receives prompts on 9001.
    parser.add_argument("--output-port", type=int, default=9001)
    parser.add_argument("--output-address", default="/txt2img")
    parser.add_argument("--no-osc", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if args.print_system_prompt:
        print(system_prompt)
        return 0
    client = OllamaClient(model=args.model, url=args.ollama_url, timeout=args.timeout,
                          num_ctx=args.num_ctx, num_predict=args.num_predict,
                          temperature=args.temperature, top_p=args.top_p,
                          system_prompt=system_prompt,
                          allow_remote=args.allow_remote_ollama)
    if args.text is not None:
        text = clean_speech(args.text)
        if not text:
            raise SystemExit("--text cannot be empty")
        prompt, _ = client.generate([{"role": "user", "content": text}])
        print(prompt)
        return 0
    if not args.transcript.exists():
        raise SystemExit(f"transcript does not exist: {args.transcript}")
    policy = TimingPolicy(args.terminal_delay, args.complete_delay,
                          args.continuation_delay, args.forced_cut_delay,
                          args.max_batch_wait)
    scheduler = SentenceScheduler(policy, args.history_turns, args.min_interval)
    sink = PromptSink(None if args.no_osc else args.output_host, args.output_port,
                      args.output_address, on_send=print if args.no_osc else None)
    worker = PromptWorker(scheduler, client, sink)

    def ingest(record: dict):
        text = record.get("text")
        if not isinstance(text, str):
            return
        channel = str(record.get("channel", "default"))
        scheduler.ingest(channel, text, str(record.get("cut", "hangover")))

    tailer = JsonlTailer(args.transcript, ingest, args.from_start)
    log.info("follow %s model=%s history=%d ctx=%d predict=%d output=%s:%d%s",
             args.transcript, args.model, args.history_turns, args.num_ctx,
             args.num_predict, args.output_host, args.output_port, args.output_address)
    log.info("Ollama idle timeout=%.3fs; no whole-request cancellation deadline",
             args.timeout)
    worker.start()
    try:
        tailer.run()
    except KeyboardInterrupt:
        return 0
    finally:
        tailer.stop.set()
        worker.close(timeout=args.timeout + 1)


if __name__ == "__main__":
    raise SystemExit(main())
