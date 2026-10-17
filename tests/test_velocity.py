import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import threading
import time

import pytest
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from mechanical_meanderings.velocity import (
    JsonlTailer,
    MAX_CHANNELS,
    OllamaClient,
    PromptDeliveryError,
    PromptSink,
    PromptWorker,
    SentenceScheduler,
    TimingPolicy,
    build_parser,
    normalize_prompt,
)


def immediate_policy():
    return TimingPolicy(0, 0, 0, 0, .2)


def test_default_output_matches_shared_touchdesigner_txt2img_route():
    args = build_parser().parse_args([])
    assert (args.output_host, args.output_port, args.output_address) == (
        "127.0.0.1", 9001, "/txt2img")


def test_timing_uses_sentence_shape_and_forced_cut():
    p = TimingPolicy(.1, .35, .85, 1.2, 2.5)
    assert p.delay("a complete thought.") == .1
    assert p.delay("a visual fragment") == .35
    assert p.delay("a visual fragment,") == .85
    assert p.delay("a visual fragment", "maxlen") == 1.2
    assert p.delay("et") == .85


def test_scheduler_coalesces_fragments_and_preserves_chat_context():
    now = [0.]
    scheduler = SentenceScheduler(TimingPolicy(.1, .2, .8, 1.2, 2.5),
                                  history_turns=2, min_interval=0,
                                  clock=lambda: now[0])
    scheduler.ingest("1", "a red room,", now=now[0])
    now[0] = .4
    scheduler.ingest("1", "filled with rain.", now=now[0])
    snapshot, due = scheduler.next_ready(now=.49)
    assert snapshot is None and due == .5
    snapshot, _ = scheduler.next_ready(now=.5)
    assert snapshot.text == "a red room, filled with rain."
    assert snapshot.messages == ({"role": "user", "content": snapshot.text},)
    assert scheduler.accept(snapshot, "red room, rainfall, reflective")

    now[0] = 1.
    scheduler.ingest("1", "make it colder.", now=now[0])
    snapshot, _ = scheduler.next_ready(now=1.1)
    assert [m["role"] for m in snapshot.messages] == ["user", "assistant", "user"]
    assert snapshot.messages[-1]["content"] == "make it colder."


def test_hard_deadline_prevents_continuation_starvation():
    scheduler = SentenceScheduler(TimingPolicy(.1, .2, 1., 1., 1.5), min_interval=0,
                                  clock=lambda: 0.)
    scheduler.ingest("1", "one,", now=0.)
    scheduler.ingest("1", "two,", now=1.)
    snapshot, due = scheduler.next_ready(now=1.49)
    assert snapshot is None and due == 1.5
    snapshot, _ = scheduler.next_ready(now=1.5)
    assert snapshot.text == "one, two,"


def test_normalize_hard_caps_model_output():
    content = "<think>hidden</think> Prompt: " + " ".join(f"w{i}" for i in range(30))
    result = normalize_prompt(content)
    assert len(result.split()) == 20
    assert result.startswith("w0 w1") and result.endswith("w19")
    assert len(normalize_prompt("x" * 10000)) == 2048


def test_ollama_payload_is_small_non_thinking_and_resident():
    captured = {}

    def transport(url, payload, timeout):
        captured.update(url=url, payload=payload, timeout=timeout)
        return {"message": {"content": "figure, rain, cyanotype"},
                "eval_count": 4, "eval_duration": 1_000_000_000}

    client = OllamaClient(transport=transport, system_prompt="system")
    prompt, metrics = client.generate([{"role": "user", "content": "rain"}])
    assert prompt == "figure, rain, cyanotype"
    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["model"] == "gemma3:4b"
    assert captured["payload"]["think"] is False
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["keep_alive"] == -1
    assert captured["payload"]["options"] == {
        "num_ctx": 4096,
        "num_predict": 64,
        "temperature": 1.3,
        "top_p": .9,
    }
    assert metrics["eval_count"] == 4


def test_remote_ollama_requires_explicit_private_speech_opt_in():
    with pytest.raises(ValueError, match="private speech"):
        OllamaClient(url="http://192.0.2.10:11434", system_prompt="system")
    with pytest.raises(ValueError, match="DNS names"):
        OllamaClient(url="http://127.attacker.invalid:11434", allow_remote=True,
                     system_prompt="system")
    client = OllamaClient(url="https://192.0.2.10", allow_remote=True,
                          system_prompt="system", transport=lambda *_: {})
    assert client.url == "https://192.0.2.10/api/chat"
    local = OllamaClient(url="http://localhost:11434", system_prompt="system")
    assert local.url == "http://127.0.0.1:11434/api/chat"


def test_scheduler_bounds_channels_text_and_pending_speech():
    scheduler = SentenceScheduler(immediate_policy(), min_interval=0)
    for channel in range(MAX_CHANNELS):
        assert scheduler.ingest(str(channel), "word")
    assert not scheduler.ingest("overflow", "word")
    assert not SentenceScheduler(immediate_policy()).ingest("1", "x" * 8193)
    crowded = SentenceScheduler(immediate_policy())
    assert crowded.ingest("1", "x" * 8192)
    assert crowded.ingest("1", "y" * 8192)
    assert crowded.ingest("1", "z" * 8192)
    assert not crowded.ingest("1", "q" * 8192)


def test_worker_discards_superseded_generation_then_coalesces_latest():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    class Client:
        def generate(self, messages):
            text = messages[-1]["content"]
            calls.append(text)
            if len(calls) == 1:
                entered.set()
                assert release.wait(2)
            return ("visual " + text, {})

    sent = []
    scheduler = SentenceScheduler(immediate_policy(), history_turns=2, min_interval=0)
    worker = PromptWorker(scheduler, Client(), PromptSink(None, on_send=sent.append))
    worker.start()
    scheduler.ingest("1", "first.")
    assert entered.wait(2)
    scheduler.ingest("1", "second.")
    release.set()
    deadline = time.time() + 2
    while len(sent) < 1 and time.time() < deadline:
        time.sleep(.01)
    worker.close()
    assert calls[:2] == ["first.", "first. second."]
    assert sent == ["visual first. second."]
    assert worker.discarded == 1 and worker.generated == 1


def test_stop_during_inference_invalidates_output():
    entered = threading.Event()
    release = threading.Event()

    class Client:
        def generate(self, messages):
            entered.set()
            assert release.wait(2)
            return "obsolete", {}

    sent = []
    scheduler = SentenceScheduler(immediate_policy(), min_interval=0)
    worker = PromptWorker(scheduler, Client(), PromptSink(None, on_send=sent.append))
    worker.start()
    scheduler.ingest("1", "first.")
    assert entered.wait(2)
    scheduler.stop()
    release.set()
    worker.thread.join(2)
    worker.close()
    assert sent == []


def test_close_does_not_close_sink_while_worker_is_alive():
    entered = threading.Event()
    release = threading.Event()

    class Client:
        def generate(self, messages):
            entered.set()
            release.wait(2)
            return "obsolete", {}

    class Sink:
        closed = False
        def send(self, prompt):
            raise AssertionError("stopped output must not send")
        def close(self):
            self.closed = True

    scheduler = SentenceScheduler(immediate_policy(), min_interval=0)
    sink = Sink()
    worker = PromptWorker(scheduler, Client(), sink)
    worker.start()
    scheduler.ingest("1", "first.")
    assert entered.wait(2)
    with pytest.raises(TimeoutError, match="left open"):
        worker.close(timeout=.01)
    assert not sink.closed
    release.set()
    worker.thread.join(2)
    worker.close()
    assert sink.closed


def test_partial_osc_delivery_consumes_speech_without_duplicate_prompt():
    scheduler = SentenceScheduler(immediate_policy(), min_interval=0)
    scheduler.ingest("1", "first.", now=0)
    snapshot, _ = scheduler.next_ready(now=0)

    def partial_send(prompt):
        raise PromptDeliveryError("trigger failed", prompt_sent=True)

    with pytest.raises(PromptDeliveryError) as raised:
        scheduler.deliver(snapshot, "visual", partial_send)
    assert raised.value.accepted
    assert list(scheduler.channels["1"].unrendered) == []
    assert list(scheduler.channels["1"].history) == [("first.", "visual")]


def test_jsonl_tailer_starts_at_eof_and_delivers_new_complete_record(tmp_path):
    path = tmp_path / "transcript.jsonl"
    path.write_text(json.dumps({"text": "old"}) + "\n", encoding="utf-8")
    seen = []
    tailer = JsonlTailer(path, seen.append, poll=.01)
    thread = threading.Thread(target=tailer.run)
    thread.start()
    time.sleep(.03)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"channel": 1, "text": "new."}) + "\n")
        handle.flush()
    deadline = time.time() + 1
    while not seen and time.time() < deadline:
        time.sleep(.01)
    tailer.stop.set()
    thread.join(1)
    assert seen == [{"channel": 1, "text": "new."}]


def test_jsonl_tailer_ignores_valid_non_objects_and_callback_failures(tmp_path):
    path = tmp_path / "transcript.jsonl"
    path.write_text("", encoding="utf-8")
    seen = []

    def callback(record):
        if record.get("text") == "bad callback":
            raise RuntimeError("fixture")
        seen.append(record)

    tailer = JsonlTailer(path, callback, poll=.01)
    thread = threading.Thread(target=tailer.run)
    thread.start()
    time.sleep(.03)
    with path.open("a", encoding="utf-8") as handle:
        for value in (None, [1], "scalar", {"text": "bad callback"}, {"text": "kept"}):
            handle.write(json.dumps(value) + "\n")
        handle.flush()
    deadline = time.time() + 1
    while not seen and time.time() < deadline:
        time.sleep(.01)
    tailer.stop.set()
    thread.join(1)
    assert seen == [{"text": "kept"}]


def test_jsonl_tailer_bounds_oversized_unterminated_input_and_invalid_utf8(tmp_path):
    path = tmp_path / "transcript.jsonl"
    path.write_bytes(b"")
    seen = []
    tailer = JsonlTailer(path, seen.append, poll=.01)
    thread = threading.Thread(target=tailer.run)
    thread.start()
    time.sleep(.03)
    with path.open("ab") as handle:
        handle.write(b"x" * 70000)
        handle.flush()
    time.sleep(.03)
    with path.open("ab") as handle:
        handle.write(b"\n\xff\xfe\n")
        handle.write(json.dumps({"text": "after malformed"}).encode() + b"\n")
        handle.flush()
    deadline = time.time() + 1
    while not seen and time.time() < deadline:
        time.sleep(.01)
    tailer.stop.set()
    thread.join(1)
    assert seen == [{"text": "after malformed"}]


def test_jsonl_tailer_reopens_replaced_file_from_start(tmp_path):
    path = tmp_path / "transcript.jsonl"
    path.write_text("", encoding="utf-8")
    seen = []
    tailer = JsonlTailer(path, seen.append, poll=.01)
    thread = threading.Thread(target=tailer.run)
    thread.start()
    time.sleep(.03)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(json.dumps({"text": "rotated"}) + "\n", encoding="utf-8")
    os.replace(replacement, path)
    deadline = time.time() + 1
    while not seen and time.time() < deadline:
        time.sleep(.01)
    tailer.stop.set()
    thread.join(1)
    assert seen == [{"text": "rotated"}]


def test_http_transport_enforces_total_deadline():
    class SlowHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for _ in range(20):
                try:
                    self.wfile.write(b'{"message":{"content":"x"},"done":false}\n')
                    self.wfile.flush()
                except OSError:
                    break
                time.sleep(.03)
        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    client = OllamaClient(url=f"http://127.0.0.1:{server.server_address[1]}",
                          timeout=.12, system_prompt="system")
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            client.generate([{"role": "user", "content": "test"}])
        assert time.monotonic() - started < .5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)


def test_http_transport_rejects_oversized_response():
    class LargeHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * 70000)
        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), LargeHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    client = OllamaClient(url=f"http://127.0.0.1:{server.server_address[1]}",
                          timeout=1, system_prompt="system")
    try:
        with pytest.raises(ValueError, match="exceeds capacity"):
            client.generate([{"role": "user", "content": "test"}])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)


def test_prompt_sink_sends_real_osc_prompt_then_trigger():
    packets = []
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(lambda address, *args: packets.append((address, args)))
    server = BlockingOSCUDPServer(("127.0.0.1", 0), dispatcher)
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    sink = PromptSink("127.0.0.1", server.server_address[1], "/txt2img")
    sink.send("mirror, rain")
    thread.join(1)
    # handle_request receives one datagram; SimpleUDPClient sends one datagram per message.
    server.handle_request()
    sink.close()
    server.server_close()
    assert packets == [
        ("/txt2img", ("mirror, rain",)),
        ("/txt2img/trigger", (1,)),
    ]
