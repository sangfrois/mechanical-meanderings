import json
import queue
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from mechanical_meanderings import Agent


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.requests = queue.Queue()
        self.responses = queue.Queue()
        self.responses.put({'message': {'content': 'A metal flower turns.'}})
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.requests.put((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                response = owner.responses.get(timeout=3)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            def log_message(self, *args):
                pass
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.received = queue.Queue()
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(lambda address, *args: self.received.put((address, args)))
        self.sink = BlockingOSCUDPServer(('127.0.0.1', 0), dispatcher)
        threading.Thread(target=self.sink.serve_forever, daemon=True).start()
        self.agent = Agent(system_prompt='test system', ollama_url=f'http://127.0.0.1:{self.http.server_port}')
        self.bridge = None

    def tearDown(self):
        if self.bridge:
            self.sender.close()
            self.bridge.close()
            self.assertEqual(self.bridge.client._sock.fileno(), -1)
        self.http.shutdown(); self.http.server_close()
        self.sink.shutdown(); self.sink.server_close()

    def start_bridge(self, **kwargs):
        from mechanical_meanderings.bridge import Bridge
        self.bridge = Bridge(self.agent, listen_host='127.0.0.1', listen_port=0,
                             output_host='127.0.0.1', output_port=self.sink.server_address[1], **kwargs)
        self.bridge.start()
        self.sender = SimpleUDPClient(*self.bridge.address)

    def test_dynamic_updates_without_trigger_and_coalesces_slow_inference(self):
        import time
        self.responses.get()
        self.start_bridge(dynamic=True, debounce=0.03, update_interval=0.08)
        self.sender.send_message('/transcription', 'first image')
        _, payload = self.requests.get(timeout=2)
        self.assertEqual(payload['messages'][-1]['content'], 'first image')
        for i in range(30):
            self.sender.send_message('/transcription', f'newest image {i}')
        time.sleep(.1)
        self.responses.put({'message': {'content': 'first prompt'}})
        self.received.get(timeout=2)
        self.received.get(timeout=2)
        _, payload = self.requests.get(timeout=2)
        self.assertEqual(payload['messages'][-1]['content'], 'newest image 29')
        self.responses.put({'message': {'content': 'latest prompt'}})
        self.received.get(timeout=2)
        self.received.get(timeout=2)
        self.sender.send_message('/transcription', 'newest image 29')
        self.sender.send_message('/trigger', 1)
        with self.assertRaises(queue.Empty):
            self.requests.get(timeout=.2)

    def test_continuous_text_holds_until_a_real_pause(self):
        import time
        self.start_bridge(dynamic=True, debounce=0.2, update_interval=0.05)
        deadline = time.monotonic() + 0.6
        i = 0
        while time.monotonic() < deadline:
            self.sender.send_message('/transcription', f'flowing image {i}')
            i += 1
            time.sleep(.03)
        # No gap in the flood ever reached debounce, so nothing was generated yet.
        self.assertTrue(self.requests.empty())
        _, payload = self.requests.get(timeout=1)
        self.assertTrue(payload['messages'][-1]['content'].startswith('flowing image'))
        self.received.get(timeout=2)
        self.received.get(timeout=2)
        self.sender.send_message('/reset', 1)

    def test_pipeline_logs_receive_generate_send(self):
        self.start_bridge(output_address='/txt2img')
        with self.assertLogs('mechanical_meanderings.bridge', level='INFO') as logs:
            self.sender.send_message('/transcription', 'a glass forest')
            self.sender.send_message('/trigger', 1)
            self.received.get(timeout=2)
            self.received.get(timeout=2)
            import time
            time.sleep(0.05)
        text = '\n'.join(logs.output)
        for marker in ('GENERATE', 'TX', '/txt2img', 'a glass forest'):
            self.assertIn(marker, text)
        self.assertNotIn('RX ', text)
        self.assertNotIn('QUEUED ', text)

    def test_udp_phrase_only_on_final_trigger_to_real_http(self):
        self.start_bridge()
        self.sender.send_message('/trigger', 0)
        self.sender.send_message('/transcription', 'partial')
        self.sender.send_message('/transcription', 'complete input')
        with self.assertRaises(queue.Empty):
            self.requests.get(timeout=0.15)
        self.sender.send_message('/trigger', 1)
        path, payload = self.requests.get(timeout=2)
        self.assertEqual(path, '/api/chat')
        self.assertEqual(payload['messages'][-1]['content'], 'complete input')
        self.assertEqual(self.received.get(timeout=2), ('/prompt', ('A metal flower turns.',)))
        self.assertEqual(self.received.get(timeout=2), ('/prompt/trigger', (1,)))
        self.sender.send_message('/trigger', 1)
        with self.assertRaises(queue.Empty):
            self.requests.get(timeout=0.15)

    def test_udp_reset_discards_inflight_and_queued_work(self):
        self.responses.get()
        self.start_bridge(queue_size=1)
        self.sender.send_message('/transcription', 'old inflight')
        self.sender.send_message('/trigger', 1)
        self.requests.get(timeout=2)
        self.sender.send_message('/transcription', 'old queued')
        self.sender.send_message('/trigger', 1)
        with self.assertLogs('mechanical_meanderings.bridge', level='ERROR') as logs:
            self.sender.send_message('/transcription', 'overflow')
            self.sender.send_message('/trigger', 1)
            import time
            time.sleep(0.1)
        self.assertIn('overflow', logs.output[0])
        self.sender.send_message('/reset', 1)
        import time
        time.sleep(0.1)
        self.responses.put({'message': {'content': 'stale output'}})
        with self.assertRaises(queue.Empty):
            self.received.get(timeout=0.2)
        self.responses.put({'message': {'content': 'fresh output'}})
        self.sender.send_message('/transcription', 'fresh')
        self.sender.send_message('/trigger', 1)
        _, payload = self.requests.get(timeout=2)
        self.assertEqual([m['content'] for m in payload['messages']], ['test system', 'fresh'])
        self.assertEqual(self.received.get(timeout=2), ('/prompt', ('fresh output',)))

    def test_channel_mode_forwarding_and_drop_in_output(self):
        self.start_bridge(channel_mode=True, forward_raw=True, output_address='/transcription')
        self.sender.send_message('/transcription', 'duplicate aggregate')
        self.sender.send_message('/trigger', 1)
        self.sender.send_message('/ch2/transcription', 'channel input')
        self.sender.send_message('/ch2/trigger', 1)
        _, payload = self.requests.get(timeout=2)
        self.assertEqual(payload['messages'][-1]['content'], 'channel input')
        events = [self.received.get(timeout=2) for _ in range(4)]
        self.assertEqual(events, [('/transcription', ('channel input',)), ('/trigger', (1,)),
                                  ('/transcription', ('A metal flower turns.',)), ('/trigger', (1,))])
        with self.assertRaises(queue.Empty):
            self.requests.get(timeout=0.1)

    def test_udp_fifo_history_word_cap_and_error_recovery(self):
        self.responses.get()
        self.responses.put({'message': {'content': ' '.join(['metal'] * 25), 'thinking': 'SECRET'}})
        self.responses.put({'message': {'content': '', 'thinking': 'SECRET'}})
        self.responses.put({'message': {'content': 'recovered'}})
        self.start_bridge()
        self.sender.send_message('/transcription', 'first')
        self.sender.send_message('/trigger', 1)
        self.requests.get(timeout=2)
        self.assertEqual(self.received.get(timeout=2), ('/prompt', (' '.join(['metal'] * 20),)))
        self.received.get(timeout=2)
        with self.assertLogs('mechanical_meanderings.bridge', level='ERROR') as logs:
            self.sender.send_message('/transcription', 'bad')
            self.sender.send_message('/trigger', 1)
            _, payload = self.requests.get(timeout=2)
            with self.assertRaises(queue.Empty):
                self.received.get(timeout=0.2)
        self.assertNotIn('SECRET', str(logs.output))
        self.assertEqual([m['role'] for m in payload['messages']], ['system', 'user', 'assistant', 'user'])
        self.sender.send_message('/transcription', 'third')
        self.sender.send_message('/trigger', 1)
        _, payload = self.requests.get(timeout=2)
        self.assertEqual([m['content'] for m in payload['messages']],
                         ['test system', 'first', ' '.join(['metal'] * 20), 'third'])
        self.assertEqual(self.received.get(timeout=2), ('/prompt', ('recovered',)))


if __name__ == '__main__':
    unittest.main()
