import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import test_bridge


class CLITests(unittest.TestCase):
    setUp = test_bridge.BridgeTests.setUp
    tearDown = test_bridge.BridgeTests.tearDown

    def test_sampling_cli_sends_explicit_options(self):
        result = subprocess.run([sys.executable, '-m', 'mechanical_meanderings',
            '--text', 'brass bird', '--temperature', '1.1', '--num-ctx', '4096',
            '--top-p', '.9', '--top-k', '55', '--repeat-penalty', '1.2',
            '--seed', '42', '--num-predict', '80',
            '--ollama-url', f'http://127.0.0.1:{self.http.server_port}'],
            capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        _, payload = self.requests.get(timeout=1)
        self.assertEqual(payload['options'], dict(temperature=1.1, num_ctx=4096,
            top_p=.9, top_k=55, repeat_penalty=1.2, seed=42, num_predict=80))

    def test_tui_cli_owns_bridge_and_journal(self):
        from unittest.mock import patch
        from mechanical_meanderings.__main__ import main
        seen = []
        def run(app):
            seen.append(app)
            self.assertEqual(app.bridge.settings.temperature, 1.1)
            app.session.close()
        with tempfile.TemporaryDirectory() as directory:
            with patch('mechanical_meanderings.tui.MeanderingsApp.run', run):
                self.assertEqual(main(['--tui', '--listen-port', '0', '--temperature', '1.1',
                                       '--session-dir', directory]), 0)
            self.assertEqual(len(list(Path(directory).glob('*.jsonl'))), 1)
        self.assertEqual(seen[0].bridge.client._sock.fileno(), -1)

    def test_cli_explicit_dynamic_without_trigger(self):
        self._assert_cli_service(dynamic=True)

    def test_cli_default_waits_for_boundary_even_past_debounce(self):
        self._assert_cli_service(dynamic=False)

    def _assert_cli_service(self, dynamic):
        import socket
        from pythonosc.udp_client import SimpleUDPClient
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        s.close()
        proc = subprocess.Popen([sys.executable, '-u', '-m', 'mechanical_meanderings',
            '--listen-host', '127.0.0.1', '--listen-port', str(port),
            '--output-host', '127.0.0.1', '--output-port', str(self.sink.server_address[1]),
            '--ollama-url', f'http://127.0.0.1:{self.http.server_port}',
            '--debounce', '0.05', '--update-interval', '0.1',
            *(['--dynamic'] if dynamic else [])],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        observed = []
        reader = None
        try:
            # Read actual readiness; a fixed sleep drops the first UDP packet on
            # a busy GPU workstation before the subprocess has bound its socket.
            import queue
            import threading
            lines = queue.Queue()
            def read_logs():
                for line in proc.stderr:
                    observed.append(line)
                    lines.put(line)
            reader = threading.Thread(target=read_logs, daemon=True)
            reader.start()
            for _ in range(5):
                if 'Listening on' in lines.get(timeout=5):
                    break
            else:
                self.fail('service never reported readiness')
            self.assertIsNone(proc.poll(), 'service exited before receiving text')
            sender = SimpleUDPClient('127.0.0.1', port)
            sender.send_message('/transcription', 'visual thought without trigger')
            if not dynamic:
                self.assertEqual(self.received.get(timeout=1),
                    ('/transcription', ('visual thought without trigger',)))
                with self.assertRaises(queue.Empty):
                    self.requests.get(timeout=.25)
                sender.send_message('/trigger', 1)
                self.assertEqual(self.received.get(timeout=1), ('/trigger', (1,)))
            sender.close()
            self.assertEqual(self.received.get(timeout=3)[1], ('A metal flower turns.',))
        finally:
            proc.terminate()
            proc.wait(timeout=3)
            if reader is not None:
                reader.join(timeout=1)
            _, logs = proc.communicate(timeout=3)
            logs += ''.join(observed)
        self.assertNotIn('INFO: RX', logs)

    def test_interactive_one_shot_prints_decision_not_a_chat_prompt(self):
        from test_mediation import judgement
        self.responses.get()
        self.responses.put({'message': {'content': json.dumps(judgement())}})
        result = subprocess.run([sys.executable, '-m', 'mechanical_meanderings',
            '--interactive', '--text', 'Turn me into Batman',
            '--ollama-url', f'http://127.0.0.1:{self.http.server_port}'],
            capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['category'], 'transformation')
        _, payload = self.requests.get(timeout=1)
        self.assertEqual(payload['format']['type'], 'object')

    def test_occupied_frame_port_reports_duplicate_bridge(self):
        import socket
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            result = subprocess.run([sys.executable, '-m', 'mechanical_meanderings',
                '--interactive', '--frame-port', str(port), '--listen-port', '0'],
                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn('already in use', result.stderr)
        self.assertIn(str(port), result.stderr)
        self.assertIn('previous bridge', result.stderr)

    def test_one_shot_cli_uses_custom_system_and_stdout(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            prompt = Path(directory) / 'system.txt'
            prompt.write_text('custom visual system', encoding='utf-8')
            result = subprocess.run([sys.executable, '-m', 'mechanical_meanderings',
                '--text', 'spoken idea', '--system-prompt', str(prompt),
                '--ollama-url', f'http://127.0.0.1:{self.http.server_port}',
                '--history-turns', '0', '--reset-after', '30', '--queue-size', '2',
                '--listen-port', '9001', '--output-port', '9000', '--output-address', '/transcription',
                '--channel-mode', '--forward-raw'], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'A metal flower turns.\n')
        self.assertEqual(result.stderr, '')
        _, payload = self.requests.get(timeout=1)
        self.assertEqual(payload['messages'][0]['content'], 'custom visual system')
        with self.assertRaises(__import__('queue').Empty):
            self.received.get(timeout=0.1)


if __name__ == '__main__':
    unittest.main()
