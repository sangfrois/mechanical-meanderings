"""Loopback-only camera callback. Frames exist only during one bounded request."""
import base64
import binascii
import json
import math
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def validate_frame(image):
    """Bound transport size before decoding; never include image data in errors."""
    if not isinstance(image, str) or not 1 <= len(image) <= FrameInbox.MAX_BODY:
        raise ValueError('invalid camera frame')
    try:
        raw = base64.b64decode(image, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError('invalid camera encoding') from None
    if not 32 <= len(raw) <= FrameInbox.MAX_JPEG or not raw.startswith(b'\xff\xd8\xff') or not raw.endswith(b'\xff\xd9'):
        raise ValueError('invalid JPEG transport')
    return image


class FrameInbox:
    MAX_BODY = 600_000
    MAX_JPEG = 400_000

    def __init__(self, port=9012):
        self.condition = threading.Condition()
        self.pending = None
        self.image = None
        self.deadline = 0.
        self.closed = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(2)

            def log_message(self, *args):
                pass  # Never log images, request bodies or visitor-related fields.

            def answer(self, status):
                self.send_response(status)
                self.send_header('Content-Length', '0')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()

            def do_POST(self):
                if self.client_address[0] != '127.0.0.1' or self.path != '/frame':
                    self.answer(403)
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= owner.MAX_BODY:
                        self.answer(413)
                        return
                    value = json.loads(self.rfile.read(length))
                    if not isinstance(value, dict):
                        raise ValueError('frame JSON must be an object')
                    request_id, image = value.get('request_id'), value.get('image')
                    if not isinstance(request_id, str) or len(request_id) > 64 or not isinstance(image, str):
                        raise ValueError('invalid frame fields')
                    validate_frame(image)
                except (ValueError, binascii.Error, UnicodeError, TimeoutError):
                    self.answer(400)
                    return
                with owner.condition:
                    if owner.closed or request_id != owner.pending or owner.image is not None or time.monotonic() >= owner.deadline:
                        status = 409
                    else:
                        owner.image = image
                        owner.condition.notify_all()
                        status = 200
                self.answer(status)

        try:
            self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        except OSError as exc:
            raise RuntimeError(f'Camera inbox 127.0.0.1:{port} cannot bind: port already in use '
                'or access denied. Stop the previous bridge before launching another copy.') from exc
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.endpoint = f'http://127.0.0.1:{self.server.server_port}/frame'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name='camera-inbox')
        self.thread.start()

    def request(self, send, timeout=3.):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('frame timeout must be positive and finite')
        with self.condition:
            if self.closed:
                raise TimeoutError('camera inbox closed')
            if self.pending is not None:
                raise RuntimeError('another camera request is pending')
            request_id = secrets.token_hex(16)
            self.pending = request_id
            self.image = None
            self.deadline = time.monotonic() + timeout
        try:
            send(request_id, self.endpoint)
            with self.condition:
                self.condition.wait_for(lambda: self.image is not None or self.pending != request_id or self.closed,
                    max(0., self.deadline - time.monotonic()))
                if self.image is None or self.pending != request_id or self.closed or time.monotonic() >= self.deadline:
                    raise TimeoutError('no current camera frame')
                return self.image
        finally:
            with self.condition:
                if self.pending == request_id:
                    self.pending = None
                    self.image = None
                    self.deadline = 0.

    def cancel(self):
        with self.condition:
            self.pending = None
            self.image = None
            self.deadline = 0.
            self.condition.notify_all()

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = True
        self.cancel()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
