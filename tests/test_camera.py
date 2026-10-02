"""Loopback-only request-correlated camera inbox integration tests."""
import base64
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest

# Test fixture only. The inbox checks JPEG transport boundaries, not image semantics.
JPEG = base64.b64encode(b'\xff\xd8\xff' + b'fixture' * 30 + b'\xff\xd9').decode()


def post(url, request_id, image=JPEG):
    body = json.dumps(dict(request_id=request_id, image=image)).encode()
    with urlopen(Request(url, body, {'Content-Type': 'application/json'}), timeout=2) as response:
        return response.status


def test_camera_requests_are_correlated_and_consumed_once():
    from mechanical_meanderings.camera import FrameInbox
    with FrameInbox(port=0) as inbox:
        def send(request_id, endpoint):
            assert post(endpoint, request_id) == 200
            with pytest.raises(HTTPError) as exc:
                post(endpoint, 'wrong-id')
            assert exc.value.code == 409
        assert inbox.request(send, timeout=.5) == JPEG
        with pytest.raises(HTTPError) as exc:
            post(inbox.endpoint, 'expired-id')
        assert exc.value.code == 409


def test_failed_or_late_frame_is_not_accepted():
    from mechanical_meanderings.camera import FrameInbox
    with FrameInbox(port=0) as inbox:
        saved = []
        with pytest.raises(TimeoutError):
            inbox.request(lambda *args: saved.append(args), timeout=.05)
        with pytest.raises(HTTPError) as exc:
            post(saved[0][1], saved[0][0])
        assert exc.value.code == 409


def test_cancel_wakes_waiter_and_close_releases_port():
    from mechanical_meanderings.camera import FrameInbox
    inbox = FrameInbox(port=0)
    begun = threading.Event()
    errors = []
    def wait():
        try:
            inbox.request(lambda *args: begun.set(), timeout=10)
        except TimeoutError:
            errors.append('cancelled')
    worker = threading.Thread(target=wait)
    worker.start()
    assert begun.wait(1)
    inbox.close()
    worker.join(1)
    assert not worker.is_alive() and errors == ['cancelled']
    assert not inbox.thread.is_alive()


def test_invalid_or_oversized_image_does_not_enter_inbox():
    from mechanical_meanderings.camera import FrameInbox
    with FrameInbox(port=0) as inbox:
        def send(request_id, endpoint):
            for bad in ('not base64!', base64.b64encode(b'not jpeg').decode()):
                with pytest.raises(HTTPError) as exc:
                    post(endpoint, request_id, bad)
                assert exc.value.code == 400
            post(endpoint, request_id)
        assert inbox.request(send, timeout=.5) == JPEG
