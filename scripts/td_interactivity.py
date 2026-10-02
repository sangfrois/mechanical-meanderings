"""Copy this DAT into a TD interactivity COMP. No TOP readback is performed.

TD operators are accessed ONLY on the main thread. The camera thread gets a
plain copied ndarray, encodes a JPEG with the installed Pillow, and posts it
only to the fixed loopback frame inbox. Images are never written to disk.
"""
import base64
import io
import json
import math
import queue
import re
import socket
import struct
import threading
import time
import textwrap
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_last_poll = 0.
_last_tick = None
_labels = None
_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_frame_thread = None
_results = queue.Queue(maxsize=4)
_last_enabled = None
_expires = 0.
_message_kind = ''
_last_speech = ''
_caption_number = 0

# Native installed fonts, not downloaded assets. Short text carries the instruction;
# typeface, stacking, spacing and colour carry the playful part.
_CAPTION_STYLES = (
    ('Cooper', True, False, 1.3, .3, (1., .95, .25)),
    ('Bauhaus 93', True, False, 1.4, 1., (.3, 1., .95)),
    ('Impact', True, False, 1.6, 1.2, (1., .45, .8)),
    ('Comic Sans MS', False, True, 1.2, .2, (.98, .97, .91)),
    ('Showcard Gothic', True, False, 1.3, .5, (.3, 1., .95)),
    ('Agency FB', True, False, 1.5, 2., (1., .95, .25)),
)
_CAPTION_POSITIONS = ((-.23, .29), (.23, -.27), (.23, .29),
    (-.23, -.27), (0., .02), (0., .30))


def caption_layout(text, number, width, height, base_size=30., kind='invitation'):
    """Positions are offsets from texture centre, NOT absolute normalized UVs."""
    # Prevent model text from injecting TD's legacy inline positioning syntax.
    text = re.sub(r'[\[\]{}]', '', ' '.join(text.split()))
    if kind == 'reply':
        return dict(text=textwrap.fill(text, 52), font='Verdana', italic=False,
            x=0., y=-.30, size=base_size * min(width / 1280., height / 720.),
            tracking=0., color=(.98, .97, .91))
    font, upper, italic, scale, tracking, color = _CAPTION_STYLES[number % len(_CAPTION_STYLES)]
    x, y = _CAPTION_POSITIONS[number % len(_CAPTION_POSITIONS)]
    text = text.upper() if upper else text
    # Deliberate stacking; sentence boundaries stay legible and words stay intact.
    lines = [line for sentence in re.split(r'(?<=[.!?])\s+', text)
        for line in textwrap.wrap(sentence, 22, break_long_words=False, break_on_hyphens=False)]
    factor = min(width / 1280., height / 720.)
    return dict(text='\n'.join(lines), font=font, italic=italic, x=x, y=y,
        size=min(52., base_size * scale) * factor, tracking=tracking * factor, color=color)


def apply_caption_layout(b, layout):
    for name, offset in (('display_text', 0.), ('caption_shadow', 2.)):
        node = b.op(name)
        values = dict(font=layout['font'], italic=layout['italic'], bold=False,
            typeface='Black' if layout['font'] == 'Cooper' else 'Regular',
            fontsizex=layout['size'], fontsizey=layout['size'],
            fontsizexunit='pixels', fontsizeyunit='pixels',
            alignx='center', aligny='center', alignxmode='bbox', alignymode='bbox',
            positionunit='fraction', wordwrap=False, legacyparsing=False,
            fontautosize='fitiffat', linespacing=0., linespacingunit='pixels',
            positionx=layout['x'] + offset / node.width,
            positiony=layout['y'] - offset / node.height, trackingx=layout['tracking'], trackingy=0.)
        for name, value in values.items():
            par = node.par[name]
            par.expr = ''
            par.val = value
        node.par['text'].expr = "op('message').text if parent().par.Messagevisible else ''"
    display = b.op('display_text')
    display.par.fontcolorr, display.par.fontcolorg, display.par.fontcolorb = layout['color']


def scene_summary(value, labels, confidence=.45):
    objects = []
    people = 0
    for obj in value.get('yolo', [])[:128]:
        score = obj.get('score', 0.)
        category = obj.get('categoryName', [])
        if not isinstance(score, (int, float)) or not math.isfinite(score) or score < confidence or score > 1:
            continue
        category = category[0] if isinstance(category, list) and category else category
        if isinstance(category, bool) or not isinstance(category, int) or not 0 <= category < len(labels):
            continue
        label = labels[category]
        if label == 'person':
            people += 1
        if len(objects) < 64:
            objects.append(dict(label=label, confidence=round(float(score), 3)))
    pose_count = sum(1 for p in value.get('yolo_pose', [])[:64]
        if isinstance(p.get('score'), (int, float)) and math.isfinite(p['score']) and confidence <= p['score'] <= 1)
    # Separate detector IDs are not compatible; pose + object counts cannot be added.
    people = min(64, max(people, pose_count))
    sequence = value.get('seq', value.get('frame'))
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError('YOLO sequence missing')
    return dict(people=people, sequence=sequence, objects=objects)


def _osc_string(value):
    raw = value.encode('utf-8') + b'\0'
    return raw + b'\0' * ((-len(raw)) % 4)


def _send(b, address, text):
    packet = _osc_string(address) + _osc_string(',s') + _osc_string(text)
    _socket.sendto(packet, ('127.0.0.1', int(b.par.Agentport)))


def _send_enabled(b, enabled):
    packet = _osc_string('/engagement/enabled') + _osc_string(',i') + struct.pack('>i', int(enabled))
    _socket.sendto(packet, ('127.0.0.1', int(b.par.Agentport)))


def _score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid mediated control')
    return float(value)


def apply_mediation(root, value):
    b = root.op('interactivity')
    if not b.par.Controls or value.get('apply_visual') is not True:
        return
    controls = value.get('controls')
    if not isinstance(controls, dict):
        raise ValueError('missing controls')
    amount, warp = _score(controls.get('amount')), _score(controls.get('warp'))
    morph = root.op('morph_controls')
    morph.par.Auto = False
    morph.par.Amount = amount
    morph.par.Warp = warp


def allowed_endpoint(endpoint, port):
    try:
        p = urlsplit(endpoint)
        return (p.scheme == 'http' and p.hostname == '127.0.0.1' and p.port == port
            and p.path == '/frame' and not p.username and not p.password and not p.query and not p.fragment)
    except (ValueError, TypeError):
        return False


def camera_request(root, request_id, endpoint):
    global _frame_thread
    b = root.op('interactivity')
    if not b.par.Enabled or not re.fullmatch('[0-9a-f]{32}', request_id) or not allowed_endpoint(endpoint, int(b.par.Frameport)):
        raise ValueError('camera request not allowed')
    if _frame_thread is not None and _frame_thread.is_alive():
        raise RuntimeError('camera request already active')
    sender = root.op('yolo/chopexec3').module
    if time.monotonic() - getattr(sender, '_packed_at', -float('inf')) > 2.:
        raise ValueError('YOLO camera buffer is stale')
    import numpy as np
    h, w = sender.INPUT_H, sender.INPUT_W
    payload = sender._payload
    if payload.dtype != np.uint8 or payload.size != h * w * 3:
        raise ValueError('unexpected YOLO camera buffer')
    rgb = payload.reshape(3, h, w).transpose(1, 2, 0)
    if b.par.Flipframe:
        rgb = rgb[::-1, :, :]
    stride = max(1, math.ceil(max(h, w) / 384))
    pixels = np.array(rgb[::stride, ::stride, :], dtype=np.uint8, copy=True)
    # Only ordinary Python values/pixels cross into this worker; never a TD OP.
    def upload():
        result = dict(request_id=request_id, status='error')
        try:
            from PIL import Image
            output = io.BytesIO()
            Image.fromarray(pixels).save(output, format='JPEG', quality=72)
            jpeg = output.getvalue()
            body = json.dumps(dict(request_id=request_id, image=base64.b64encode(jpeg).decode('ascii'))).encode()
            request = Request(endpoint, body, {'Content-Type': 'application/json'})
            with urlopen(request, timeout=2.) as response:
                result.update(status='uploaded', http_status=response.status, bytes=len(jpeg),
                    width=pixels.shape[1], height=pixels.shape[0])
        except Exception as exc:
            result['error'] = type(exc).__name__
        try:
            _results.put_nowait(result)
        except queue.Full:
            pass  # Bounded diagnostics; never hold images to wait for TD.
    _frame_thread = threading.Thread(target=upload, daemon=True, name='meanderings-camera-jpeg')
    _frame_thread.start()


def show_message(root, text, kind, ttl=12.):
    global _expires, _message_kind, _caption_number
    b = root.op('interactivity')
    if not b.par.Enabled:
        return
    if not isinstance(text, str) or len(text) > 2048 or kind not in ('invitation', 'reply'):
        raise ValueError('invalid machine message')
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 < ttl <= 60:
        raise ValueError('invalid message lifetime')
    text = ' '.join(text.split())
    if kind == 'invitation' and (len(text.split()) > 7 or len(text) > 48):
        text = ('Dis au micro quoi changer.' if text.casefold().startswith(('parle', 'dis', 'dites'))
            else 'Tell the mic what to change.')
    elif kind == 'reply':
        text = ' '.join(text.split()[:28])
    if not text:
        b.op('message').text = ''
        _expires = 0.
        return
    display = b.op('display_text')
    layout = caption_layout(text, _caption_number, display.width, display.height,
        float(b.par.Fontsize), kind)
    apply_caption_layout(b, layout)
    b.op('message').text = layout['text']
    b.op('last_message').text = json.dumps(dict(message=text, kind=kind, ttl=ttl,
        typography=layout, sequence=_caption_number), ensure_ascii=False)
    if kind == 'invitation':
        _caption_number += 1
    _message_kind = kind
    _expires = time.monotonic() + ttl


def speech(root, text):
    global _last_speech, _expires
    text = ' '.join(text.split())
    if text and text != _last_speech:
        _last_speech = text
        # Do not let the mirror talk over a human who accepted its invitation.
        if _message_kind == 'invitation':
            root.op('interactivity/message').text = ''
            _expires = 0.


def handle(root, address, args):
    b = root.op('interactivity')
    try:
        if address == '/camera/request' and len(args) == 2:
            camera_request(root, args[0], args[1])
        elif address == '/mediation' and len(args) == 1 and isinstance(args[0], str) and len(args[0]) <= 32768:
            value = json.loads(args[0])
            if not isinstance(value, dict):
                raise ValueError('invalid mediation object')
            apply_mediation(root, value)
            b.op('judgement').text = json.dumps(value, ensure_ascii=False, indent=2)
        elif address == '/engagement/message' and len(args) == 3:
            show_message(root, *args)
        elif address == '/transcription' and args and isinstance(args[0], str):
            speech(root, args[0])
    except Exception as exc:
        b.op('status').text = 'interaction error: ' + type(exc).__name__


def tick(b):
    global _last_poll, _last_tick, _labels, _last_enabled, _expires
    now = time.monotonic()
    if _expires and (now >= _expires or not b.par.Enabled):
        b.op('message').text = ''
        _expires = 0.
    while True:
        try:
            b.op('camera_status').text = json.dumps(_results.get_nowait())
        except queue.Empty:
            break
    if now - _last_poll < 1.:
        return
    _last_poll = now
    try:
        enabled = bool(b.par.Enabled)
        if _last_enabled != enabled:
            _send_enabled(b, enabled)
            _last_enabled = enabled
        if not enabled:
            return
        root = b.parent()
        yolo = root.op('yolo')
        if _labels is None:
            _labels = [row[0].val for row in yolo.op('yolo_labels').rows()]
        value = json.loads(yolo.op('yolo_server/predictions').text)
        summary = scene_summary(value, _labels, float(b.par.Confidence))
        _send(b, '/scene', json.dumps(summary))
        b.op('scene').text = json.dumps(summary, ensure_ascii=False, indent=2)
        b.op('status').text = f"{summary['people']} people; detector frame {summary['sequence']}; scene sent once/second"
    except Exception as exc:
        b.op('status').text = 'scene error: ' + type(exc).__name__


def close():
    _socket.close()
