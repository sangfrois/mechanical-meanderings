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
SCENE_TIMEOUT = 3.
_detector = None
_labels = None
_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_frame_thread = None
_results = queue.Queue(maxsize=4)
_last_enabled = None
_last_profile = None
_last_heartbeat = 0.
_expires = 0.
_message_kind = ''
_last_speech = ''
_caption_number = 0

# Native installed fonts, not downloaded assets. Short text carries the instruction;
# typeface, stacking, spacing and colour carry the playful part.
_CAPTION_STYLES = (
    ('Georgia', True, False, 1.3, .3, (1., .95, .25)),
    ('Consolas', True, False, 1.4, 1., (.3, 1., .95)),
    ('Impact', True, False, 1.6, 1.2, (1., .45, .8)),
    ('Verdana', False, True, 1.2, .2, (.98, .97, .91)),
    ('Arial', True, False, 1.3, .5, (.3, 1., .95)),
    ('Georgia', True, False, 1.5, 2., (1., .95, .25)),
)
_CAPTION_POSITIONS = ((-.23, .29), (.23, -.27), (.23, .29),
    (-.23, -.27), (0., .02), (0., .30))


_AGENT_FONTS = ('Verdana', 'Georgia', 'Consolas', 'Impact', 'Arial')
_DECORATIONS = {'none': '', 'tilde': '~', 'swung': '⁓', 'slashes': '//'}


def parse_typography(value):
    """Untrusted model design data; never evaluate text, font paths or expressions."""
    if isinstance(value, str):
        if len(value) > 4096:
            raise ValueError('typography packet too large')
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError('invalid typography')
    result = dict(font='Verdana', size=30., x=.5, y=.8, align='center',
        italic=False, bold=False, tracking=0., color=[.98, .97, .91],
        line_width=40, decoration='none')
    if set(value) - set(result):
        raise ValueError('unknown typography field')
    result.update(value)
    for key, low, high in [('size', 18, 64), ('x', 0, 1), ('y', 0, 1),
            ('tracking', -2, 8), ('line_width', 12, 60)]:
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError('invalid typography ' + key)
    if result['font'] not in _AGENT_FONTS or result['align'] not in ('left', 'center', 'right'):
        raise ValueError('invalid typography font/alignment')
    if result['decoration'] not in _DECORATIONS or not all(isinstance(result[k], bool) for k in ('bold', 'italic')):
        raise ValueError('invalid typography style')
    rgb = result['color']
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        raise ValueError('invalid typography color')
    result['color'] = tuple(_score(v) for v in rgb)
    result['line_width'] = int(result['line_width'])
    return result


def caption_layout(text, number, width, height, base_size=30., kind='invitation', typography=None):
    """Agent x/y are the text block centre, 0..1 from the image's top-left."""
    if typography is not None:
        style = parse_typography(typography)
        # Keep the model's punctuation and deliberate line breaks, disable legacy parsing.
        clean = re.sub(r'[\[\]{}]', '', text).strip()
        lines = [line for paragraph in clean.splitlines() for line in
            textwrap.wrap(paragraph, style['line_width'], break_long_words=True, break_on_hyphens=False)]
        marker = _DECORATIONS[style['decoration']]
        decorated = '\n'.join(lines)
        if marker:
            decorated = marker + ' ' + decorated + ' ' + marker
        factor = min(width / 1280., height / 720.)
        return dict(text=decorated, font=style['font'], italic=style['italic'], bold=style['bold'],
            x=style['x'] - .5, y=.5 - style['y'], size=style['size'] * factor,
            tracking=style['tracking'] * factor, color=style['color'], align=style['align'],
            requested_center=[style['x'], style['y']], source='agent')
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
    measured = None
    for name, offset in (('display_text', 0.), ('caption_shadow', 2.)):
        node = b.op(name)
        values = dict(font=layout['font'], fontfile='', italic=layout['italic'], bold=layout.get('bold', False),
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
        if layout.get('source') == 'agent':
            face = ('Bold Italic' if layout['bold'] and layout['italic'] else
                'Bold' if layout['bold'] else 'Italic' if layout['italic'] else 'Regular')
            if face in node.par.typeface.menuNames:
                node.par.typeface = face
            node.par.fontautosize = 'nofit'
            node.par.alignx = layout['align']
            if measured is None:
                w, h = node.evalTextSize(layout['text'])
                ratio = min(1., (node.width - 24.) / max(w, 1.), (node.height - 24.) / max(h, 1.))
                if ratio < 1.:
                    layout['size'] *= ratio
                    layout['tracking'] *= ratio
                    node.par.fontsizex = layout['size']
                    node.par.trackingx = layout['tracking']
                    w, h = node.evalTextSize(layout['text'])
                cx = max(12. + w / 2., min(node.width - 12. - w / 2., layout['requested_center'][0] * node.width))
                cy = max(12. + h / 2., min(node.height - 12. - h / 2., layout['requested_center'][1] * node.height))
                layout['actual_center'] = [cx / node.width, cy / node.height]
                layout['bounds_px'] = [cx - w / 2., cy - h / 2., cx + w / 2., cy + h / 2.]
                measured = w, h, cx, cy
            w, h, cx, cy = measured
            anchor = {'left': (cx - w / 2.) / node.width,
                'center': cx / node.width - .5, 'right': (cx + w / 2.) / node.width - 1.}[layout['align']]
            node.par.positionx = anchor + offset / node.width
            node.par.positiony = .5 - cy / node.height - offset / node.height
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


class DetectorFreshness:
    """Only sequence changes observed this session can establish live detections."""
    def __init__(self):
        self.sequence = None
        self.changed_at = None
        self.seen_live = False

    def observe(self, sequence, now, packed_at):
        if self.sequence is None:
            # A saved predictions DAT is not evidence of a live detector.
            self.sequence, self.changed_at = sequence, now
            return 'waiting'
        changed = sequence != self.sequence
        if changed:
            self.sequence, self.changed_at = sequence, now
        if (not isinstance(packed_at, (int, float)) or not math.isfinite(packed_at)
                or packed_at <= 0. or now < packed_at or now - packed_at >= SCENE_TIMEOUT):
            return 'camera_stale'
        if changed:
            self.seen_live = True
            return 'fresh'
        if now - self.changed_at >= SCENE_TIMEOUT:
            return 'detector_stale'
        return 'unchanged' if self.seen_live else 'waiting'


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


# Speech analysis sets smooth targets; manual composition controls stay untouched.
_motion = None
_control_last_step = None
_control_status = ''


def speech_targets(value):
    controls = value.get('controls')
    if not isinstance(controls, dict):
        raise ValueError('missing controls')
    amount, warp = _score(controls.get('amount')), _score(controls.get('warp'))
    names = ('preserve_identity', 'transformation_intent', 'visual_specificity')
    if all(name in value for name in names):
        preserve, intent, specificity = (_score(value[name]) for name in names)
        strength = (1. - preserve) * (.65 + .35 * max(intent, specificity))
    else:
        # Compatibility with older control-only packets; remove the old .12 floor.
        strength = min(1., max(0., (amount - .12) / .78))
    return dict(amount=.5 + .5 * strength, warp=warp,
        step0=40. - 39. * strength, step1=49. - 47. * strength)


def bounded_steps(step0, step1):
    first = max(1, min(48, int(round(step0))))
    second = max(2, first + 1, min(49, int(round(step1))))
    return first, second


class ControlTransition:
    def __init__(self, start, target, now, seconds):
        if not math.isfinite(seconds) or seconds <= 0.:
            raise ValueError('interpolation seconds must be positive and finite')
        self.start, self.target, self.started, self.seconds = dict(start), dict(target), now, seconds
        self.elapsed, self.last_sample = 0., now

    def sample(self, now):
        # Pausing/stalling TD must not fast-forward a control straight to its target.
        delta = max(0., now - self.last_sample)
        self.last_sample = max(self.last_sample, now)
        self.elapsed = min(self.seconds, self.elapsed + min(.1, delta))
        phase = 1. if self.elapsed + 1e-9 >= self.seconds else self.elapsed / self.seconds
        ease = phase * phase * (3. - 2. * phase)
        return {name: value + (self.target[name] - value) * ease
            for name, value in self.start.items()}, phase >= 1.


def apply_mediation(root, value):
    global _motion, _control_last_step, _control_status
    b = root.op('interactivity')
    if not b.par.Enabled or not b.par.Controls or value.get('apply_visual') is not True:
        return
    target = speech_targets(value)
    morph, stream = root.op('morph_controls'), root.op('StreamDiffusionTD')
    start = dict(amount=float(morph.par.Speechamount), warp=float(morph.par.Speechwarp),
        step0=float(stream.par.Tindexblock0step), step1=float(stream.par.Tindexblock1step))
    _motion = ControlTransition(start, target, time.monotonic(), float(morph.par.Smoothing))
    _control_last_step = None
    _control_status = 'Speech targets queued; manual controls unchanged'
    return target


def _write_steps(stream, first, second):
    old_first, old_second = int(stream.par.Tindexblock0step), int(stream.par.Tindexblock1step)
    # Write in an order that cannot trigger the existing neighbour clamp spuriously.
    order = ((1, second), (0, first)) if first >= old_second else ((0, first), (1, second))
    for index, value in order:
        name = 'Tindexblock' + str(index) + 'step'
        par = stream.par[name]
        # Own the numeric slider while speech automation is enabled; bindings/exports
        # must not silently replace the interpolated value.
        try:
            par.mode = ParMode.CONSTANT
        except NameError:
            pass  # Plain CPython has no TouchDesigner ParMode.
        if int(par) != value:
            par.val = value


def interpolate_controls(b, now):
    global _motion, _control_last_step, _control_status
    if _motion is None:
        return
    if not b.par.Enabled or not b.par.Controls:
        _motion = None
        _control_status = 'Speech automation stopped; manual controls unchanged'
        return
    root = b.parent()
    morph, stream = root.op('morph_controls'), root.op('StreamDiffusionTD')
    values, finished = _motion.sample(now)
    # These are separate speech-owned sliders, never Amount, Warp, or Auto.
    morph.par.Speechamount = min(1., max(.5, values['amount']))
    morph.par.Speechwarp = min(1., max(0., values['warp']))
    if morph.par.Stepsdrive and (finished or _control_last_step is None or now - _control_last_step >= .1):
        first, second = bounded_steps(values['step0'], values['step1'])
        _write_steps(stream, first, second)
        _control_last_step = now
    _control_status = 'Speech Morph {:.3f}; Step targets {:.1f}/{:.1f}'.format(
        values['amount'], _motion.target['step0'], _motion.target['step1'])
    if finished:
        _motion = None
        _control_status += '; interpolation complete'


def set_status(b, text):
    b.op('status').text = text + (' | ' + _control_status if _control_status else '')


def allowed_endpoint(endpoint, port):
    try:
        p = urlsplit(endpoint)
        return (p.scheme == 'http' and p.hostname == '127.0.0.1' and p.port == port
            and p.path == '/frame' and not p.username and not p.password and not p.query and not p.fragment)
    except (ValueError, TypeError):
        return False


def detector_component(root):
    b = root.op('interactivity')
    par = getattr(b.par, 'Yolo', None)
    if par is not None:
        target = par.eval()
        if target is None:
            raise ValueError('configured YOLO component is missing')
        return target
    # Compatibility for older saves; live patch uses an explicit OP reference.
    target = root.op('yolo1') or root.op('yolo')
    if target is None:
        raise ValueError('YOLO component is missing')
    return target


def camera_request(root, request_id, endpoint):
    global _frame_thread
    b = root.op('interactivity')
    if not b.par.Enabled or not re.fullmatch('[0-9a-f]{32}', request_id) or not allowed_endpoint(endpoint, int(b.par.Frameport)):
        raise ValueError('camera request not allowed')
    if _frame_thread is not None and _frame_thread.is_alive():
        raise RuntimeError('camera request already active')
    sender = detector_component(root).op('chopexec3').module
    if time.monotonic() - getattr(sender, '_packed_at', -float('inf')) > 2.:
        raise ValueError('YOLO camera buffer is stale')
    import numpy as np
    h, w = getattr(sender, '_packed_shape', (sender.INPUT_H, sender.INPUT_W))
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
        result = dict(request_id=request_id, status='error', requested_at=time.monotonic())
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


def show_message(root, text, kind, ttl=12., typography=None):
    global _expires, _message_kind, _caption_number
    b = root.op('interactivity')
    if not b.par.Enabled:
        return
    if not isinstance(text, str) or len(text) > 2048 or kind not in ('invitation', 'reply', 'aside', 'challenge'):
        raise ValueError('invalid machine message')
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 < ttl <= 60:
        raise ValueError('invalid message lifetime')
    text = ' '.join(text.split())
    if kind == 'invitation' and (len(text.split()) > 7 or len(text) > 48):
        text = ('Dis au micro quoi changer.' if text.casefold().startswith(('parle', 'dis', 'dites'))
            else 'Tell the mic what to change.')
    elif kind == 'reply':
        text = ' '.join(text.split()[:28])
    elif kind == 'aside':
        text = ' '.join(text.split()[:16])[:120]
    elif kind == 'challenge' and (len(text.split()) > 24 or len(text) > 160):
        raise ValueError('challenge exceeds caption limit')
    if not text:
        b.op('message').text = ''
        _expires = 0.
        return
    display = b.op('display_text')
    layout = caption_layout(text, _caption_number, display.width, display.height,
        float(b.par.Fontsize), kind, typography)
    apply_caption_layout(b, layout)
    b.op('message').text = layout['text']
    b.op('last_message').text = json.dumps(dict(message=text, kind=kind, ttl=ttl,
        typography=layout, sequence=_caption_number), ensure_ascii=False)
    if kind in ('invitation', 'challenge'):
        _caption_number += 1
    _message_kind = kind
    _expires = time.monotonic() + ttl


def speech(root, text):
    global _last_speech, _expires
    text = ' '.join(text.split())
    if text and text != _last_speech:
        _last_speech = text
        # Do not let the mirror talk over a human who accepted its invitation.
        if _message_kind in ('invitation', 'aside', 'challenge'):
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
            targets = apply_mediation(root, value)
            if targets is not None:
                value['td_targets'] = targets
            b.op('judgement').text = json.dumps(value, ensure_ascii=False, indent=2)
        elif address == '/engagement/message' and len(args) in (3, 4):
            show_message(root, *args)
        elif address == '/creative/status' and len(args) == 1 and isinstance(args[0], str) and len(args[0]) <= 1024:
            value = json.loads(args[0])
            if isinstance(value, dict) and set(value) == {'disco', 'asides', 'camera'}:
                b.op('profile_status').text = json.dumps(dict(value, acknowledged_at=time.monotonic()))
        elif address == '/transcription' and args and isinstance(args[0], str):
            speech(root, args[0])
    except Exception as exc:
        if address == '/camera/request':
            b.op('camera_status').text = json.dumps(dict(status='error', error=str(exc), at=time.monotonic()))
        set_status(b, 'interaction error: ' + type(exc).__name__)


def tick(b):
    global _last_poll, _detector, _labels, _last_enabled, _expires, _motion, _control_status
    global _last_profile, _last_heartbeat
    now = time.monotonic()
    try:
        interpolate_controls(b, now)
    except Exception as exc:
        _motion = None
        _control_status = 'Speech control error: ' + type(exc).__name__
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
        heartbeat = now - _last_heartbeat >= 5.
        if _last_enabled != enabled or heartbeat:
            _send_enabled(b, enabled)
            _last_enabled = enabled
        profile = dict(disco=bool(b.par.Disco), asides=bool(b.par.Asides), camera=bool(b.par.Cameraenabled))
        if profile != _last_profile or heartbeat:
            _send(b, '/creative/profile', json.dumps(profile))
            _last_profile = profile
        if heartbeat:
            _last_heartbeat = now
        if not enabled:
            _detector = None
            b.op('scene').text = json.dumps(dict(status='disabled'))
            set_status(b, 'Machine interaction disabled')
            return
        root = b.parent()
        yolo = detector_component(root)
        if _labels is None:
            _labels = [row[0].val for row in yolo.op('yolo_labels').rows()]
        value = json.loads(yolo.op('yolo_server/predictions').text)
        summary = scene_summary(value, _labels, float(b.par.Confidence))
        sender = yolo.op('chopexec3').module
        if _detector is None:
            _detector = DetectorFreshness()
        state = _detector.observe(summary['sequence'], now, getattr(sender, '_packed_at', 0.))
        if state != 'fresh':
            if state == 'camera_stale':
                reason = getattr(sender, '_last_error', '') or 'YOLO camera buffer is stale'
            elif state == 'detector_stale':
                reason = f"YOLO detector stale: frame {summary['sequence']} stopped advancing"
            else:
                reason = f"Waiting for a new YOLO detector frame; last sequence {summary['sequence']}"
            # Do not keep old people/objects displayed as a current scene.
            b.op('scene').text = json.dumps(dict(status=state,
                last_sequence=summary['sequence'], reason=reason), ensure_ascii=False, indent=2)
            set_status(b, reason + '; no scene sent')
            return
        _send(b, '/scene', json.dumps(summary))
        b.op('scene').text = json.dumps(dict(summary, status='fresh'), ensure_ascii=False, indent=2)
        set_status(b, f"{summary['people']} people; fresh detector frame {summary['sequence']}; scene sent")
    except Exception as exc:
        b.op('scene').text = json.dumps(dict(status='unavailable', error=type(exc).__name__))
        set_status(b, 'scene error: ' + type(exc).__name__ + '; no scene sent')


def close():
    _socket.close()
