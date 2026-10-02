"""Install the verified local TD interactivity profile; call install(root, scripts, backups).

Run from TD's Python environment, not ordinary CPython. It preserves existing
callbacks in an exclusive backup, creates only owned nodes, and never saves or
replaces the original .toe. The supplied root must contain the inspected
Mechanical Meanderings source/YOLO/StreamDiffusion network.
"""
import json
from datetime import datetime
from pathlib import Path


def install(root, scripts, backups):
    scripts, backups = Path(scripts), Path(backups)
    needed = ('oscin3', 'oscin3_callbacks', 'source_morph_pixel', 'morph_controls',
              'yolo/chopexec3', 'yolo/yolo_server/predictions', 'yolo/yolo_labels', 'over1', 'null3')
    if any(root.op(name) is None for name in needed):
        raise ValueError('this TD network does not match the verified installation profile')
    b = root.op('interactivity')
    if b is not None and not b.fetch('mm_owned', False) and any(c.name not in ('tick', 'display_text') for c in b.children):
        raise ValueError('refusing to replace an unowned interactivity component')
    backups.mkdir(parents=True, exist_ok=True)
    backup = backups / ('td-before-' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.json')
    snapshot = dict(root=root.path, project_folder=project.folder, project_name=project.name,
        dats={name: root.op(name).text for name in ('oscin3_callbacks', 'source_morph_pixel', 'yolo/chopexec3')},
        oscin1_active=root.op('oscin1').par.active.eval() if root.op('oscin1') is not None else None,
        morph={p.name: p.eval() for p in root.op('morph_controls').customPars},
        null3_inputs=[o.path for o in root.op('null3').inputs])
    with backup.open('x', encoding='utf-8') as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)
    b = b or root.create(baseCOMP, 'interactivity')
    b.store('mm_owned', True)
    if not hasattr(b.par, 'Enabled'):
        page = b.appendCustomPage('Interaction')
        for name, label, default in [('Enabled', 'Machine invitations and replies', True),
            ('Controls', 'LLM regulates morph amount / warp', True),
            ('Messagevisible', 'Show machine caption', True),
            ('Flipframe', 'Flip cached TD camera rows upright', True)]:
            par = page.appendToggle(name, label=label)[0]
            par.default, par.val = default, default
        for name, label, default in [('Agentport', 'Agent input OSC port', 9000), ('Frameport', 'Loopback JPEG inbox port', 9012)]:
            par = page.appendInt(name, label=label)[0]
            par.default, par.val = default, default
            par.min, par.max, par.clampMin, par.clampMax = 1, 65535, True, True
        for name, label, default, low, high in [('Confidence', 'YOLO confidence threshold', .45, 0., 1.),
                ('Fontsize', 'Machine caption font size', 30., 12., 72.)]:
            par = page.appendFloat(name, label=label)[0]
            par.default, par.val = default, default
            par.min, par.max, par.clampMin, par.clampMax = low, high, True, True
    for name in ('logic', 'message', 'last_message', 'scene', 'judgement', 'status', 'camera_status', 'notes'):
        if b.op(name) is None:
            b.create(textDAT, name)
    b.op('logic').text = (scripts / 'td_interactivity.py').read_text(encoding='utf-8')
    b.op('message').text = ''
    b.op('notes').text = (
        'Mechanical Meanderings: selective machine engagement.\n'
        'YOLO summary -> Agentport once/second; no periodic camera/model calls.\n'
        'Camera requested only by /camera/request; copied cached CHW buffer, JPEG/HTTP off TD thread.\n'
        'No TOP readback added. No images saved. Frameport must match --frame-port.\n'
        'Enabled disables invitations/replies; Controls disables LLM amount/warp overrides.\n'
        'Messagevisible hides captions; Flipframe converts bottom-up TD rows to upright JPEG.\n'
        'Source sampling and fixed crop stay unchanged; generated branch alone may warp.\n'
        'Judgements describe intent, not identity or truth. Live modifications require project saving.\n')
    tick = b.op('tick') or b.create(executeDAT, 'tick')
    # Preserve TD's actual template/signatures; replace only the onFrameEnd/onExit bodies.
    original = tick.text
    marker = 'def onFrameEnd(frame: int):'
    if 'op(\'logic\').module.tick' not in original:
        start = original.index(marker)
        end = original.index('\ndef onPlayStateChange', start)
        original = original[:start] + marker + "\n\tme.parent().op('logic').module.tick(me.parent())\n\treturn\n" + original[end:]
        start = original.index('def onExit():')
        end = original.index('\ndef onFrameStart', start)
        original = original[:start] + "def onExit():\n\tme.parent().op('logic').module.close()\n\treturn\n" + original[end:]
        tick.text = original
    tick.par.framestart = False
    tick.par.frameend = True
    tick.par.active = True
    display = b.op('display_text') or b.create(textTOP, 'display_text')
    display.par.text.expr = "op('message').text if parent().par.Messagevisible else ''"
    display.par.outputresolution = 'custom'
    display.par.resolutionw.expr = "parent().parent().op('over1').width"
    display.par.resolutionh.expr = "parent().parent().op('over1').height"
    display.par.fontsizexunit = 'pixels'
    display.par.fontsizeyunit = 'pixels'
    display.par.fontsizex.expr = 'parent().par.Fontsize'
    display.par.fontsizey.expr = 'parent().par.Fontsize'
    display.par.alignx, display.par.aligny = 'center', 'center'
    display.par.alignxmode, display.par.alignymode = 'bbox', 'bbox'
    display.par.positionunit = 'fraction'
    display.par.positionx, display.par.positiony = 0., -.3
    display.par.wordwrap = False
    display.par.fontautosize = 'fitiffat'
    display.par.borderspace1 = .07
    display.par.borderspace2 = .07
    display.par.fontcolorr, display.par.fontcolorg, display.par.fontcolorb = .96, .96, .9
    display.par.fontalpha = 1.
    display.par.bgalpha = 0.
    # Text-only shadow: legibility without dimming the entire original camera frame.
    shadow = b.op('caption_shadow') or b.copy(display, name='caption_shadow')
    shadow.par.fontcolorr, shadow.par.fontcolorg, shadow.par.fontcolorb = 0., 0., 0.
    shadow.par.positionx.expr = "op('display_text').par.positionx + 2. / parent().parent().op('over1').width"
    shadow.par.positiony.expr = "op('display_text').par.positiony - 2. / parent().parent().op('over1').height"
    captions = b.op('caption_over') or b.create(compositeTOP, 'caption_over')
    captions.par.operand = 'over'
    captions.inputConnectors[0].connect(display.outputConnectors[0])
    captions.inputConnectors[1].connect(shadow.outputConnectors[0])
    out = b.op('out') or b.create(outTOP, 'out')
    out.inputConnectors[0].connect(captions.outputConnectors[0])
    layer = root.op('machine_over') or root.create(compositeTOP, 'machine_over')
    layer.par.operand = 'over'
    layer.inputConnectors[0].connect(b.outputConnectors[0])
    layer.inputConnectors[1].connect(root.op('over1').outputConnectors[0])
    root.op('null3').inputConnectors[0].connect(layer.outputConnectors[0])
    receiver = root.op('oscin3_callbacks')
    signature = 'def onReceiveOSC(dat, rowIndex, message, byteData, timeStamp, address, args, peer):\n'
    if "op('interactivity/logic').module.handle" not in receiver.text:
        if receiver.text.count(signature) != 1:
            raise ValueError('unexpected OSC callback signature; backup created, refusing blind callback replacement')
        receiver.text = receiver.text.replace(signature, signature +
            "\tme.parent().op('interactivity/logic').module.handle(me.parent(), address, args)\n")
    sender = root.op('yolo/chopexec3')
    if '_packed_at = time.monotonic()' not in sender.text:
        marker = 'def _send(client, webserver, payload_len, H, W):\n'
        if sender.text.count(marker) != 1:
            raise ValueError('unexpected YOLO sender contract')
        sender.text = sender.text.replace(marker, marker + '    global _packed_at\n    _packed_at = time.monotonic()\n')
    root.op('source_morph_pixel').text = (scripts / 'source_center_morph.glsl').read_text(encoding='utf-8')
    # This unused listener competes with the agent; forwarded raw subtitles arrive on 9001.
    unused = root.op('oscin1')
    if unused is not None and int(unused.par.port) == int(b.par.Agentport):
        unused.par.active = False
    b.nodeX, b.nodeY = 600, 1040
    layer.nodeX, layer.nodeY = 1000, 780
    layout = [('message', 0, 260), ('last_message', 0, 130), ('logic', 200, 260),
        ('tick', 200, 130), ('display_text', 400, 260), ('caption_shadow', 400, 130),
        ('caption_over', 600, 260), ('out', 800, 260),
        ('scene', 0, -130), ('judgement', 200, -130), ('status', 400, -130),
        ('camera_status', 600, -130), ('notes', 200, -390)]
    for name, x, y in layout:
        o = b.op(name)
        o.nodeX, o.nodeY = x, y
        o.nodeWidth, o.nodeHeight = 130, 90
        o.viewer = True
        o.activeViewer = False
    b.nodeWidth, b.nodeHeight = 130, 90
    layer.nodeWidth, layer.nodeHeight = 130, 90
    b.viewer, layer.viewer = True, True
    b.activeViewer, layer.activeViewer = False, False
    return dict(backup=str(backup), component=b.path, caption_output=layer.path,
        camera_readback_added=False, original_project_saved=False)
