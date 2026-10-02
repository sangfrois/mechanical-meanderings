"""TD helper policy tested outside TD; no camera/GPU operations in these tests."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def helper():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'td_interactivity.py'
    spec = importlib.util.spec_from_file_location('td_helper_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_yolo_summary_uses_actual_labels_and_does_not_double_count_pose():
    module = helper()
    value = dict(seq=33, yolo=[dict(categoryName=[0], score=.8, id=3),
        dict(categoryName=[1], score=.9, id=4), dict(categoryName=[1], score=.1, id=5)],
        yolo_pose=[dict(score=.7, id=8)])
    summary = module.scene_summary(value, ['person', 'cup'], .45)
    assert summary['people'] == 1
    assert summary['objects'] == [dict(label='person', confidence=.8), dict(label='cup', confidence=.9)]
    assert summary['sequence'] == 33
    assert all('id' not in obj for obj in summary['objects'])


def test_mediation_is_bounded_and_ignored_for_incidental_speech():
    module = helper()
    morph = SimpleNamespace(par=SimpleNamespace(Auto=True, Amount=1., Warp=.1))
    unit = SimpleNamespace(par=SimpleNamespace(Controls=True))
    root = SimpleNamespace(op=lambda name: unit if name == 'interactivity' else morph)
    value = dict(apply_visual=True, controls=dict(amount=.65, warp=.2))
    module.apply_mediation(root, value)
    assert morph.par.Amount == .65 and morph.par.Warp == .2 and morph.par.Auto is False
    module.apply_mediation(root, dict(apply_visual=False, controls=dict(amount=0., warp=0.)))
    assert morph.par.Amount == .65


def test_camera_callback_refuses_non_loopback_or_wrong_port():
    module = helper()
    assert module.allowed_endpoint('http://127.0.0.1:9012/frame', 9012)
    for address in ['http://example.com/frame', 'http://127.0.0.1:80/frame',
                    'file:///etc/passwd', 'http://user@127.0.0.1:9012/frame',
                    'http://127.0.0.1:9012/frame?other=1']:
        assert not module.allowed_endpoint(address, 9012)


def test_source_sampling_and_crop_bounds_are_unwarped():
    shader = (Path(__file__).resolve().parents[1] / 'scripts' / 'source_center_morph.glsl').read_text()
    assert 'texture(sTD2DInputs[0],uv);' in shader
    assert 'edge=min(cropUV,1.0-cropUV)' in shader
    assert 'vec2 su=' not in shader
    assert 'mix(src,gen,mask)' in shader  # Preserve source alpha where the mask is zero.


def test_new_invitation_layout_moves_and_varies_readable_typography():
    module = helper()
    layouts = [module.caption_layout('Parle au micro. Transforme ton reflet.', i, 1280, 720)
        for i in range(6)]
    assert len({(item['x'], item['y']) for item in layouts}) == 6
    assert len({item['font'] for item in layouts}) >= 4
    assert len({item['size'] for item in layouts}) >= 3
    assert len({item['tracking'] for item in layouts}) >= 2
    for item in layouts:
        assert -.3 <= item['x'] <= .3 and -.3 <= item['y'] <= .3
        assert 24 <= item['size'] <= 52
        assert len(item['text'].splitlines()) <= 4
        assert '[' not in item['text'] and 'MIRROR /' not in item['text']
    assert any(item['text'].isupper() for item in layouts)
    assert any(not item['text'].isupper() for item in layouts)


def test_invitation_event_updates_both_text_layers_and_empty_clear_does_not_move():
    module = helper()
    class Pars(SimpleNamespace):
        def __getitem__(self, name):
            if name not in self.__dict__:
                setattr(self, name, SimpleNamespace(val=None, expr=None))
            return getattr(self, name)
    nodes = {name: SimpleNamespace(par=Pars(), width=1280, height=720, text='')
        for name in ('display_text', 'caption_shadow', 'message', 'last_message')}
    unit = SimpleNamespace(par=SimpleNamespace(Enabled=True, Fontsize=30), op=nodes.get)
    root = SimpleNamespace(op=lambda name: unit)
    module.show_message(root, 'Speak into the microphone. Change your reflection.', 'invitation')
    first = json.loads(nodes['last_message'].text)
    module.show_message(root, 'Parle au micro. Transforme ton reflet.', 'invitation')
    second = json.loads(nodes['last_message'].text)
    assert first['typography']['x'] != second['typography']['x']
    assert nodes['display_text'].par.positionx.val != nodes['caption_shadow'].par.positionx.val
    assert 'MIRROR' not in nodes['display_text'].par.text.expr
    assert nodes['display_text'].par.alignxmode.val == 'bbox'
    assert nodes['caption_shadow'].par.font.val == second['typography']['font']
    module.show_message(root, '', 'reply')
    assert not nodes['message'].text and module._caption_number == 2


def test_caption_preserves_french_accents_and_scales_for_small_outputs():
    module = helper()
    layout = module.caption_layout('Parle au micro. Réinvente ton reflet !', 1, 640, 360)
    assert 'RÉINVENTE' in layout['text']
    assert layout['size'] < module.caption_layout('Parle au micro.', 1, 1280, 720)['size']
    assert '[' not in module.caption_layout('[100,0]Speak into the microphone.', 0, 1280, 720)['text']


def test_caption_boundary_rejects_long_invitation_without_cutting_a_fragment():
    module = helper()
    class Pars(SimpleNamespace):
        def __getitem__(self, name):
            if name not in self.__dict__:
                setattr(self, name, SimpleNamespace(val=None, expr=None))
            return getattr(self, name)
    nodes = {name: SimpleNamespace(par=Pars(), width=1280, height=720, text='')
        for name in ('display_text', 'caption_shadow', 'message', 'last_message')}
    unit = SimpleNamespace(par=SimpleNamespace(Enabled=True, Fontsize=30), op=nodes.get)
    root = SimpleNamespace(op=lambda name: unit)
    module.show_message(root, 'Tell the microphone what to become while the entire room changes around your reflection.', 'invitation')
    value = json.loads(nodes['last_message'].text)
    assert len(value['message'].split()) <= 7 and len(value['message']) <= 48
    assert value['message'].endswith('.')
    assert value['message'] == 'Tell the mic what to change.'
