"""Reviewed invitation boundaries, using synthetic responses and isolated OSC."""
import json
import queue

import pytest

from mechanical_meanderings.__main__ import main
from mechanical_meanderings.invitation import InvitationPrompt
from mechanical_meanderings.invite_preview import preview_invitations
from mechanical_meanderings.live import Settings
from mechanical_meanderings.mediation import MediatingAgent
from test_interactive import harness, event_of
from test_invitation_prompt import MESSAGE, answer, prompt_path, ready_state


SECOND = 'Dis au micro ce que tu veux voir. Cette machine manque sérieusement de goût.'


def test_timeout_during_pre_send_file_read_cannot_deliver(monkeypatch, prompt_path):
    original, reads, active = InvitationPrompt.read, [0], []
    def expire_third_read(self):
        reads[0] += 1
        if reads[0] == 3:
            assert active
            active[0].cancel('timeout')
        return original(self)
    def transport(url, payload, timeout, control=None):
        active.append(control)
        return answer()
    monkeypatch.setattr(InvitationPrompt, 'read', expire_third_read)
    with harness(transport, agent_options=dict(invitation_prompt=prompt_path),
            engagement=ready_state(), boundary_speech=True) as (bridge, sender, packets, events):
        event = event_of(events, 'error')
        assert event['error'] == 'GenerationTimeout' and not event['udp_sent']
        assert reads[0] == 3 and not bridge.recent_invitations
        assert bridge.engagement.invitation_count == 0
        with pytest.raises(queue.Empty):
            packets.get(timeout=.1)


@pytest.mark.parametrize('failure,expected', [
    ('timeout', 'GenerationTimeout'), ('stale', 'StaleGeneration'),
    ('prompt_missing', 'InvitationPromptError'), ('resident_changed', 'RuntimeError'),
    ('network', 'OSError'),
])
def test_preview_terminal_failure_keeps_prior_and_current_evidence(monkeypatch, prompt_path, failure, expected):
    ps_calls, chats = [], []
    def residency(*args):
        ps_calls.append(True)
        if len(ps_calls) == 3:
            if failure == 'resident_changed':
                return 4096
            if failure == 'prompt_missing':
                prompt_path.unlink()
        return 8192
    def transport(url, payload, timeout, control=None):
        chats.append(payload)
        if len(chats) == 2:
            if failure == 'timeout':
                control.cancel('timeout')
                control.check()
            elif failure == 'stale':
                prompt_path.write_text('Edited during second inference.', encoding='utf-8')
            elif failure == 'network':
                raise OSError('synthetic network failure')
            return answer(SECOND)
        return answer(MESSAGE)
    monkeypatch.setattr('mechanical_meanderings.invite_preview.resident_context', residency)
    agent = MediatingAgent('unused visual', invitation_prompt=prompt_path, transport=transport)
    result = preview_invitations(agent, Settings(seed=731), count=4)
    assert result['status'] == 'aborted' and result['terminal_error']['type'] == expected
    assert len(result['samples']) == 2  # Abort after the failing attempt, never retry.
    first, second = result['samples']
    assert first['accepted'] is True and first['message'] == MESSAGE
    assert first['payload'] == chats[0] and first['payload_kind'] == 'logical'
    assert json.loads(first['raw_response'])['message'] == MESSAGE
    assert second['accepted'] is False and second['error']
    if failure == 'resident_changed':
        assert len(chats) == 1 and second['raw_response'] is None
        # Preserve the prepared request, without claiming that another chat occurred.
        assert second['payload']['options']['seed'] == 732
        assert second['logical_payload'] == second['payload']
    else:
        assert len(chats) == 2 and second['payload'] == chats[1]
        if failure in ('stale', 'prompt_missing'):
            assert json.loads(second['raw_response'])['message'] == SECOND
        else:
            assert second['raw_response'] is None
    assert agent.transport is transport


def test_seed_sequence_overflow_is_rejected_before_any_network(monkeypatch, prompt_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid seed sequence must be rejected before residency or chat')
    monkeypatch.setattr('mechanical_meanderings.invite_preview.resident_context', forbidden)
    agent = MediatingAgent('unused', invitation_prompt=prompt_path, transport=forbidden)
    result = preview_invitations(agent, Settings(seed=2147483647), count=2)
    assert result['status'] == 'aborted' and result['samples'] == []
    assert result['terminal_error']['type'] == 'ValueError'
    assert 'seed' in result['terminal_error']['message']
    assert agent.transport is forbidden


def test_cli_prints_prior_success_and_terminal_failure_json(monkeypatch, prompt_path, capsys):
    original = MediatingAgent.__init__
    chats = []
    def transport(url, payload, timeout, control=None):
        chats.append(payload)
        if len(chats) == 2:
            control.cancel('timeout')
            control.check()
        return answer()
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.transport = transport
    monkeypatch.setattr(MediatingAgent, '__init__', initialize)
    monkeypatch.setattr('mechanical_meanderings.invite_preview.resident_context', lambda *args: 8192)
    assert main(['--preview-invite', '--invite-prompt', str(prompt_path), '--invite-count', '3']) == 1
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert report['status'] == 'aborted' and report['terminal_error']['type'] == 'GenerationTimeout'
    assert len(report['samples']) == 2 and report['samples'][0]['message'] == MESSAGE
    assert report['samples'][1]['payload'] == chats[1]


def test_preview_ignores_unused_visual_file_and_service_only_settings(monkeypatch, prompt_path, capsys):
    original = MediatingAgent.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.transport = lambda *args: answer()
    monkeypatch.setattr(MediatingAgent, '__init__', initialize)
    monkeypatch.setattr('mechanical_meanderings.invite_preview.resident_context', lambda *args: 8192)
    absent = prompt_path.with_name('nonexistent-visual.txt')
    assert main(['--preview-invite', '--invite-prompt', str(prompt_path), '--system-prompt', str(absent),
                 '--idle-after', '-1', '--max-invitations', '-1']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'complete' and report['samples'][0]['message'] == MESSAGE
