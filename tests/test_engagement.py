"""Deterministic presence/quiet scheduling, without cameras or models."""
import json
import pytest


class Clock:
    now = 0.
    def __call__(self):
        return self.now


def scene(seq, people=1):
    return json.dumps(dict(sequence=seq, people=people, objects=[dict(label='cup', confidence=.8, id=7)]))


def test_presence_and_silence_are_both_required():
    from mechanical_meanderings.engagement import EngagementState
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=20, presence_dwell=8)
    assert state.reserve() is None
    state.observe(scene(1))
    # A live detector supplies fresh sequence numbers throughout the dwell.
    for n in range(1, 21):
        clock.now = n
        state.observe(scene(n + 1))
        if n < 20:
            assert state.reserve() is None
    token = state.reserve()
    assert token is not None and state.valid(token)
    state.speech('hello')
    assert not state.valid(token)


def test_stale_or_repeated_scene_cannot_trigger_camera():
    from mechanical_meanderings.engagement import EngagementState
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=1, presence_dwell=1, scene_timeout=3)
    state.observe(scene(1))
    clock.now = 5
    state.observe(scene(1))  # A frozen detector is not a fresh scene.
    assert state.reserve() is None


def test_cooldown_budget_and_sustained_absence():
    from mechanical_meanderings.engagement import EngagementState
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=1, presence_dwell=1,
                            invite_cooldown=10, max_invitations=1, absence_reset=5)
    state.observe(scene(1))
    clock.now = 2
    state.observe(scene(2))
    token = state.reserve()
    assert token is not None
    assert state.reserve() is None  # At most one request, including before model completion.
    assert state.sent(token)
    clock.now = 20
    state.observe(scene(3))
    assert state.reserve() is None  # Budget exhausted for this occupancy.
    state.observe(scene(4, people=0))
    clock.now = 26
    state.observe(scene(5, people=0))
    state.observe(scene(6))
    clock.now = 28
    state.observe(scene(7))
    assert state.reserve() is not None


def test_held_transcript_does_not_keep_microphone_busy_forever():
    from mechanical_meanderings.engagement import EngagementState
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=1, presence_dwell=1)
    state.observe(scene(1))
    state.speech('Hello')
    clock.now = 2
    state.speech('Hello')
    state.speech('')
    state.observe(scene(2))
    assert state.reserve() is not None


def test_pause_reset_vad_and_departure_cancel_inflight_invitation():
    from mechanical_meanderings.engagement import EngagementState
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=1, presence_dwell=1, retry_after=.1)
    state.observe(scene(1))
    clock.now = 2
    state.observe(scene(2))
    token = state.reserve()
    state.set_enabled(False)
    assert not state.valid(token)
    state.set_enabled(True)
    state.activity(True)
    assert state.reserve() is None
    state.activity(False)
    clock.now = 4
    state.observe(scene(3))
    token = state.reserve()
    assert token is not None
    state.observe(scene(4, people=0))
    assert not state.valid(token)
    state.reset()
    assert state.reserve() is None


@pytest.mark.parametrize('raw', ['{}', '{"people":true,"sequence":1}',
    '{"people":-1,"sequence":1}', '{"people":1,"sequence":1,"objects":[{"label":"cup","confidence":NaN}]}'])
def test_invalid_scene_does_not_mutate_state(raw):
    from mechanical_meanderings.engagement import EngagementState
    state = EngagementState()
    with pytest.raises(ValueError):
        state.observe(raw)
    assert state.scene is None
