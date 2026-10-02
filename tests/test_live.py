import threading


def test_settings_snapshot_survives_inflight_tuning():
    from mechanical_meanderings import Agent
    from mechanical_meanderings.live import Settings
    entered, release = threading.Event(), threading.Event()
    calls = []
    def transport(url, payload, timeout):
        calls.append(payload)
        entered.set()
        release.wait(2)
        return {'message': {'content': 'brass bird'}, 'eval_count': 4, 'eval_duration': 200000000}
    agent = Agent('visual', transport=transport)
    old = Settings(temperature=0.4, history=1)
    details = {}
    worker = threading.Thread(target=lambda: agent.generate('bird', settings=old, telemetry=details))
    worker.start()
    assert entered.wait(1)
    new = old.updated(temperature=1.4, history=0)
    release.set()
    worker.join(2)
    assert calls[0]['options']['temperature'] == .4
    assert details['eval_tokens_per_second'] == 20
    agent.generate('forest', settings=new)
    assert len(calls[1]['messages']) == 2
    assert agent.history.get('default', []) == []


def test_bridge_journal_snapshot_pause_recovery_and_bounds(tmp_path):
    import json
    import time
    from mechanical_meanderings import Agent
    from mechanical_meanderings.bridge import Bridge
    from mechanical_meanderings.live import Session
    entered, release = threading.Event(), threading.Event()
    def transport(url, payload, timeout):
        entered.set()
        release.wait(2)
        if payload['messages'][-1]['content'] == 'fail':
            raise OSError('offline')
        return {'message': {'content': 'liquid bronze'}}
    session = Session(tmp_path, capacity=3)
    bridge = Bridge(Agent('visual', transport=transport), listen_port=0, on_event=session.record)
    try:
        bridge.start()
        bridge.configure(temperature=.4, history=1)
        bridge.submit('bird')
        assert entered.wait(1)
        bridge.configure(temperature=1.4, history=0)
        release.set()
        deadline = time.monotonic() + 3
        while session.counts['result'] < 1 and time.monotonic() < deadline:
            time.sleep(.02)
        bridge.set_paused(True)
        bridge.submit('fail')
        time.sleep(.15)
        assert session.counts['error'] == 0
        bridge.set_paused(False)
        deadline = time.monotonic() + 3
        while session.counts['error'] < 1 and time.monotonic() < deadline:
            time.sleep(.02)
        bridge.submit('recover')
        deadline = time.monotonic() + 3
        while session.counts['result'] < 2 and time.monotonic() < deadline:
            time.sleep(.02)
        assert session.counts['error'] == 1
        assert session.counts['result'] == 2
        assert session.events.qsize() <= 3
        assert session.dropped > 0
        rows = [json.loads(line) for line in session.path.read_text(encoding='utf-8').splitlines()]
        results = [row for row in rows if row['kind'] == 'result']
        assert results[0]['settings']['temperature'] == .4
        assert results[1]['settings']['temperature'] == 1.4
        assert results[0]['udp_sent'] is True
        assert results[0]['elapsed'] > 0
        assert results[0]['input'] == 'bird'
    finally:
        release.set()
        bridge.close()
        session.close()


def test_rejects_oversized_output_and_nonfinite_telemetry():
    from mechanical_meanderings import Agent
    import pytest
    agent = Agent('visual', transport=lambda *a: {'message': {'content': 'x' * 9000}})
    with pytest.raises(ValueError):
        agent.generate('bird')
    details = {}
    agent.transport = lambda *a: {'message': {'content': 'bird'}, 'eval_duration': float('inf')}
    agent.generate('bird', telemetry=details)
    assert 'eval_duration' not in details


def test_session_disk_failure_is_visible_and_queue_capacity_validated(tmp_path):
    from mechanical_meanderings.live import Session
    import pytest
    with pytest.raises(ValueError):
        Session(tmp_path, capacity=0)
    session = Session(tmp_path)
    session.file.close()
    session.record({'kind': 'input', 'input': 'bird'})
    assert session.metrics()['disk_error'] == 'ValueError'
    assert session.counts['input'] == 1


def test_bounds_and_udp_failure_recover_without_crashing(tmp_path):
    import time
    from mechanical_meanderings import Agent
    from mechanical_meanderings.bridge import Bridge
    from mechanical_meanderings.live import Session
    session = Session(tmp_path)
    bridge = Bridge(Agent('visual', transport=lambda *a: {'message': {'content': 'bird'}}),
                    listen_port=0, channel_mode=True, on_event=session.record)
    try:
        for i in range(bridge.MAX_CHANNELS + 5):
            bridge.receive(f'/ch{i}/transcription', 'bird')
        assert len(bridge.epochs) == bridge.MAX_CHANNELS
        assert len(bridge.latest) == bridge.MAX_CHANNELS
        bridge.receive('/ch0/transcription', 'x' * (bridge.MAX_TEXT + 1))
        assert bridge.latest['ch0'] == 'bird'
        bridge.client.close()
        bridge.start()
        bridge.submit('bird', 'ch0')
        deadline = time.monotonic() + 2
        while not session.counts['error'] and time.monotonic() < deadline:
            time.sleep(.02)
        assert session.counts['error'] == 1
        assert bridge.worker.is_alive()
        assert session.counts['result'] == 0
    finally:
        bridge.close()
        session.close()


def test_settings_validation_is_atomic():
    from mechanical_meanderings.live import Settings
    import pytest
    old = Settings()
    with pytest.raises(ValueError):
        old.updated(temperature=.1, num_ctx=-1)
    assert old.temperature is None
    with pytest.raises(ValueError):
        old.updated(temperature=float('nan'))


def test_observer_failure_does_not_leak_exception_text(caplog):
    from mechanical_meanderings import Agent
    from mechanical_meanderings.bridge import Bridge
    def broken_observer(event):
        raise ValueError('PRIVATE_RESPONSE_BODY')
    bridge = Bridge(Agent('visual'), listen_port=0, on_event=broken_observer)
    try:
        bridge.emit('input', input='bird')
        assert 'PRIVATE_RESPONSE_BODY' not in caplog.text
        assert 'ValueError' in caplog.text
    finally:
        bridge.close()


def test_display_consumer_can_drain_queue_during_eviction(tmp_path):
    from mechanical_meanderings.live import Session
    session = Session(tmp_path, capacity=1)
    original_full = session.events.full
    def concurrent_drain():
        full = original_full()
        if full:
            session.events.get_nowait()
        return full
    session.events.full = concurrent_drain
    try:
        session.record({'kind': 'input', 'input': 'bird'})
        assert session.events.get_nowait()['input'] == 'bird'
        assert session.dropped == 0
    finally:
        session.close()
