"""Healthy-progress/default-budget regressions; no live Ollama or installation OSC."""
import queue
import threading
import time
from unittest.mock import patch

import pytest

from mechanical_meanderings import Agent
from mechanical_meanderings.__main__ import main
from mechanical_meanderings.boundary import BoundaryBridge
from mechanical_meanderings.final_turn import FinalTurnBridge
from mechanical_meanderings.interactive import InteractiveBridge
from mechanical_meanderings.mediation import MediatingAgent
from test_interactive_boundary import rig, raw


@pytest.mark.parametrize('bridge_type', [FinalTurnBridge, BoundaryBridge, InteractiveBridge])
@pytest.mark.parametrize('timeout', [120, 17])
def test_unset_budget_inherits_existing_agent_timeout(bridge_type, timeout):
    agent = MediatingAgent('style', timeout=timeout) if bridge_type is InteractiveBridge else Agent('style', timeout=timeout)
    options = {'frame_port': 0} if bridge_type is InteractiveBridge else {}
    bridge = bridge_type(agent, listen_host='127.0.0.1', listen_port=0, **options)
    try:
        assert bridge.generation_budget == timeout
    finally:
        bridge.close()


@pytest.mark.parametrize('budget', [None, 5])
def test_cli_inherits_timeout_but_preserves_explicit_budget(budget):
    seen = []

    class Panel:
        def __init__(self, bridge, session):
            seen.append(bridge)
        def run(self):
            pass

    argv = ['--tui', '--interactive', '--listen-host', '127.0.0.1',
            '--listen-port', '0', '--frame-port', '0', '--timeout', '17']
    if budget is not None:
        argv.extend(['--generation-budget', str(budget)])
    with patch('mechanical_meanderings.tui.MeanderingsApp', Panel), \
            patch('mechanical_meanderings.__main__.Session'):
        assert main(argv) == 0
    assert seen[0].generation_budget == (17 if budget is None else budget)


def test_default_accepts_six_seconds_of_progress_without_blocking_raw_udp():
    with rig(progress_seconds=6) as (bridge, sender, requests, packets, events, _):
        raw(sender, packets, '/transcription', 'complete speech')
        raw(sender, packets, '/trigger', 1)
        requests.get(timeout=2)
        # Leave the same request current. Redraws must still forward immediately.
        for _ in range(3):
            raw(sender, packets, '/transcription', 'complete speech')
            time.sleep(.2)
        deadline = time.monotonic() + 8
        while True:
            result = events.get(timeout=max(.001, deadline - time.monotonic()))
            if result['kind'] in ('result', 'error'):
                break
        assert result['kind'] == 'result', result
        assert result['telemetry']['generation_status'] == 'complete'
        assert 6 <= result['telemetry']['http_elapsed'] < 8
        assert bridge.generation_budget == bridge.agent.timeout == 120
        assert packets.get(timeout=2)[0] == '/mediation'
        assert packets.get(timeout=2)[0] == '/txt2img'
        assert packets.get(timeout=2) == ('/txt2img/trigger', (1,))
        assert packets.empty()
        assert bridge.agent.history['default'][-2]['content'] == 'complete speech'
        assert not any(t.name == 'ollama-deadline' for t in threading.enumerate())
        print(f'healthy progressive interactive HTTP: {result["telemetry"]["http_elapsed"]:.3f}s; '
              f'default budget={bridge.generation_budget}s; raw UDP concurrent; one accepted visual')
