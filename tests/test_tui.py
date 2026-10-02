import asyncio
import json


def test_pilot_controls_manual_trajectory_and_lifecycle(tmp_path):
    from mechanical_meanderings import Agent
    from mechanical_meanderings.bridge import Bridge
    from mechanical_meanderings.live import Session
    from mechanical_meanderings.tui import MeanderingsApp
    from textual.widgets import Input, RichLog, Static
    calls = []
    def transport(url, payload, timeout):
        calls.append(payload)
        return {'message': {'content': 'A glass bird folds into rain.'},
                'eval_count': 8, 'eval_duration': 400000000}
    session = Session(tmp_path)
    bridge = Bridge(Agent('visual', transport=transport), listen_port=0, dynamic=True)
    app = MeanderingsApp(bridge, session)
    async def exercise():
        async with app.run_test(size=(130, 44)) as pilot:
            await pilot.pause()
            app.query_one('#temperature', Input).value = '1.2'
            app.query_one('#num_ctx', Input).value = '4096'
            await pilot.press('ctrl+a')
            await pilot.pause()
            assert bridge.settings.temperature == 1.2
            assert bridge.settings.num_ctx == 4096
            app.query_one('#num_ctx', Input).value = '-5'
            app.action_apply()
            assert bridge.settings.num_ctx == 4096
            app.query_one('#manual', Input).value = 'a glass bird'
            app.action_send()
            for _ in range(30):
                await pilot.pause(.05)
                if session.counts['result']:
                    break
            assert session.counts['result'] == 1
            assert calls[0]['options']['temperature'] == 1.2
            assert 'unavailable' in str(app.query_one('#metrics', Static).content)
            assert app.query_one('#trajectory', RichLog).max_lines == 300
            await pilot.press('ctrl+p')
            assert bridge.paused
            await pilot.press('ctrl+r')
            assert not bridge.agent.history
            assert session.counts['reset'] == 1
    asyncio.run(exercise())
    assert not bridge.worker.is_alive()
    assert not bridge.receiver.is_alive()
    assert session.file.closed
    rows = [json.loads(line) for line in session.path.read_text(encoding='utf-8').splitlines()]
    assert rows[-1]['kind'] == 'session_end'
