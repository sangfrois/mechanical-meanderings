"""Opt-in real Ollama + isolated UDP + headless Textual smoke; no audio or TD."""
import asyncio
import json
from pathlib import Path
import socket
import time
import xml.etree.ElementTree as ET
from importlib.resources import files
from urllib.request import urlopen

from pythonosc.osc_packet import OscPacket
from pythonosc.udp_client import SimpleUDPClient
from textual.widgets import Input, Static
from mechanical_meanderings import Agent, http_chat
from mechanical_meanderings.bridge import Bridge
from mechanical_meanderings.live import Session
from mechanical_meanderings.tui import MeanderingsApp


async def main():
    directory = Path('sessions/tui-smoke')
    directory.mkdir(parents=True, exist_ok=True)
    with urlopen('http://localhost:11434/api/tags', timeout=10) as response:
        models = [m['name'] for m in json.load(response)['models']]
    assert 'gemma4:12b' in models, models
    requests = []

    def observed_chat(url, payload, timeout):
        # This is the real HTTP transport, not a fixture. Do not save reasoning.
        requests.append({'model': payload['model'], 'options': payload.get('options'),
                         'message_count': len(payload['messages']),
                         'input': payload['messages'][-1]['content']})
        return http_chat(url, payload, timeout)

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', 0))
    receiver.setblocking(False)
    session = Session(directory)
    agent = Agent(files('mechanical_meanderings').joinpath('system.txt').read_text(encoding='utf-8'),
                  transport=observed_chat, timeout=90)
    bridge = Bridge(agent, listen_host='127.0.0.1', listen_port=0,
                    output_host='127.0.0.1', output_port=receiver.getsockname()[1],
                    output_address='/txt2img', dynamic=True)
    bridge.configure(temperature=.4, history=1, num_ctx=4096, num_predict=96, debounce=.05, interval=.1)
    sender = SimpleUDPClient(*bridge.address)
    app = MeanderingsApp(bridge, session)
    packets = []

    async def wait_for(kind, count):
        deadline = time.monotonic() + 100
        while session.counts[kind] < count:
            assert time.monotonic() < deadline, session.metrics()
            assert not session.counts['error'], session.metrics()
            await asyncio.sleep(.05)

    try:
        async with app.run_test(size=(150, 50)) as pilot:
            await pilot.pause()
            sender.send_message('/transcription', 'A blue glass forest reflected in rain, woodcut texture.')
            await wait_for('start', 1)
            app.query_one('#temperature', Input).value = '1.2'
            app.query_one('#history', Input).value = '2'
            await pilot.press('ctrl+a')
            await wait_for('result', 1)
            app.query_one('#manual', Input).value = 'Keep the forest, let copper birds unfold from its branches.'
            app.action_send()
            await wait_for('result', 2)
            await pilot.pause(.3)
            while True:
                try:
                    data, address = receiver.recvfrom(65535)
                except BlockingIOError:
                    break
                for timed in OscPacket(data).messages:
                    packets.append({'address': timed.message.address, 'args': timed.message.params})
            assert [p['address'] for p in packets] == ['/txt2img', '/txt2img/trigger'] * 2
            assert requests[0]['options']['temperature'] == .4
            assert requests[1]['options']['temperature'] == 1.2
            assert requests[1]['message_count'] == 4
            assert 'unavailable' in str(app.query_one('#metrics', Static).content)
            screenshot = directory / 'tui-real.svg'
            screenshot.write_text(app.export_screenshot(), encoding='utf-8')
            assert ET.parse(screenshot).getroot().tag.endswith('svg')
            await pilot.press('ctrl+p')
            assert bridge.paused
            await pilot.press('ctrl+r')
            assert not agent.history
            await pilot.press('ctrl+q')
    finally:
        bridge.close()
        session.close()
        sender.close()
        receiver.close()
    rows = [json.loads(line) for line in session.path.read_text(encoding='utf-8').splitlines()]
    results = [row for row in rows if row['kind'] == 'result']
    assert len(results) == 2 and rows[-1]['kind'] == 'session_end'
    assert [row['output'] for row in results] == [p['args'][0] for p in packets if p['address'] == '/txt2img']
    assert not bridge.worker.is_alive() and not bridge.receiver.is_alive()
    assert session.file.closed and bridge.server.socket.fileno() == -1
    evidence = dict(model='gemma4:12b', real_http=True, input_endpoint=list(bridge.address),
                    output_endpoint=bridge.output_target, requests=requests, results=results,
                    received_udp=packets, journal=str(session.path), screenshot=str(screenshot),
                    worker_stopped=True, receiver_stopped=True, sockets_closed=True,
                    limitations='No microphone, TouchDesigner, rendered frames, or downstream latency measured.')
    (directory / 'evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(evidence, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    asyncio.run(main())
