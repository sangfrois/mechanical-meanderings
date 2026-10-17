import unittest


class AgentTests(unittest.TestCase):
    def test_chat_contract_and_content_only(self):
        from mechanical_meanderings import Agent
        calls = []
        def transport(url, payload, timeout):
            calls.append((url, payload, timeout))
            return {'message': {'content': '"A brass bird\nopens its wings."', 'thinking': 'SECRET'}}
        agent = Agent(system_prompt='visual instructions', transport=transport)
        self.assertEqual(agent.generate('a bird'), 'A brass bird opens its wings.')
        url, payload, timeout = calls[0]
        self.assertEqual(url, 'http://127.0.0.1:11434/api/chat')
        self.assertEqual(payload, {'model': 'gemma4:12b', 'think': False, 'stream': False,
            'messages': [{'role': 'system', 'content': 'visual instructions'}, {'role': 'user', 'content': 'a bird'}]})
        self.assertGreater(timeout, 0)

    def test_explicit_ollama_urls_are_not_rewritten(self):
        from mechanical_meanderings import Agent
        for url in ('http://localhost:11434', 'https://remote.example:443'):
            calls = []
            def transport(target, payload, timeout):
                calls.append(target)
                return {'message': {'content': 'visual'}}
            agent = Agent('system', transport=transport, ollama_url=url)
            agent.generate('speech')
            self.assertEqual(calls, [url + '/api/chat'])

    def test_safe_output_normalization(self):
        from mechanical_meanderings import Agent
        outputs = iter(['<think>SECRET</think>\n```text\nPrompt: "' + 'word ' * 25 + '"\n```', '', '<think>secret</think>', '<think>unfinished secret', None])
        agent = Agent(system_prompt='x', transport=lambda *args: {'message': {'content': next(outputs)}})
        self.assertEqual(agent.generate('x'), ' '.join(['word'] * 20))
        for _ in range(4):
            with self.assertRaises(ValueError):
                agent.generate('x')

    def test_bounded_successful_history_isolated_by_channel(self):
        from mechanical_meanderings import Agent
        calls = []
        def transport(url, payload, timeout):
            calls.append(payload['messages'])
            if payload['messages'][-1]['content'] == 'fail':
                raise OSError('offline')
            return {'message': {'content': 'result'}}
        agent = Agent(system_prompt='x', transport=transport, history_turns=1)
        agent.generate('first', 'ch1')
        with self.assertRaises(OSError):
            agent.generate('fail', 'ch1')
        agent.generate('second', 'ch1')
        self.assertEqual([m['content'] for m in calls[-1]], ['x', 'first', 'result', 'second'])
        agent.generate('third', 'ch1')
        self.assertEqual([m['content'] for m in calls[-1]], ['x', 'second', 'result', 'third'])
        agent.generate('other', 'ch2')
        self.assertEqual(len(calls[-1]), 2)
        agent = Agent(system_prompt='x', transport=transport, history_turns=0)
        agent.generate('a'); agent.generate('b')
        self.assertEqual(len(calls[-1]), 2)

    def test_reset_invalidates_inflight_and_inactivity(self):
        from mechanical_meanderings import Agent, StaleGeneration
        calls = []
        now = [0]
        def transport(url, payload, timeout):
            calls.append(payload['messages'])
            if payload['messages'][-1]['content'] == 'reset during request':
                agent.reset('ch1')
            return {'message': {'content': 'result'}}
        agent = Agent(system_prompt='x', transport=transport, reset_after=5, clock=lambda: now[0])
        agent.generate('first', 'ch1')
        with self.assertRaises(StaleGeneration):
            agent.generate('reset during request', 'ch1')
        agent.generate('second', 'ch1')
        self.assertEqual(len(calls[-1]), 2)
        now[0] = 6
        agent.generate('third', 'ch1')
        self.assertEqual(len(calls[-1]), 2)
        agent.reset()
        agent.generate('fourth', 'ch1')
        self.assertEqual(len(calls[-1]), 2)


if __name__ == '__main__':
    unittest.main()
