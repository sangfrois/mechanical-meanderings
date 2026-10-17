import pytest

from mechanical_meanderings import Agent
from mechanical_meanderings.live import Settings
from mechanical_meanderings.timing import generation_timing


def test_actual_counts_and_duration_attribution_not_token_cap():
    telemetry = {}
    agent = Agent('system', transport=lambda *args: {
        'message': {'content': 'visual'}, 'eval_count': 24,
        'eval_duration': 300_000_000, 'prompt_eval_count': 1500,
        'prompt_eval_duration': 1_500_000_000, 'load_duration': 200_000_000,
        'total_duration': 2_500_000_000})
    assert agent.generate('speech', settings=Settings(num_predict=96), telemetry=telemetry) == 'visual'
    text = generation_timing(telemetry, .125, 3.0, 96)
    assert 'output_tokens=24 num_predict_max=96 output_eval=0.300s output_rate=80.0tok/s' in text
    assert 'input_tokens=1500 input_eval=1.500s load=0.200s ollama_total=2.500s' in text
    assert 'queue=0.125s request=3.000s' in text
    assert telemetry['http_elapsed'] >= 0
    assert 'http=n/a' not in text


@pytest.mark.parametrize('telemetry', [{}, {
    'eval_count': True, 'eval_duration': float('nan'),
    'eval_tokens_per_second': float('inf'), 'load_duration': -1,
    'http_elapsed': 'private response body', 'total_duration': None,
}])
def test_absent_or_invalid_metrics_safe_and_not_fabricated(telemetry):
    text = generation_timing(telemetry, 0, 1)
    for field in ('output_tokens', 'output_eval', 'output_rate', 'input_eval', 'load', 'ollama_total', 'http'):
        assert f'{field}=n/a' in text
    assert 'num_predict_max=model-default' in text
    assert 'private' not in text


def test_missing_response_metrics_still_measure_http_and_zero_eval_rate_safe():
    agent = Agent('system', transport=lambda *args: {
        'message': {'content': 'visual', 'thinking': 'PRIVATE'},
        'eval_count': 0, 'eval_duration': 0, 'load_duration': 'PRIVATE'})
    telemetry = {}
    agent.generate('speech', telemetry=telemetry)
    assert telemetry['eval_tokens_per_second'] is None
    assert telemetry['http_elapsed'] >= 0
    assert 'PRIVATE' not in generation_timing(telemetry, 0, 1)


def test_transport_error_still_captures_http_time_without_response_metrics():
    def fail(*args):
        raise TimeoutError('private transport error')
    agent = Agent('system', transport=fail)
    telemetry = {}
    with pytest.raises(TimeoutError):
        agent.generate('speech', telemetry=telemetry)
    assert telemetry['http_elapsed'] >= 0
    text = generation_timing(telemetry, 0, 1)
    assert 'output_tokens=n/a' in text and 'http=n/a' not in text
    assert 'private' not in text
