"""Safe concise inference timing; no invented values for absent Ollama fields."""
import math


def generation_timing(telemetry, queue_wait, request_elapsed, num_predict=None):
    def value(key, scale=1., unit='', digits=3):
        raw = telemetry.get(key)
        if (not isinstance(raw, (int, float)) or isinstance(raw, bool)
                or not math.isfinite(raw) or raw < 0):
            return 'n/a'
        return f'{raw * scale:.{digits}f}{unit}'

    return ' '.join((
        'output_tokens=' + value('eval_count', digits=0),
        f'num_predict_max={num_predict if num_predict is not None else "model-default"}',
        'output_eval=' + value('eval_duration', 1e-9, 's'),
        'output_rate=' + value('eval_tokens_per_second', unit='tok/s', digits=1),
        'input_tokens=' + value('prompt_eval_count', digits=0),
        'input_eval=' + value('prompt_eval_duration', 1e-9, 's'),
        'load=' + value('load_duration', 1e-9, 's'),
        'ollama_total=' + value('total_duration', 1e-9, 's'),
        'http=' + value('http_elapsed', unit='s'),
        f'queue={queue_wait:.3f}s',
        f'request={request_elapsed:.3f}s',
    ))
