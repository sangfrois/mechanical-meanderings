"""Bounded model-selected caption design, shared by replies and invitations."""
import math
import logging

FONTS = ('Verdana', 'Georgia', 'Consolas', 'Impact', 'Arial')
ALIGNMENTS = ('left', 'center', 'right')
DECORATIONS = ('none', 'tilde', 'swung', 'slashes')
BOUNDS = dict(size=(18, 64), x=(0, 1), y=(0, 1), tracking=(-2, 8), line_width=(12, 60))


def default_typography():
    # Compatibility fallback, not a sequence of authored/rotating type treatments.
    return dict(font='Verdana', size=34, x=.5, y=.77, align='center', italic=False,
                bold=False, tracking=0, color=[1., .96, .86], line_width=36, decoration='none')


TYPOGRAPHY_SCHEMA = dict(type='object', additionalProperties=False, properties=dict(
    font=dict(type='string', enum=list(FONTS)),
    **{key: dict(type='integer' if key == 'line_width' else 'number', minimum=lo, maximum=hi)
       for key, (lo, hi) in BOUNDS.items()},
    align=dict(type='string', enum=list(ALIGNMENTS)), italic=dict(type='boolean'), bold=dict(type='boolean'),
    color=dict(type='array', minItems=3, maxItems=3, items=dict(type='number', minimum=0, maximum=1)),
    decoration=dict(type='string', enum=list(DECORATIONS))), required=list(default_typography()))

TYPOGRAPHY_INSTRUCTION = """
Choose typography deliberately for THIS caption, not a preset rotation. Return typography:
font (Verdana, Georgia, Consolas, Impact, Arial), size (18..64 pixels at 1280x720),
x,y (0..1 measured from TOP-LEFT, locating the caption's CENTRE anchor),
align (left, center, right; text alignment, not an alternate anchor), italic and bold (booleans),
tracking (-2..8 pixels), color ([R,G,B], each 0..1), line_width (12..60 integer characters),
RGB is NORMALIZED: white=[1,1,1], never [255,255,255].
and decoration (none, tilde, swung, slashes). Keep the whole caption legible and on screen;
prefer x=.15..85, y=.15..85 and modest sizes for longer text. Adapt placement, type, spacing
and colour to the meaning and rhythm. Replies may use playful ~, ⁓, / and // in their text.
A typography decoration does not require repeating that ornament in the text.
"""


def parse_typography(value=None):
    """Validate without coercing booleans or silently clipping a malformed design.

    Omitted fields use neutral defaults for older model responses/Decision callers.
    """
    result = default_typography()
    if value is None:
        return result
    if not isinstance(value, dict) or set(value) - set(result):
        raise ValueError('invalid typography object')
    result.update(value)
    for key, choices in (('font', FONTS), ('align', ALIGNMENTS), ('decoration', DECORATIONS)):
        if not isinstance(result[key], str) or result[key] not in choices:
            raise ValueError(f'invalid typography {key}')
    for key, (lo, hi) in BOUNDS.items():
        n = result[key]
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or not lo <= n <= hi:
            raise ValueError(f'invalid typography {key}')
    if not isinstance(result['line_width'], int):
        raise ValueError('typography line_width must be an integer')
    for key in ('italic', 'bold'):
        if not isinstance(result[key], bool):
            raise ValueError(f'typography {key} must be boolean')
    color = result['color']
    if not isinstance(color, list) or len(color) != 3:
        raise ValueError('typography color must have three components')
    if any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1
           for c in color):
        raise ValueError('invalid typography color')
    result['color'] = list(color)
    return result


def model_typography(value):
    """Bad optional design data must not discard an otherwise valid reply.

    Keep individually validated choices and default only malformed fields.
    Strict wire validation remains unchanged: paths, expressions, arbitrary
    fonts and out-of-bounds numbers cannot escape this adapter.
    """
    try:
        return parse_typography(value)
    except ValueError:
        result = default_typography()
        if isinstance(value, dict):
            for key in result:
                if key in value:
                    try:
                        result[key] = parse_typography({key: value[key]})[key]
                    except ValueError:
                        logging.getLogger(__name__).warning('Ignored malformed caption field: %s', key)
        return result
