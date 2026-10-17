"""Narrow spoken caption controls; never an image-engine stop command."""
import re

REPLY_MODES = ('sparse', 'auto', 'off')

# Match a whole imperative, not quoted speech, a substring, or an imagined scene.
_MUTE = re.compile(
    r"(?:please[, ]+)?(?:"
    r"(?:don't|don’t|do not) (?:talk back|reply|respond|speak)(?: to me| anymore| again)?|"
    r"stop (?:talking|speaking|replying)(?: to me)?|be quiet|shut up|"
    r"tais[- ]toi|taisez[- ]vous|ta gueule|arrête de (?:parler|répondre)|"
    r"ne (?:parle|réponds) (?:plus|pas)(?: à moi)?|"
    r"(?:parle|réponds) (?:plus|pas)"
    r")(?:[, ]+(?:please|s'il te plaît|s’il te plaît|svp))?", re.I)
_UNMUTE = re.compile(
    r"(?:please[, ]+)?(?:talk back|speak again|reply again|"
    r"you can (?:talk|speak|reply)(?: again)?|"
    r"tu peux (?:parler|répondre|reparler)(?: maintenant)?|"
    r"parle[- ]moi|réponds[- ]moi|recommence à (?:parler|répondre)"
    r")(?:[, ]+(?:please|s'il te plaît|s’il te plaît|svp))?", re.I)
_GREETINGS = {'hello', 'hi', 'hey', 'salut', 'bonjour', 'bonsoir', 'allo', 'allô'}


def speech_directive(text):
    """Return True to mute, False to resume, or None for ordinary speech."""
    if not isinstance(text, str):
        return None
    clean = ' '.join(text.split()).strip().rstrip('.!?,;:').casefold()
    if _MUTE.fullmatch(clean):
        return True
    if _UNMUTE.fullmatch(clean):
        return False
    return None


def permit_reply(mode, muted, text, decision):
    """A sparse caption answers real questions, not each visual update or greeting."""
    if muted or mode == 'off':
        return False
    if mode == 'auto':
        return True  # Existing parser still enforces direct-address confidence.
    greeting = ' '.join(text.split()).rstrip('.!?,;:').casefold() in _GREETINGS
    return not greeting and decision.category in ('question', 'conversation')
