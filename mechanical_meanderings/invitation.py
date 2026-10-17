"""Operator-owned invitation text: full replacement, reread for each request."""
from dataclasses import dataclass
import hashlib
from importlib.resources import files
from pathlib import Path


KINDS = ('nudge', 'embodied', 'invitation', 'aside')
MAX_PROMPT_BYTES = 65536


class InvitationPromptError(ValueError):
    """Do not silently replace a broken operator prompt with a different voice."""


class InvitationRejected(ValueError):
    """Safe diagnostics for successful model responses rejected by our validator.

    Only the intended visible message is retained, never a provider error body,
    prompt or hidden reasoning. Oversized candidates are explicitly abbreviated.
    """
    def __init__(self, code, reason, *, kind, candidate=None):
        super().__init__(reason)
        self.code, self.reason, self.kind = code, reason, kind
        self.message_type = type(candidate).__name__
        text = ' '.join(candidate.split()) if isinstance(candidate, str) else None
        self.words = len(text.split()) if text is not None else None
        self.characters = len(text) if text is not None else None
        self.candidate = text[:512] if text is not None else None
        self.candidate_truncated = text is not None and len(text) > 512

    def metadata(self):
        return dict(code=self.code, reason=self.reason, kind=self.kind,
            words=self.words, characters=self.characters, message_type=self.message_type,
            candidate=self.candidate, candidate_truncated=self.candidate_truncated)


@dataclass(frozen=True)
class PromptSnapshot:
    text: str
    source: str
    sha256: str

    def metadata(self):
        return dict(source=self.source, sha256=self.sha256)


class InvitationPrompt:
    """One editable source; no appended Python personality or output instructions.

    The schema/context are sent separately from this exact system message.
    Reading on demand handles editor atomic-replace saves and preserves the
    current agent, speech history, model and sampling settings.
    """
    def __init__(self, path=None):
        self.source = (Path(path).expanduser().resolve() if path is not None else
                       files('mechanical_meanderings').joinpath('prompts', 'invite.md'))

    def read(self):
        source = str(self.source)
        try:
            with self.source.open('rb') as stream:
                raw = stream.read(MAX_PROMPT_BYTES + 1)
        except OSError as exc:
            raise InvitationPromptError(f'Cannot read invitation prompt: {source}') from exc
        if len(raw) > MAX_PROMPT_BYTES:
            raise InvitationPromptError(f'Invitation prompt exceeds {MAX_PROMPT_BYTES} bytes: {source}')
        try:
            text = raw.decode('utf-8-sig').strip()
        except UnicodeError as exc:
            raise InvitationPromptError(f'Invitation prompt must be UTF-8: {source}') from exc
        if not text:
            raise InvitationPromptError(f'Invitation prompt is empty: {source}')
        return PromptSnapshot(text=text, source=source,
                              sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
