import re
from abc import ABC, abstractmethod

TURN_PATTERN = re.compile(r"^(d|p):\s*(.*)$", re.IGNORECASE)


class CondenserModule(ABC):
    """Base class for all transcript fluff-removal (condenser) modules."""

    @abstractmethod
    def condense(self, transcript):
        """Removes non-clinically-relevant content from a transcript.

        Returns (condensed_transcript, elapsed).
        """
        raise NotImplementedError


def split_turns(transcript):
    """Splits a d:/p: tagged transcript into a list of (speaker, text) tuples."""
    turns = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        match = TURN_PATTERN.match(line)
        if match:
            turns.append((match.group(1).lower(), match.group(2)))
        else:
            turns.append((None, line))
    return turns


def join_turns(turns):
    """Rejoins (speaker, text) tuples back into d:/p: tagged transcript text."""
    lines = [f"{speaker}: {text}" if speaker else text for speaker, text in turns]
    return "\n".join(lines)
