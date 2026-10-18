import re
from dataclasses import dataclass
from typing import Literal

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

# Matches [S1], [S1, S2] and, via findall, each tag in [S1][S2].
_TAG_GROUP = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]")

Status = Literal["answered", "insufficient_evidence", "citation_error"]


@dataclass(frozen=True)
class ParsedAnswer:
    status: Status
    cited: list[int]  # valid source numbers that were cited
    invalid: list[int]  # tag numbers that match no retrieved source


def parse_answer(text: str, source_count: int) -> ParsedAnswer:
    """Classify a model answer. Checks that tags exist, not that claims are true."""
    if INSUFFICIENT in text:
        return ParsedAnswer("insufficient_evidence", [], [])

    numbers: list[int] = []
    for group in _TAG_GROUP.findall(text):
        numbers.extend(int(part.strip()[1:]) for part in group.split(","))

    cited = sorted({n for n in numbers if 1 <= n <= source_count})
    invalid = sorted({n for n in numbers if not 1 <= n <= source_count})
    if invalid or not cited:
        return ParsedAnswer("citation_error", cited, invalid)
    return ParsedAnswer("answered", cited, [])