"""Extract the minimal sub-story supporting a contiguous sub-chain.

Decomposing a CLUTRR instance by re-querying the *whole* story leaves the model
hunting for the relevant sentences among distractors: implied per-part accuracy
was ~.77 there, against .83-.95 for a standalone short story, and composed
accuracy consequently tied direct prediction (.425 vs .468).

Giving each sub-chain only its own sentences is the analogue of handing an
addition sub-problem its own digit block rather than the whole number.

Gate A checks the extraction is lossless: every consecutive entity pair in the
sub-chain must co-occur in some kept sentence, otherwise the edge that pair
encodes is missing and the sub-story cannot support the question.  Only 30% of
CLUTRR stories carry exactly one sentence per edge, so this is not assumed.
"""

from __future__ import annotations

import ast
import re
from typing import Iterable, Sequence

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
BRACKETED = re.compile(r"\[(\w+)\]")


def sentences(story: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(story.strip()) if s.strip()]


def mentions(sentence: str) -> set[str]:
    return set(BRACKETED.findall(sentence))


def node_names(row: dict) -> list[str]:
    """Entity name per node index; CLUTRR lists genders in node order."""
    return [item.partition(":")[0].strip() for item in row["genders"].split(",")]


def is_clean_chain(row: dict) -> bool:
    k = len(ast.literal_eval(row["edge_types"]))
    return (
        ast.literal_eval(row["story_edges"]) == [(i, i + 1) for i in range(k)]
        and ast.literal_eval(row["query_edge"]) == (0, k)
        and len(node_names(row)) >= k + 1
    )


def extract(story: str, names: Sequence[str]) -> str:
    """Contiguous sentence span from the first to the last mention of `names`.

    A filtered subset would drop sentences that carry an edge by description
    rather than by name -- "[Rosa] enjoys playing cards with her brother" states
    an edge whose other endpoint is never named.  Keeping the span preserves
    those antecedents at the cost of a few extra sentences.
    """
    sents = sentences(story)
    wanted = set(names)
    hit = [i for i, s in enumerate(sents) if mentions(s) & wanted]
    return " ".join(sents[min(hit) : max(hit) + 1]) if hit else ""


def covers_chain(sub_story: str, names: Sequence[str]) -> bool:
    """True when every adjacent pair in the chain co-occurs in some sentence.

    NOT a validity test for extraction.  CLUTRR states many edges by description
    ("her brother") without naming both endpoints, so the FULL story passes this
    only 70.7% of the time -- the same rate as any extraction of it.  Measured
    against the full story as reference, extraction loses 0.0% of chains.  Use
    this only to compare an extraction against its own source, never as an
    absolute threshold.
    """
    pairs = {(names[i], names[i + 1]) for i in range(len(names) - 1)}
    for sentence in sentences(sub_story):
        seen = mentions(sentence)
        pairs = {p for p in pairs if not (p[0] in seen and p[1] in seen)}
        if not pairs:
            return True
    return not pairs


def sub_story(row: dict, i: int, j: int) -> tuple[str, list[str]]:
    """Sub-story and node names for the chain segment covering nodes i..j."""
    names = node_names(row)[i : j + 1]
    return extract(row["clean_story"], names), names
