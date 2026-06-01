"""Utilities for memory extraction and retrieval."""

from __future__ import annotations

import re

from jarvis.memory.memory_types import MemoryCandidate, MemoryType


STOPWORDS = {
    "a",
    "about",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "do",
    "for",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
    "working",
}

IGNORED_MESSAGES = {
    "hello",
    "hi",
    "hey",
    "good morning",
    "good evening",
    "good night",
    "thanks",
    "thank you",
}


def extract_memory_candidates(text: str) -> list[MemoryCandidate]:
    """Extract durable memories from user text using transparent rules.

    This is deliberately deterministic in Phase 2. Later phases can replace or
    augment it with a local classifier.
    """
    cleaned = _clean_sentence(text)
    if not cleaned:
        return []

    lowered = cleaned.lower()
    if lowered in IGNORED_MESSAGES:
        return []
    if "?" in cleaned and not lowered.startswith("remember that "):
        return []

    statement = _remove_memory_prefix(cleaned)
    candidates: list[MemoryCandidate] = []

    candidates.extend(_extract_project(statement, cleaned))
    candidates.extend(_extract_interest(statement, cleaned))
    candidates.extend(_extract_preference(statement, cleaned))
    candidates.extend(_extract_task(statement, cleaned))
    candidates.extend(_extract_fact(statement, cleaned))

    unique: dict[tuple[MemoryType, str], MemoryCandidate] = {}
    for candidate in candidates:
        unique[(candidate.memory_type, candidate.content.lower())] = candidate
    return list(unique.values())


def tokenize(text: str) -> set[str]:
    """Return normalized retrieval terms."""
    terms = {
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_+-]*", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }
    return terms


def infer_requested_memory_types(text: str) -> set[MemoryType]:
    """Infer whether a question is asking for a specific memory category."""
    lowered = text.lower()
    requested: set[MemoryType] = set()

    if any(term in lowered for term in ("project", "projects", "working on", "building")):
        requested.add(MemoryType.PROJECT)
    if any(term in lowered for term in ("interest", "interests", "enjoy", "like", "love")):
        requested.add(MemoryType.INTEREST)
        requested.add(MemoryType.PREFERENCE)
    if any(term in lowered for term in ("preference", "preferences", "favorite", "prefer")):
        requested.add(MemoryType.PREFERENCE)
    if any(term in lowered for term in ("task", "tasks", "remind", "todo", "to do")):
        requested.add(MemoryType.TASK)
    if any(term in lowered for term in ("fact", "facts", "know about me", "remember about me")):
        requested.add(MemoryType.FACT)

    return requested


def _extract_project(statement: str, source_text: str) -> list[MemoryCandidate]:
    patterns = [
        r"^(?:i am|i'm) (?:currently )?(?:building|working on|developing|creating) (?P<topic>.+)$",
        r"^my project is (?P<topic>.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, statement, flags=re.IGNORECASE)
        if match:
            topic = _clean_fragment(match.group("topic"))
            return [
                MemoryCandidate(
                    MemoryType.PROJECT,
                    f"Atharva is building {topic}.",
                    source_text,
                    importance=4,
                )
            ]
    return []


def _extract_interest(statement: str, source_text: str) -> list[MemoryCandidate]:
    match = re.match(
        r"^i (?:love|enjoy|am interested in) (?P<topic>.+)$",
        statement,
        flags=re.IGNORECASE,
    )
    if not match:
        return []

    topic = _clean_fragment(match.group("topic"))
    return [
        MemoryCandidate(
            MemoryType.INTEREST,
            f"Atharva enjoys {topic}.",
            source_text,
            importance=3,
        )
    ]


def _extract_preference(statement: str, source_text: str) -> list[MemoryCandidate]:
    favorite_match = re.match(
        r"^my favorite (?P<category>.+) is (?P<item>.+)$",
        statement,
        flags=re.IGNORECASE,
    )
    if favorite_match:
        category = _clean_fragment(favorite_match.group("category")).lower()
        item = _clean_fragment(favorite_match.group("item"))
        return [
            MemoryCandidate(
                MemoryType.PREFERENCE,
                f"Atharva's favorite {category} is {item}.",
                source_text,
                importance=4,
            )
        ]

    match = re.match(r"^i (?P<verb>like|prefer|dislike|hate) (?P<topic>.+)$", statement, re.IGNORECASE)
    if not match:
        return []

    verb = match.group("verb").lower()
    topic = _clean_fragment(match.group("topic"))
    verb_map = {
        "like": "likes",
        "prefer": "prefers",
        "dislike": "dislikes",
        "hate": "dislikes",
    }
    return [
        MemoryCandidate(
            MemoryType.PREFERENCE,
            f"Atharva {verb_map[verb]} {topic}.",
            source_text,
            importance=3,
        )
    ]


def _extract_task(statement: str, source_text: str) -> list[MemoryCandidate]:
    match = re.match(r"^remind me to (?P<task>.+)$", statement, flags=re.IGNORECASE)
    if not match:
        return []

    task = _clean_fragment(match.group("task"))
    return [
        MemoryCandidate(
            MemoryType.TASK,
            f"Atharva wants to be reminded to {task}.",
            source_text,
            importance=4,
        )
    ]


def _extract_fact(statement: str, source_text: str) -> list[MemoryCandidate]:
    ownership_match = re.match(
        r"^my (?P<thing>.+) (?P<verb>has|is|uses) (?P<value>.+)$",
        statement,
        flags=re.IGNORECASE,
    )
    if ownership_match:
        thing = _clean_fragment(ownership_match.group("thing")).lower()
        verb = ownership_match.group("verb").lower()
        value = _clean_fragment(ownership_match.group("value"))
        return [
            MemoryCandidate(
                MemoryType.FACT,
                f"Atharva's {thing} {verb} {value}.",
                source_text,
                importance=3,
            )
        ]

    have_match = re.match(r"^i have (?P<fact>.+)$", statement, flags=re.IGNORECASE)
    if have_match:
        fact = _clean_fragment(have_match.group("fact"))
        return [
            MemoryCandidate(
                MemoryType.FACT,
                f"Atharva has {fact}.",
                source_text,
                importance=3,
            )
        ]

    return []


def _clean_sentence(text: str) -> str:
    return " ".join(text.strip().split()).strip()


def _remove_memory_prefix(text: str) -> str:
    return re.sub(r"^remember that\s+", "", text, flags=re.IGNORECASE)


def _clean_fragment(text: str) -> str:
    return text.strip().strip(" .!?;:")
