"""Composable structural signature registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

from ..models import SignatureMatch
from ..normalization import DetectionContext, KeyIndex

Detector = Callable[[DetectionContext], Iterable[SignatureMatch]]
_REGISTERED: List[Tuple[int, str, Detector]] = []


def register(name: str, order: int = 100):
    def decorator(detector: Detector) -> Detector:
        if any(existing == name for _, existing, _ in _REGISTERED):
            raise ValueError(f"duplicate structural detector name: {name}")
        _REGISTERED.append((order, name, detector))
        _REGISTERED.sort(key=lambda item: (item[0], item[1]))
        return detector
    return decorator


@dataclass(frozen=True)
class RegisteredDetector:
    order: int
    name: str
    detector: Detector


class SignatureRegistry:
    def __init__(self, entries: Sequence[Tuple[int, str, Detector]]) -> None:
        self._entries = tuple(RegisteredDetector(*entry) for entry in entries)

    @property
    def detector_names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self._entries)

    def evaluate(self, index: KeyIndex) -> List[SignatureMatch]:
        matches: List[SignatureMatch] = []
        for entry in self._entries:
            try:
                matches.extend(entry.detector(index))
            except Exception as exc:
                matches.append(SignatureMatch(
                    rule_id=f"detector.error.{entry.name}",
                    family="Unknown",
                    confidence="low",
                    priority=-100,
                    ambiguous=True,
                    message=f"structural detector {entry.name} failed: {exc}",
                ))
        return matches

    def summary(self) -> List[dict]:
        return [
            {"order": item.order, "name": item.name, "callable": item.detector.__name__}
            for item in self._entries
        ]


def build_default_registry() -> SignatureRegistry:
    # Importing builtin performs registration exactly once through Python's module cache.
    from . import builtin  # noqa: F401
    return SignatureRegistry(tuple(_REGISTERED))
