"""Key indexing and hint normalization."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Set


def camel_split(text: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)


def hint_text(text: str) -> str:
    return camel_split(str(text)).replace("\\", " ").replace("/", " ").lower()


def bounded(pattern: str) -> str:
    return r"(?<![a-z0-9])(?:" + pattern + r")(?![a-z0-9])"


def matches_hint(text: str, pattern: str) -> bool:
    return re.search(bounded(pattern), text, re.IGNORECASE) is not None


class KeyIndex:
    """Reusable regex and shape view over a sparse adapter header."""

    _DOWN = re.compile(
        r"(?:lora_down|lora_A(?:[._]default)?|hada_w1_b|lokr_w1_b|dora_down)",
        re.IGNORECASE,
    )

    def __init__(self, header: Mapping[str, Mapping[str, Any]]) -> None:
        self.header = header
        self.keys = list(header)

    @classmethod
    def from_header(cls, header: Mapping[str, Mapping[str, Any]]) -> "KeyIndex":
        return cls(header)

    def has(self, pattern: str) -> bool:
        regex = re.compile(pattern, re.IGNORECASE)
        return any(regex.search(key) for key in self.keys)

    def indices(self, *patterns: str) -> Set[int]:
        result: Set[int] = set()
        compiled = [re.compile(value, re.IGNORECASE) for value in patterns]
        for key in self.keys:
            for regex in compiled:
                match = regex.search(key)
                if match:
                    result.add(int(match.group(1)))
        return result

    def dims(self, key_pattern: str) -> Set[int]:
        regex = re.compile(key_pattern, re.IGNORECASE)
        result: Set[int] = set()
        for key, info in self.header.items():
            if not regex.search(key) or not self._DOWN.search(key):
                continue
            shape = info.get("shape") if isinstance(info, Mapping) else None
            if not isinstance(shape, list) or len(shape) != 2:
                continue
            try:
                result.add(int(shape[-1]))
            except (TypeError, ValueError):
                continue
        return result

    def module_prefix_histogram(self, depth: int = 6) -> List[Dict[str, Any]]:
        counts: Counter[str] = Counter()
        for key in self.keys:
            normalized = re.sub(
                r"(?:[._](?:lora_(?:up|down)|lora_[AB](?:[._]default)?|alpha|dora_scale|hada_[^._]+|lokr_[^._]+))(?:[._]weight)?$",
                "",
                key,
                flags=re.IGNORECASE,
            )
            parts = [value for value in re.split(r"[._]", normalized) if value]
            prefix = ".".join(parts[:depth])
            if prefix:
                counts[prefix] += 1
        return [
            {"prefix": prefix, "count": count}
            for prefix, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]


DetectionContext = KeyIndex


def detect_format(keys: Sequence[str]) -> str:
    lower = [key.lower() for key in keys]
    if any("hada_" in key or "lokr_" in key for key in lower):
        return "lycoris"
    if any(key.startswith(("lora_unet_", "lora_te", "lora_transformer_")) for key in lower):
        return "kohya"
    if any(key.startswith(("unet.", "text_encoder", "transformer.")) for key in lower):
        return "diffusers"
    if any(key.startswith("diffusion_model.") for key in lower):
        return "comfy"
    return "unknown"


_FOLDERABLE_VERSION = re.compile(r"\d{1,3}(?:\.\d{1,3}){0,2}")


def folderable_version(value: str) -> str:
    """Return a version string safe to become a directory qualifier, or "".

    Trainer metadata carries version-shaped noise: kohya float formatting
    ("10.000002"), dimension-like runs ("1.1024.8"), and similar. Minting one
    bucket per float pollutes a sorted library, so only conservative version
    shapes may reach folder names: one to three dot-separated components of
    at most three digits each. The verbatim value stays in the model_version
    record field for review; this gate only guards sort_subfolder.
    """
    cleaned = str(value or "").strip().strip(".")
    while cleaned[:1] in ("v", "V"):
        cleaned = cleaned[1:]
    if _FOLDERABLE_VERSION.fullmatch(cleaned):
        return cleaned
    return ""
