"""Structured metadata and filename hint extraction.

Every field is parsed independently. This is deliberate: a token in one metadata field
must not combine with a token in another field to manufacture a model identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Tuple

from .models import Evidence, EvidenceCategory, EvidenceSource
from .normalization import hint_text, matches_hint

SCOPE_ANY = "any"
SCOPE_IDENTITY = "identity"


@dataclass(frozen=True)
class ArchitectureHintRule:
    rule_id: str
    pattern: str
    arch: str
    scope: str = SCOPE_ANY
    precise: bool = True


ARCHITECTURE_RULES: Tuple[ArchitectureHintRule, ...] = (
    ArchitectureHintRule("hint.pony.v7", r"pony(?:[ ._-]*diffusion)?[ ._-]*(?:v|version)?[ ._-]*7", "AuraFlow"),
    ArchitectureHintRule("hint.auraflow", r"aura[ ._-]*flow|auraflow(?:transformer2dmodel)?", "AuraFlow"),
    ArchitectureHintRule("hint.pony.v6.xl", r"pony(?:[ ._-]*diffusion)?[ ._-]*(?:v|version)?[ ._-]*6[ ._-]*(?:xl|sdxl)", "SDXL"),
    ArchitectureHintRule("hint.pony.v6.sd15", r"pony(?:[ ._-]*diffusion)?[ ._-]*(?:v|version)?[ ._-]*6[ ._-]*(?:1[ ._-]*5|sd15)", "SD15"),
    ArchitectureHintRule("hint.pony.xl", r"pony(?:[ ._-]*diffusion)?[ ._-]*xl|pd[ ._-]*xl", "SDXL"),
    ArchitectureHintRule("hint.minimax.h3", r"mini[ ._-]*max[ ._-]*h3|minimaxh3(?:model|ditmodel)?", "MiniMaxH3"),
    ArchitectureHintRule("hint.ideogram4", r"ideogram[ ._-]*(?:v[ ._-]*)?4|ideogram4transformer2dmodel", "Ideogram4"),
    ArchitectureHintRule("hint.zimage", r"z[ ._-]*image|zimagetransformer2dmodel", "ZImage"),
    ArchitectureHintRule("hint.ernie_image", r"ernie[ ._-]*image|ernieimagetransformer2dmodel", "ErnieImage"),
    ArchitectureHintRule(
        "hint.anima.qualified",
        r"anima(?!gine)[ ._-]*(?:base|turbo|aesthetic)|"
        r"anima(?!gine)[ ._-]*(?:v|version)[ ._-]*[0-9][0-9 ._-]*",
        "Anima",
    ),
    ArchitectureHintRule("hint.anima.identity", r"anima(?!gine)", "Anima", SCOPE_IDENTITY),
    ArchitectureHintRule("hint.flux3", r"flux[ ._-]*3(?:[ ._-]*image)?|flux3transformer2dmodel", "Flux3"),
    ArchitectureHintRule("hint.moss-image", r"moss[ ._-]*image|mossimagetransformer", "MOSSImage"),
    ArchitectureHintRule("hint.fibo2", r"fibo[ ._-]*2(?:[ ._-]*0)?", "FIBO2"),
    ArchitectureHintRule("hint.fibo.identity", r"fibo(?![ ._-]*2)", "FIBO", SCOPE_IDENTITY),
    ArchitectureHintRule("hint.flux2", r"flux[ ._-]*2|flux2", "Flux2"),
    ArchitectureHintRule("hint.flux2.klein", r"flux[ ._-]*(?:2[ ._-]*)?klein", "Flux2"),
    ArchitectureHintRule("hint.chroma.identity", r"chroma", "Chroma", SCOPE_IDENTITY),
    ArchitectureHintRule("hint.longcat.image", r"long[ ._-]*cat[ ._-]*image|longcatimagetransformer2dmodel", "LongCat"),
    ArchitectureHintRule("hint.longcat.identity", r"long[ ._-]*cat", "LongCat", SCOPE_IDENTITY),
    ArchitectureHintRule("hint.krea2", r"krea[ ._-]*2|krea2transformer2dmodel", "Krea2"),
    ArchitectureHintRule(
        "hint.flux1",
        r"flux[ ._-]*1|flux1|flux[ ._-]*(?:dev|schnell|kontext)|"
        r"flux[ ._-]*transformer2[ ._-]*d[ ._-]*model",
        "Flux1",
    ),
    ArchitectureHintRule("hint.illustrious", r"illustrious(?:[ ._-]*xl)?|il[ ._-]*xl|illust[ ._-]*xl", "SDXL"),
    ArchitectureHintRule("hint.sdxl", r"stable[ ._-]*diffusion[ ._-]*xl|sd[ ._-]*xl|sdxl", "SDXL"),
    ArchitectureHintRule("hint.sd3", r"stable[ ._-]*diffusion[ ._-]*(?:v)?3(?:[ ._-]*5)?|sd3(?:[ ._-]*5)?", "SD3"),
    ArchitectureHintRule("hint.sd2", r"stable[ ._-]*diffusion[ ._-]*(?:v)?2|sd[ ._-]*v?2|sd2", "SD2"),
    ArchitectureHintRule("hint.sd15", r"stable[ ._-]*diffusion[ ._-]*(?:v)?1(?:[ ._-]*5)?|sd[ ._-]*1[ ._-]*5|sd15", "SD15"),
    ArchitectureHintRule("hint.hunyuan_video", r"hunyuan[ ._-]*video|hunyuanvideo|hyvideo", "Hunyuan"),
    ArchitectureHintRule("hint.bernini", r"bernini(?:[ ._-]*r)?", "Wan"),
    ArchitectureHintRule(
        "hint.wan.qualified",
        r"wan[ ._-]*2(?:[ ._-]*[12])?|wan[ ._-]*(?:video|vace|fun|animate|i2v|t2v|ti2v|s2v|flf2v)",
        "Wan",
    ),
    ArchitectureHintRule("hint.wan.identity", r"wan", "Wan", SCOPE_IDENTITY),
    ArchitectureHintRule("hint.ltx", r"ltx(?:[ ._-]*(?:video|2))?", "LTX"),
    ArchitectureHintRule("hint.qwen_image", r"qwen[ ._-]*image|qwenimagetransformer2dmodel", "QwenImage"),
)

ECOSYSTEM_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("tag.ecosystem.pony", "Pony", r"pony(?!tail)(?:[ ._-]*diffusion)?|pd[ ._-]*xl"),
    ("tag.ecosystem.illustrious", "Illustrious", r"illustrious(?:[ ._-]*xl)?|il[ ._-]*xl|ilxl|illust"),
    ("tag.ecosystem.noobai", "NoobAI", r"noob(?:[ ._-]*ai)?"),
    ("tag.ecosystem.animagine", "Animagine", r"animagine"),
)

VARIANT_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("tag.variant.bernini_r", "Bernini-R", r"bernini(?:[ ._-]*r)?"),
    ("tag.variant.zimage_omni_base", "Omni-Base", r"z[ ._-]*image[ ._-]*omni[ ._-]*base"),
    ("tag.variant.base", "Base", r"(?:z[ ._-]*image|anima)[ ._-]*(?:base|foundation|undistilled)"),
    ("tag.variant.turbo", "Turbo", r"(?:z[ ._-]*image|anima|ernie[ ._-]*image)[ ._-]*turbo"),
    ("tag.variant.anima_aesthetic", "Aesthetic", r"anima[ ._-]*aesthetic"),
    ("tag.variant.ernie_aes", "Aes", r"ernie[ ._-]*image[ ._-]*aes"),
    ("tag.variant.zimage_edit", "Edit", r"z[ ._-]*image[ ._-]*edit"),
    ("tag.variant.flux_kontext", "Kontext", r"flux[ ._-]*(?:1[ ._-]*)?(?:dev[ ._-]*)?kontext"),
    ("tag.variant.flux_schnell", "Schnell", r"flux[ ._-]*(?:1[ ._-]*)?schnell"),
    ("tag.variant.flux2_klein", "Klein", r"flux[ ._-]*(?:2[ ._-]*)?klein"),
)

UNRESOLVED_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("hint.unresolved.bernie", "Bernie (ambiguous: Bernini or ERNIE-Image)", r"bernie"),
    ("hint.unresolved.minimax", "MiniMax (family unspecified)", r"mini[ ._-]*max(?![ ._-]*h3)|minimax(?![ ._-]*h3)"),
    ("hint.unresolved.qwen", "Qwen (family unspecified; Qwen-Image must be explicit)", r"qwen(?![ ._-]*image)"),
)

TECHNIQUE_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("tag.technique.lcm", "LCM", r"lcm"),
    ("tag.technique.turbo", "Turbo", r"turbo"),
    ("tag.technique.lightning", "Lightning", r"lightning"),
    ("tag.technique.hyper", "Hyper", r"hyper(?:[ ._-]*sd)?"),
    ("tag.technique.dmd", "DMD", r"dmd2?"),
)

DEMOTED_FILENAME_TOKENS: Tuple[Tuple[str, str, str], ...] = (
    ("Chroma", "Chroma", r"chroma"),
    ("Wan", "Wan", r"wan"),
    ("Anima", "Anima", r"anima(?!gine)"),
    ("LongCat", "LongCat", r"long[ ._-]*cat"),
    ("Klein", "Flux2", r"klein"),
    ("Schnell", "Flux1", r"schnell"),
    ("Kontext", "Flux1", r"kontext"),
)

# field, confidence, identity field, priority
METADATA_FIELDS: Tuple[Tuple[str, str, bool, int], ...] = (
    ("modelspec.architecture", "high", True, 90),
    ("architecture", "high", True, 90),
    ("model_type", "high", True, 88),
    ("ss_base_model_version", "high", True, 88),
    ("ss_base_model", "high", True, 87),
    ("base_model", "high", True, 87),
    ("base_model_name", "high", True, 87),
    ("base_model_name_or_path", "high", True, 87),
    ("pretrained_model_name_or_path", "high", True, 87),
    ("ss_sd_model_name", "medium", True, 70),
    ("modelspec.title", "medium", False, 55),
)

TASK_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("tag.task.ref2va", "Ref2VA", r"ref2[ ._-]*va|reference[ ._-]*to[ ._-]*video[ ._-]*(?:and|with)?[ ._-]*audio"),
    ("tag.task.fl2va", "FL2VA", r"fl2[ ._-]*va|first[ ._-]*last[ ._-]*frame[ ._-]*to[ ._-]*video[ ._-]*(?:and|with)?[ ._-]*audio"),
    ("tag.task.t2va", "T2VA", r"t2[ ._-]*va|text[ ._-]*to[ ._-]*video[ ._-]*(?:and|with)?[ ._-]*audio"),
    ("tag.task.ti2v", "TI2V", r"ti2[ ._-]*v|text[ ._-]*image[ ._-]*to[ ._-]*video"),
    ("tag.task.i2v", "I2V", r"i2[ ._-]*v|image[ ._-]*to[ ._-]*video"),
    ("tag.task.t2v", "T2V", r"t2[ ._-]*v|text[ ._-]*to[ ._-]*video"),
    ("tag.task.v2v", "V2V", r"v2[ ._-]*v|video[ ._-]*to[ ._-]*video"),
    ("tag.task.i2i", "I2I", r"i2[ ._-]*i|image[ ._-]*to[ ._-]*image"),
    ("tag.task.t2i", "T2I", r"t2[ ._-]*i|text[ ._-]*to[ ._-]*image"),
    ("tag.task.edit", "Edit", r"edit"),
    ("tag.task.animate", "Animate", r"animate"),
)


def _evidence(
    rule_id: str,
    source: str,
    category: str,
    value: str,
    confidence: str,
    priority: int,
    detail: str,
    field: str = "",
    precise: bool = False,
) -> Evidence:
    return Evidence(
        rule_id=rule_id,
        source=source,
        category=category,
        value=value,
        confidence=confidence,
        priority=priority,
        detail=detail,
        field=field,
        precise=precise,
    )


def _architecture_evidence(
    normalized: str,
    source: str,
    field: str,
    identity: bool,
    confidence: str,
    priority: int,
) -> List[Evidence]:
    for rule in ARCHITECTURE_RULES:
        if rule.scope == SCOPE_IDENTITY and not identity:
            continue
        if matches_hint(normalized, rule.pattern):
            return [
                _evidence(
                    rule.rule_id,
                    source,
                    EvidenceCategory.ARCHITECTURE.value,
                    rule.arch,
                    confidence,
                    priority,
                    "%s identifies %s: %s" % (field or source, rule.arch, normalized[:120]),
                    field=field,
                    precise=rule.precise,
                )
            ]
    return []


def _version_evidence(
    normalized: str,
    source: str,
    field: str,
    confidence: str,
    priority: int,
) -> List[Evidence]:
    value = ""
    rule_id = ""
    if matches_hint(normalized, r"pony(?:[ ._-]*diffusion)?[ ._-]*(?:v|version)?[ ._-]*7"):
        value, rule_id = "7", "tag.version.pony7"
    elif matches_hint(normalized, r"pony(?:[ ._-]*diffusion)?[ ._-]*(?:v|version)?[ ._-]*6"):
        value, rule_id = "6", "tag.version.pony6"
    elif matches_hint(normalized, r"bernini(?:[ ._-]*r)?"):
        value, rule_id = "2.2", "tag.version.bernini22"
    else:
        anima = re.search(
            r"(?<![a-z0-9])anima(?!gine)(?:[ ._-]*(?:base|turbo|aesthetic))?"
            r"(?:[ ._-]*(?:v|version))?[ ._-]*([0-9]+(?:[ ._-][0-9]+)+)",
            normalized,
        )
        if anima:
            value = re.sub(r"[ _-]", ".", anima.group(1))
            rule_id = "tag.version.anima"
    if not value and re.search(r"(?<![a-z0-9])wan[ ._-]*2[ ._-]*2(?![a-z0-9])", normalized):
        value, rule_id = "2.2", "tag.version.wan22"
    elif not value and re.search(r"(?<![a-z0-9])wan[ ._-]*2[ ._-]*1(?![a-z0-9])", normalized):
        value, rule_id = "2.1", "tag.version.wan21"
    elif not value and re.search(r"(?<![a-z0-9])sd3[ ._-]*5(?![a-z0-9])", normalized):
        value, rule_id = "3.5", "tag.version.sd35"
    if not value:
        return []
    return [
        _evidence(
            rule_id,
            source,
            EvidenceCategory.MODEL_VERSION.value,
            value,
            confidence,
            priority - 3,
            "%s indicates model version %s" % (field or source, value),
            field=field,
            precise=True,
        )
    ]


def extract_text_evidence(
    text: str,
    *,
    source: str,
    field: str = "",
    identity: bool = False,
    confidence: str = "low",
    priority: int = 20,
) -> List[Evidence]:
    normalized = hint_text(text)
    if not normalized.strip():
        return []
    evidence: List[Evidence] = []
    evidence.extend(
        _architecture_evidence(normalized, source, field, identity, confidence, priority)
    )

    for rule_id, value, pattern in ECOSYSTEM_RULES:
        if matches_hint(normalized, pattern):
            evidence.append(
                _evidence(
                    rule_id,
                    source,
                    EvidenceCategory.ECOSYSTEM.value,
                    value,
                    confidence,
                    priority - 5,
                    "%s tags ecosystem %s" % (field or source, value),
                    field=field,
                    precise=True,
                )
            )
    for rule_id, value, pattern in VARIANT_RULES:
        if matches_hint(normalized, pattern):
            evidence.append(
                _evidence(
                    rule_id,
                    source,
                    EvidenceCategory.MODEL_VARIANT.value,
                    value,
                    confidence,
                    priority - 6,
                    "%s tags model variant %s" % (field or source, value),
                    field=field,
                    precise=True,
                )
            )
    for rule_id, value, pattern in UNRESOLVED_RULES:
        if matches_hint(normalized, pattern):
            evidence.append(
                _evidence(
                    rule_id,
                    source,
                    EvidenceCategory.WARNING.value,
                    value,
                    "low",
                    5,
                    "unresolved name hint: %s" % value,
                    field=field,
                )
            )
    for rule_id, value, pattern in TECHNIQUE_RULES:
        if matches_hint(normalized, pattern):
            evidence.append(
                _evidence(
                    rule_id,
                    source,
                    EvidenceCategory.TECHNIQUE.value,
                    value,
                    confidence,
                    priority - 10,
                    "%s tags technique %s" % (field or source, value),
                    field=field,
                    precise=False,
                )
            )

    evidence.extend(_version_evidence(normalized, source, field, confidence, priority))
    for rule_id, value, pattern in TASK_RULES:
        if matches_hint(normalized, pattern):
            evidence.append(
                _evidence(
                    rule_id,
                    source,
                    EvidenceCategory.TASK_VARIANT.value,
                    value,
                    confidence,
                    priority - 4,
                    "%s indicates task %s" % (field or source, value),
                    field=field,
                    precise=True,
                )
            )
            break
    if matches_hint(normalized, r"high[ ._-]*noise"):
        evidence.append(
            _evidence(
                "tag.noise.high",
                source,
                EvidenceCategory.NOISE_STAGE.value,
                "high",
                confidence,
                priority - 4,
                "%s indicates high-noise expert" % (field or source),
                field=field,
                precise=True,
            )
        )
    elif matches_hint(normalized, r"low[ ._-]*noise"):
        evidence.append(
            _evidence(
                "tag.noise.low",
                source,
                EvidenceCategory.NOISE_STAGE.value,
                "low",
                confidence,
                priority - 4,
                "%s indicates low-noise expert" % (field or source),
                field=field,
                precise=True,
            )
        )
    return evidence


def metadata_evidence(metadata: Mapping[str, Any]) -> List[Evidence]:
    result: List[Evidence] = []
    for field, confidence, identity, priority in METADATA_FIELDS:
        raw = metadata.get(field, "")
        value = str(raw or "").strip()
        if not value:
            continue
        result.extend(
            extract_text_evidence(
                value,
                source=EvidenceSource.METADATA.value,
                field=field,
                identity=identity,
                confidence=confidence,
                priority=priority,
            )
        )

    if str(metadata.get("ss_v2", "")).strip().lower() in {"true", "1", "yes"}:
        result.append(
            _evidence(
                "metadata.ss_v2",
                EvidenceSource.METADATA.value,
                EvidenceCategory.ARCHITECTURE.value,
                "SD2",
                "high",
                85,
                "metadata ss_v2=True identifies SD2",
                field="ss_v2",
                precise=True,
            )
        )

    # The official foundation checkpoint is named simply Tongyi-MAI/Z-Image.
    # This inference is restricted to one identity field at a time.
    for field in (
        "ss_base_model_version",
        "ss_base_model",
        "base_model",
        "base_model_name",
        "base_model_name_or_path",
        "pretrained_model_name_or_path",
        "ss_sd_model_name",
    ):
        value = str(metadata.get(field, "") or "").strip().rstrip("/\\")
        if not value:
            continue
        leaf = re.split(r"[/\\]", value)[-1]
        leaf = re.sub(r"\.(?:safetensors|ckpt|pt|pth|bin)$", "", leaf, flags=re.IGNORECASE)
        if re.fullmatch(r"z[ _-]*image", leaf, re.IGNORECASE):
            result.append(
                _evidence(
                    "metadata.zimage.foundation",
                    EvidenceSource.METADATA.value,
                    EvidenceCategory.MODEL_VARIANT.value,
                    "Base",
                    "high" if field != "ss_sd_model_name" else "medium",
                    82,
                    "%s identifies the Z-Image foundation/Base checkpoint" % field,
                    field=field,
                    precise=True,
                )
            )
            break
    return _dedupe_evidence(result)


def filename_evidence(filename: str) -> List[Evidence]:
    stem = Path(filename).stem
    result = extract_text_evidence(
        stem,
        source=EvidenceSource.FILENAME.value,
        field="filename",
        identity=False,
        confidence="low",
        priority=20,
    )
    architecture = next(
        (value.value for value in result if value.category == EvidenceCategory.ARCHITECTURE.value),
        "",
    )
    normalized = hint_text(stem)
    for label, implied_arch, pattern in DEMOTED_FILENAME_TOKENS:
        if architecture != implied_arch and matches_hint(normalized, pattern):
            result.append(
                _evidence(
                    "filename.demoted.%s" % label.lower(),
                    EvidenceSource.FILENAME.value,
                    EvidenceCategory.WARNING.value,
                    "%s (unscoped)" % label,
                    "low",
                    3,
                    "filename token %r alone is not precise enough to classify" % label,
                    field="filename",
                )
            )
    return _dedupe_evidence(result)


def _dedupe_evidence(values: Iterable[Evidence]) -> List[Evidence]:
    seen = set()
    result = []
    for value in values:
        marker = (
            value.rule_id,
            value.source,
            value.field,
            value.category,
            value.value,
            value.detail,
        )
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result
