"""Evidence reconciliation for exact architecture and compatibility qualifiers."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .constants import CONFIDENCE_RANK
from .hints import filename_evidence, metadata_evidence
from .models import (
    Classification,
    Conflict,
    ConflictSeverity,
    Evidence,
    EvidenceCategory,
    EvidenceSource,
    SignatureMatch,
)
from .normalization import KeyIndex, detect_format, folderable_version
from .signatures import build_default_registry
from .utils import join_semicolon, unique

ARCH_FAMILY = {
    "SD15": "SDUNet",
    "SD2": "SDUNet",
    "SDXL": "SDUNet",
    "AuraFlow": "AuraFlowDiT",
    "MiniMaxH3": "MiniMaxH3OmniDiT",
    "Krea2": "Krea2DiT",
    "Anima": "CosmosPredict2DiT",
    "ErnieImage": "ErnieImageDiT",
    "ZImage": "ZImageDiT",
    "Ideogram4": "Ideogram4DiT",
    "Flux1": "DualSingleStreamDiT",
    "Flux2": "DualSingleStreamDiT",
    "Chroma": "DualSingleStreamDiT",
    "LongCat": "DualSingleStreamDiT",
    "Hunyuan": "DualSingleStreamDiT",
    "Wan": "WanLikeVideoDiT",
    "LTX": "DualAttentionTransformerDiT",
    "SD3": "JointAttentionDiT",
    "QwenImage": "JointAttentionDiT",
}

FAMILY_COMPATIBILITY = {
    "Unknown": set(ARCH_FAMILY),
    "TextEncoderOnly": set(ARCH_FAMILY),
    "TransformerDiT": set(ARCH_FAMILY),
    "SingleStreamSwiGLUDiT": {"ZImage", "Ideogram4"},
    "JointAttentionDiT": {"SD3", "QwenImage"},
    "DualSingleStreamDiT": {"Flux1", "Flux2", "Chroma", "LongCat", "Hunyuan"},
    "DualAttentionTransformerDiT": {"LTX", "Anima"},
    "CosmosPredict2DiT": {"Anima"},
    "MiniMaxH3OmniDiT": {"MiniMaxH3"},
    "AuraFlowDiT": {"AuraFlow"},
    "Krea2DiT": {"Krea2"},
    "ZImageDiT": {"ZImage"},
    "Ideogram4DiT": {"Ideogram4"},
    "ErnieImageDiT": {"ErnieImage"},
    "WanLikeVideoDiT": {"Wan"},
    "SDUNet": {"SD15", "SD2", "SDXL"},
}

ECOSYSTEM_ALLOWED_ARCHES = {
    "Pony": {"SD15", "SDXL", "AuraFlow"},
    "Illustrious": {"SDXL"},
    "NoobAI": {"SDXL"},
    "Animagine": {"SDXL"},
}

ECOSYSTEM_IMPLIES = {
    "Illustrious": "SDXL",
    "NoobAI": "SDXL",
    "Animagine": "SDXL",
}

VARIANT_IMPLIES = {
    "Bernini-R": "Wan",
    "Kontext": "Flux1",
    "Schnell": "Flux1",
    "Klein": "Flux2",
}

SINGLETON_CATEGORIES = (
    EvidenceCategory.MODEL_VERSION.value,
    EvidenceCategory.TASK_VARIANT.value,
    EvidenceCategory.NOISE_STAGE.value,
    EvidenceCategory.MODEL_VARIANT.value,
)


class ClassificationEngine:
    def __init__(self) -> None:
        self.registry = build_default_registry()

    def classify(
        self,
        tensors: Mapping[str, Mapping[str, Any]],
        metadata: Mapping[str, Any],
        filename: str,
    ) -> Classification:
        index = KeyIndex(tensors)
        matches = self.registry.evaluate(index)
        structural_evidence: List[Evidence] = []
        for match in matches:
            structural_evidence.extend(match.evidence())

        family, structural_arch, structural_decisive, structural_conflicts = self._resolve_structure(matches)
        hints = metadata_evidence(metadata) + filename_evidence(filename)
        implied = self._implied_architecture_evidence(hints)
        evidence = self._dedupe_evidence(structural_evidence + hints + implied)

        conflicts: List[Conflict] = list(structural_conflicts)
        architecture_evidence = [
            value for value in evidence if value.category == EvidenceCategory.ARCHITECTURE.value
        ]
        conflicts.extend(self._architecture_conflicts(architecture_evidence, structural_arch))

        qualifier_values: Dict[str, str] = {}
        qualifier_conflicts: List[Conflict] = []
        for category in SINGLETON_CATEGORIES:
            chosen, category_conflicts = self._resolve_singleton(category, evidence)
            qualifier_values[category] = chosen
            qualifier_conflicts.extend(category_conflicts)
        conflicts.extend(qualifier_conflicts)

        ecosystems = self._values(evidence, EvidenceCategory.ECOSYSTEM.value)
        techniques = self._values(evidence, EvidenceCategory.TECHNIQUE.value)
        unresolved = [
            value.value for value in evidence if value.rule_id.startswith("hint.unresolved.")
        ]
        warnings = [
            value.detail
            for value in evidence
            if value.category == EvidenceCategory.WARNING.value
            and not value.rule_id.startswith("hint.unresolved.")
        ]

        if len(ecosystems) > 1:
            conflicts.append(
                Conflict(
                    conflict_id="qualifier.ecosystem.multiple",
                    category=EvidenceCategory.ECOSYSTEM.value,
                    values=tuple(ecosystems),
                    severity=ConflictSeverity.HARD.value,
                    detail="multiple compatibility ecosystems were asserted: %s"
                    % ", ".join(ecosystems),
                    evidence_rule_ids=tuple(
                        value.rule_id
                        for value in evidence
                        if value.category == EvidenceCategory.ECOSYSTEM.value
                    ),
                )
            )

        candidate, confidence = self._choose_architecture(
            architecture_evidence, structural_arch
        )
        conflicts.extend(self._family_conflicts(family, candidate, architecture_evidence))
        conflicts.extend(self._ecosystem_conflicts(ecosystems, candidate, architecture_evidence))
        conflicts = self._dedupe_conflicts(conflicts)

        model_variant = qualifier_values[EvidenceCategory.MODEL_VARIANT.value]
        if model_variant == "Omni-Base":
            # The generic Base tag is intentionally subsumed by the more specific variant.
            pass

        if model_variant == "Turbo" and candidate in {"ZImage", "Anima", "ErnieImage"}:
            techniques = [value for value in techniques if value != "Turbo"]
            evidence.append(
                Evidence(
                    rule_id="resolution.turbo_model_variant",
                    source=EvidenceSource.SYSTEM.value,
                    category=EvidenceCategory.WARNING.value,
                    value="Turbo",
                    confidence="high",
                    priority=0,
                    detail="Turbo interpreted as a %s checkpoint variant, not duplicated as a technique" % candidate,
                )
            )

        hard_conflict = any(value.severity == ConflictSeverity.HARD.value for value in conflicts)
        candidate_sources = [value for value in architecture_evidence if value.value == candidate]
        metadata_candidate = any(value.source == EvidenceSource.METADATA.value for value in candidate_sources)
        only_filename = bool(candidate_sources) and all(
            value.source in {EvidenceSource.FILENAME.value, EvidenceSource.IMPLIED.value}
            and (value.source != EvidenceSource.IMPLIED.value or value.field == "filename")
            for value in candidate_sources
        )

        if hard_conflict:
            status = "conflict"
        elif candidate == "Unknown":
            status = "ambiguous"
        elif only_filename:
            status = "ambiguous"
            warnings.append("filename-only architecture hint requires review")
        elif structural_arch == candidate and not structural_decisive and not metadata_candidate:
            status = "ambiguous"
            warnings.append("structural signature is shared; metadata confirmation required")
        else:
            status = "ok"

        model_version = qualifier_values[EvidenceCategory.MODEL_VERSION.value]
        if model_version and not folderable_version(model_version):
            warnings.append(
                "model version %r is not folder-safe and was omitted from sort_subfolder"
                % model_version
            )
        task_variant = qualifier_values[EvidenceCategory.TASK_VARIANT.value]
        noise_stage = qualifier_values[EvidenceCategory.NOISE_STAGE.value]
        sort_subfolder = self._sort_subfolder(
            candidate=candidate,
            ecosystems=ecosystems,
            model_variant=model_variant,
            model_version=model_version,
            task_variant=task_variant,
            noise_stage=noise_stage,
            techniques=techniques,
        )

        return Classification(
            arch=candidate,
            structural_family=family,
            model_version=model_version,
            task_variant=task_variant,
            noise_stage=noise_stage,
            confidence=confidence,
            status=status,
            ecosystems=ecosystems,
            model_variants=[model_variant] if model_variant else [],
            unresolved_hints=unique(unresolved),
            techniques=techniques,
            sort_subfolder=sort_subfolder,
            format=detect_format(index.keys),
            evidence=self._dedupe_evidence(evidence),
            conflicts=conflicts,
            warnings=unique(warnings),
        )

    def _resolve_structure(
        self, matches: Sequence[SignatureMatch]
    ) -> Tuple[str, str, bool, List[Conflict]]:
        if not matches:
            return "Unknown", "", False, []
        exact = [value for value in matches if value.arch]
        exact_arches = unique(value.arch for value in exact if value.arch)
        conflicts: List[Conflict] = []
        if len(exact_arches) > 1:
            conflicts.append(
                Conflict(
                    conflict_id="structure.multiple_exact_architectures",
                    category=EvidenceCategory.ARCHITECTURE.value,
                    values=tuple(exact_arches),
                    severity=ConflictSeverity.HARD.value,
                    detail="multiple structural signatures claim incompatible architectures: %s"
                    % ", ".join(exact_arches),
                    evidence_rule_ids=tuple(value.rule_id for value in exact),
                )
            )
        chosen = max(
            exact or list(matches),
            key=lambda value: (CONFIDENCE_RANK.get(value.confidence, 0), value.priority),
        )
        if exact:
            chosen_exact = max(
                exact,
                key=lambda value: (CONFIDENCE_RANK.get(value.confidence, 0), value.priority),
            )
            return (
                chosen_exact.family,
                chosen_exact.arch or "",
                chosen_exact.decisive,
                conflicts,
            )
        return chosen.family, "", False, conflicts

    def _implied_architecture_evidence(self, evidence: Sequence[Evidence]) -> List[Evidence]:
        result: List[Evidence] = []
        versions = self._values(evidence, EvidenceCategory.MODEL_VERSION.value)
        version = versions[0] if len(versions) == 1 else ""
        for item in evidence:
            arch = ""
            detail = ""
            if item.category == EvidenceCategory.ECOSYSTEM.value:
                if item.value == "Pony" and version == "7":
                    arch, detail = "AuraFlow", "Pony V7 targets AuraFlow"
                else:
                    arch = ECOSYSTEM_IMPLIES.get(item.value, "")
                    if arch:
                        detail = "%s normally targets %s" % (item.value, arch)
            elif item.category == EvidenceCategory.MODEL_VARIANT.value:
                arch = VARIANT_IMPLIES.get(item.value, "")
                if arch:
                    detail = "%s targets %s" % (item.value, arch)
            if not arch:
                continue
            confidence = "medium" if item.source == EvidenceSource.METADATA.value else "low"
            priority = 58 if item.source == EvidenceSource.METADATA.value else 16
            result.append(
                Evidence(
                    rule_id="implied.%s" % item.rule_id,
                    source=EvidenceSource.IMPLIED.value,
                    category=EvidenceCategory.ARCHITECTURE.value,
                    value=arch,
                    confidence=confidence,
                    priority=priority,
                    detail=detail,
                    field=item.field,
                    precise=item.precise,
                )
            )
        return result

    def _architecture_conflicts(
        self,
        evidence: Sequence[Evidence],
        structural_arch: str,
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []
        values = unique(item.value for item in evidence)
        if len(values) <= 1:
            return conflicts
        authoritative = [
            item
            for item in evidence
            if item.source in {EvidenceSource.STRUCTURE.value, EvidenceSource.METADATA.value}
            or item.precise
        ]
        authoritative_values = unique(item.value for item in authoritative)
        if len(authoritative_values) > 1:
            conflicts.append(
                Conflict(
                    conflict_id="architecture.evidence_disagrees",
                    category=EvidenceCategory.ARCHITECTURE.value,
                    values=tuple(authoritative_values),
                    severity=ConflictSeverity.HARD.value,
                    detail="architecture evidence disagrees: %s"
                    % ", ".join(
                        "%s%s=%s"
                        % (item.source, (":" + item.field) if item.field else "", item.value)
                        for item in sorted(authoritative, key=lambda value: -value.priority)
                    ),
                    evidence_rule_ids=tuple(item.rule_id for item in authoritative),
                )
            )
        return conflicts

    def _family_conflicts(
        self,
        family: str,
        candidate: str,
        architecture_evidence: Sequence[Evidence],
    ) -> List[Conflict]:
        if not candidate or candidate == "Unknown":
            return []
        allowed = FAMILY_COMPATIBILITY.get(family)
        if allowed is None or candidate in allowed:
            return []
        # A low-priority generic family should never overrule an exact metadata claim.
        family_items = [
            item
            for item in architecture_evidence
            if item.source == EvidenceSource.STRUCTURE.value
        ]
        if family in {"Unknown", "TransformerDiT", "TextEncoderOnly"}:
            return []
        return [
            Conflict(
                conflict_id="architecture.structural_family_incompatible",
                category=EvidenceCategory.STRUCTURAL_FAMILY.value,
                values=(family, candidate),
                severity=ConflictSeverity.HARD.value,
                detail="structural family %s is incompatible with architecture %s" % (family, candidate),
                evidence_rule_ids=tuple(item.rule_id for item in family_items),
            )
        ]

    def _ecosystem_conflicts(
        self,
        ecosystems: Sequence[str],
        candidate: str,
        architecture_evidence: Sequence[Evidence],
    ) -> List[Conflict]:
        result: List[Conflict] = []
        if not candidate or candidate == "Unknown":
            return result
        for ecosystem in ecosystems:
            allowed = ECOSYSTEM_ALLOWED_ARCHES.get(ecosystem)
            if allowed and candidate not in allowed:
                result.append(
                    Conflict(
                        conflict_id="ecosystem.%s.incompatible" % ecosystem.lower(),
                        category=EvidenceCategory.ECOSYSTEM.value,
                        values=(ecosystem, candidate),
                        severity=ConflictSeverity.HARD.value,
                        detail="ecosystem %s supports %s, not %s"
                        % (ecosystem, ",".join(sorted(allowed)), candidate),
                        evidence_rule_ids=tuple(
                            item.rule_id
                            for item in architecture_evidence
                            if item.value == candidate
                        ),
                    )
                )
        return result

    def _resolve_singleton(
        self, category: str, evidence: Sequence[Evidence]
    ) -> Tuple[str, List[Conflict]]:
        items = [value for value in evidence if value.category == category]
        if not items:
            return "", []
        values = unique(value.value for value in items)
        # Omni-Base is a more specific form of Base, not a disagreement.
        if category == EvidenceCategory.MODEL_VARIANT.value and set(values) == {"Base", "Omni-Base"}:
            return "Omni-Base", []
        chosen = max(
            items,
            key=lambda value: (CONFIDENCE_RANK.get(value.confidence, 0), value.priority),
        )
        if len(values) == 1:
            return chosen.value, []
        conflict = Conflict(
            conflict_id="qualifier.%s.disagrees" % category,
            category=category,
            values=tuple(values),
            severity=ConflictSeverity.HARD.value,
            detail="%s evidence disagrees: %s"
            % (
                category,
                ", ".join(
                    "%s%s=%s"
                    % (item.source, (":" + item.field) if item.field else "", item.value)
                    for item in sorted(items, key=lambda value: -value.priority)
                ),
            ),
            evidence_rule_ids=tuple(item.rule_id for item in items),
        )
        return chosen.value, [conflict]

    def _choose_architecture(
        self,
        evidence: Sequence[Evidence],
        structural_arch: str,
    ) -> Tuple[str, str]:
        if not evidence:
            return "Unknown", "low"
        ordered = sorted(
            evidence,
            key=lambda value: (CONFIDENCE_RANK.get(value.confidence, 0), value.priority),
            reverse=True,
        )
        if structural_arch:
            structural = [
                value
                for value in ordered
                if value.source == EvidenceSource.STRUCTURE.value and value.value == structural_arch
            ]
            if structural:
                chosen = max(structural, key=lambda value: value.priority)
                return structural_arch, chosen.confidence
        chosen = ordered[0]
        confidence = chosen.confidence
        agreeing_high = any(
            value.value == chosen.value and value.confidence == "high" for value in evidence
        )
        if agreeing_high:
            confidence = "high"
        return chosen.value, confidence

    def _values(self, evidence: Sequence[Evidence], category: str) -> List[str]:
        values = [value.value for value in evidence if value.category == category and value.value]
        return unique(values)

    def _sort_subfolder(
        self,
        *,
        candidate: str,
        ecosystems: Sequence[str],
        model_variant: str,
        model_version: str,
        task_variant: str,
        noise_stage: str,
        techniques: Sequence[str],
    ) -> str:
        base = ecosystems[0] if ecosystems else model_variant
        qualifiers: List[str] = []
        safe_version = folderable_version(model_version)
        if safe_version:
            qualifiers.append("v" + safe_version)
        if candidate in {"Wan", "LTX", "QwenImage", "Krea2", "MiniMaxH3"}:
            if task_variant:
                qualifiers.append(task_variant)
            if noise_stage:
                qualifiers.append(noise_stage + "-noise")
        if base and qualifiers:
            return base + "__" + "+".join(qualifiers)
        if base:
            return base
        if qualifiers:
            return "+".join(qualifiers)
        if techniques:
            return "+".join(techniques)
        return ""

    def _dedupe_evidence(self, evidence: Sequence[Evidence]) -> List[Evidence]:
        seen = set()
        result = []
        for item in evidence:
            marker = (
                item.rule_id,
                item.source,
                item.field,
                item.category,
                item.value,
                item.detail,
            )
            if marker not in seen:
                seen.add(marker)
                result.append(item)
        return result

    def _dedupe_conflicts(self, conflicts: Sequence[Conflict]) -> List[Conflict]:
        seen = set()
        result = []
        for item in conflicts:
            marker = (item.conflict_id, item.values, item.detail)
            if marker not in seen:
                seen.add(marker)
                result.append(item)
        return result


_DEFAULT_ENGINE: Optional[ClassificationEngine] = None


def default_engine() -> ClassificationEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = ClassificationEngine()
    return _DEFAULT_ENGINE


def classify_record(
    tensors: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
    filename: str,
) -> Classification:
    return default_engine().classify(tensors, metadata, filename)


def classify(
    tensors: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
    filename: str,
) -> Dict[str, str]:
    """Legacy-compatible dict facade used by the original scripts and tests."""
    result = classify_record(tensors, metadata, filename)
    structural_arches = unique(
        value.value
        for value in result.evidence
        if value.source == EvidenceSource.STRUCTURE.value
        and value.category == EvidenceCategory.ARCHITECTURE.value
    )
    metadata_arches = unique(
        value.value
        for value in result.evidence
        if value.source == EvidenceSource.METADATA.value
        and value.category == EvidenceCategory.ARCHITECTURE.value
    )
    filename_arches = unique(
        value.value
        for value in result.evidence
        if value.source == EvidenceSource.FILENAME.value
        and value.category == EvidenceCategory.ARCHITECTURE.value
    )
    return {
        "arch": result.arch,
        "structural_family": result.structural_family,
        "model_version": result.model_version,
        "task_variant": result.task_variant,
        "noise_stage": result.noise_stage,
        "confidence": result.confidence,
        "status": result.status,
        "ecosystem": join_semicolon(result.ecosystems),
        "model_variant": join_semicolon(result.model_variants),
        "unresolved_hint": join_semicolon(result.unresolved_hints),
        "techniques": join_semicolon(result.techniques),
        "lineage": result.sort_subfolder,
        "sort_subfolder": result.sort_subfolder,
        "format": result.format,
        "structure_arch": join_semicolon(structural_arches),
        "metadata_arch": join_semicolon(metadata_arches),
        "name_arch": join_semicolon(filename_arches),
        "evidence": "; ".join(
            [value for value in [result.evidence_summary()] if value]
            + ["CONFLICT: " + conflict.detail for conflict in result.conflicts]
            + ["WARNING: " + warning for warning in result.warnings]
        ),
        "evidence_json": json.dumps(result.evidence_json(), ensure_ascii=False, separators=(",", ":")),
        "conflicts_json": json.dumps(result.conflicts_json(), ensure_ascii=False, separators=(",", ":")),
        "warnings_json": json.dumps(result.warnings, ensure_ascii=False, separators=(",", ":")),
    }


# Public typed facade retained for callers that need structured evidence.
classify_result = classify_record
