"""Deterministic, verdict-locked product evidence extraction."""

from .extractor import (
    CONCEPT_VOCABULARY,
    VOCABULARY_SHA256,
    VOCABULARY_VERSION,
    EvidenceChecks,
    EvidenceResult,
    SurfaceText,
    extract_evidence,
    render_submission,
    surface_text,
)

__all__ = [
    "CONCEPT_VOCABULARY",
    "VOCABULARY_SHA256",
    "VOCABULARY_VERSION",
    "EvidenceChecks",
    "EvidenceResult",
    "SurfaceText",
    "extract_evidence",
    "render_submission",
    "surface_text",
]
