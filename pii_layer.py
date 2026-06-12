#!/usr/bin/env python3
"""
Production-Grade PII Masking Layer
------------------------------------
Uses Microsoft Presidio for detecting and masking sensitive information
before it reaches the LLM. Privacy-first: no restoration, no DB storage.

Supported entities (out of the box + custom):
  - PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, DATE_TIME
  - CREDIT_CARD, IBAN_CODE, IP_ADDRESS, URL, NRP
  - US_SSN, US_PASSPORT, US_BANK_NUMBER, US_DRIVER_LICENSE
  - MEDICAL_LICENSE, IN_PAN, IN_AADHAAR (custom)
  - IND_PHONE (custom)

Usage:
    from pii_layer import mask_pii, PIIMasker, PIIMaskResult
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerResult, Pattern, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE       = "en"
MAX_TEXT_LENGTH        = 32_000       # characters; truncate silently beyond this
ANALYZER_SCORE_THRESH  = 0.4          # minimum confidence to treat as PII
LOG_DETECTED_TYPES     = True         # log entity types (NOT values) for observability

# ---------------------------------------------------------------------------
# Entities Presidio will look for
# ---------------------------------------------------------------------------
STANDARD_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "DATE_TIME",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "NRP",             # Nationality, Religious, Political groups
    "US_SSN",
    "US_PASSPORT",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "MEDICAL_LICENSE",
    "IN_PAN",          # India PAN card (built-in in newer Presidio)
    # Custom entities added via _add_custom_recognizers:
    "IN_AADHAAR",
    "IND_PHONE",
]

# Replacement tokens shown in the masked text
ENTITY_REPLACEMENT_MAP: dict[str, str] = {
    "PERSON":              "[PERSON]",
    "EMAIL_ADDRESS":       "[EMAIL]",
    "PHONE_NUMBER":        "[PHONE]",
    "LOCATION":            "[LOCATION]",
    "DATE_TIME":           "[DATE]",
    "CREDIT_CARD":         "[CREDIT_CARD]",
    "IBAN_CODE":           "[IBAN]",
    "IP_ADDRESS":          "[IP_ADDRESS]",
    "URL":                 "[URL]",
    "NRP":                 "[NRP]",
    "US_SSN":              "[SSN]",
    "US_PASSPORT":         "[PASSPORT]",
    "US_BANK_NUMBER":      "[BANK_ACCOUNT]",
    "US_DRIVER_LICENSE":   "[DRIVER_LICENSE]",
    "MEDICAL_LICENSE":     "[MEDICAL_LICENSE]",
    "IN_PAN":              "[IN_PAN]",
    "IN_AADHAAR":          "[IN_AADHAAR]",
    "IND_PHONE":           "[PHONE]",
}

# ---------------------------------------------------------------------------
# Result dataclass  (carry stats without storing actual PII values)
# ---------------------------------------------------------------------------
@dataclass
class PIIMaskResult:
    original_length:    int
    masked_text:        str
    entities_detected:  list[str]        = field(default_factory=list)  # types only, never values
    entity_count:       int              = 0
    pii_detected:       bool             = False
    processing_time_ms: float            = 0.0
    error:              Optional[str]    = None

    # ------------------------------------------------------------------
    # DB storage hook — COMMENTED OUT intentionally.
    # Uncomment + implement once a storage backend is wired up.
    # ------------------------------------------------------------------
    # def to_audit_record(self) -> dict:
    #     """Return a safe audit record (no PII values, only metadata)."""
    #     return {
    #         "original_length":    self.original_length,
    #         "entity_count":       self.entity_count,
    #         "entities_detected":  self.entities_detected,
    #         "pii_detected":       self.pii_detected,
    #         "processing_time_ms": self.processing_time_ms,
    #         "timestamp":          time.time(),
    #     }
    #
    # async def save_to_db(self, db_session) -> None:
    #     """Persist audit record to database (no PII stored)."""
    #     record = self.to_audit_record()
    #     # await db_session.execute(INSERT_AUDIT_SQL, record)
    #     pass

# ---------------------------------------------------------------------------
# PIIMasker
# ---------------------------------------------------------------------------
class PIIMasker:
    """
    Production-grade PII masker.

    Thread-safe after __init__; analyzer and anonymizer are stateless per call.
    """

    def __init__(self, score_threshold: float = ANALYZER_SCORE_THRESH):
        self.score_threshold = score_threshold
        self._ready          = False
        self.analyzer        = None
        self.anonymizer      = None
        self._init()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init(self) -> None:
        try:
            # Spacy NLP engine (en_core_web_lg gives best accuracy;
            # fall back to en_core_web_sm if lg is not installed)
            try:
                provider = NlpEngineProvider(nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
                })
                nlp_engine = provider.create_engine()
                logger.info("PII: using spacy en_core_web_lg")
            except Exception:
                provider = NlpEngineProvider(nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                })
                nlp_engine = provider.create_engine()
                logger.warning("PII: en_core_web_lg not found, falling back to en_core_web_sm")

            self.analyzer  = AnalyzerEngine(nlp_engine=nlp_engine)
            self.anonymizer = AnonymizerEngine()
            self._add_custom_recognizers()
            self._build_operator_config()
            self._ready = True
            logger.info("PII layer initialised (Presidio, score_threshold=%.2f)", self.score_threshold)
        except Exception as exc:
            logger.error("PII layer failed to initialise: %s", exc, exc_info=True)
            self._ready = False

    def _add_custom_recognizers(self) -> None:
        """Register India-specific and any project-specific PII recognizers."""
        recognizers = [
            # ----------------------------------------------------------
            # Aadhaar  (12-digit, optional spaces every 4 digits)
            # ----------------------------------------------------------
            PatternRecognizer(
                supported_entity="IN_AADHAAR",
                name="IN_AADHAAR",
                patterns=[
                    Pattern(name="aadhaar_spaced", regex=r"\b\d{4}\s\d{4}\s\d{4}\b", score=0.9),
                    Pattern(name="aadhaar_compact", regex=r"\b\d{12}\b", score=0.6),
                ],
                context=["aadhaar", "aadhar", "uid", "uidai"],
            ),
            # ----------------------------------------------------------
            # Indian mobile numbers  (10 digits starting 6-9)
            # ----------------------------------------------------------
            PatternRecognizer(
                supported_entity="IND_PHONE",
                name="IND_PHONE",
                patterns=[
                    Pattern(name="ind_mobile",        regex=r"(?<!\d)[6-9]\d{9}(?!\d)",    score=0.75),
                    Pattern(name="ind_mobile_plus91",  regex=r"(?:\+91[\s\-]?)[6-9]\d{9}", score=0.95),
                ],
            ),
            # ----------------------------------------------------------
            # Patient / Medical Record Number  (common hospital formats)
            # ----------------------------------------------------------
            PatternRecognizer(
                supported_entity="MEDICAL_RECORD_NUMBER",
                name="MEDICAL_RECORD_NUMBER",
                patterns=[
                    Pattern(name="mrn", regex=r"\b(?:MRN|mrn|Patient\s*ID|pid)[\s:\-#]*[A-Z0-9]{6,12}\b", score=0.85),
                ],
                context=["mrn", "medical record", "patient id", "patient number"],
            ),
        ]

        for rec in recognizers:
            try:
                self.analyzer.registry.add_recognizer(rec)
                logger.debug("PII: added recognizer '%s'", rec.name)
            except Exception as exc:
                logger.warning("PII: could not add recognizer '%s': %s", rec.name, exc)

    def _build_operator_config(self) -> None:
        """Pre-build the operator config dict used by the anonymizer."""
        self._operators: dict[str, OperatorConfig] = {}
        for entity, replacement in ENTITY_REPLACEMENT_MAP.items():
            self._operators[entity] = OperatorConfig(
                "replace", {"new_value": replacement}
            )
        # Default for anything not explicitly listed
        self._operators["DEFAULT"] = OperatorConfig(
            "replace", {"new_value": "[REDACTED]"}
        )

    # ------------------------------------------------------------------
    # Core mask method
    # ------------------------------------------------------------------
    def mask(self, text: str) -> PIIMaskResult:
        """
        Detect and mask PII in *text*.

        Returns a PIIMaskResult with masked_text and metadata.
        Never raises; errors are captured in result.error.
        """
        start = time.perf_counter()

        # --- Guard: uninitialised engine ---
        if not self._ready:
            logger.warning("PII layer not ready — passing text through unmasked")
            return PIIMaskResult(
                original_length=len(text),
                masked_text=text,
                error="PII engine not initialised",
            )

        # --- Guard: empty or non-string ---
        if not text or not isinstance(text, str):
            return PIIMaskResult(original_length=0, masked_text=text or "")

        # --- Guard: length cap ---
        truncated = False
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning("PII: text length %d exceeds max %d — truncating", len(text), MAX_TEXT_LENGTH)
            text = text[:MAX_TEXT_LENGTH]
            truncated = True

        try:
            # 1. Analyse
            results: list[RecognizerResult] = self.analyzer.analyze(
                text=text,
                language=DEFAULT_LANGUAGE,
                entities=STANDARD_ENTITIES,
                score_threshold=self.score_threshold,
            )

            if not results:
                elapsed = (time.perf_counter() - start) * 1000
                return PIIMaskResult(
                    original_length=len(text),
                    masked_text=text,
                    pii_detected=False,
                    processing_time_ms=elapsed,
                )

            # 2. Collect entity types for logging/metrics (NO values stored)
            detected_types = sorted({r.entity_type for r in results})
            if LOG_DETECTED_TYPES:
                logger.info(
                    "PII detected: types=%s count=%d",
                    detected_types, len(results),
                )

            # 3. Anonymize
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=self._operators,
            )

            elapsed = (time.perf_counter() - start) * 1000

            # ----------------------------------------------------------
            # Audit token storage — COMMENTED OUT.
            # When you're ready to persist audit metadata (not PII values):
            #
            # result = PIIMaskResult(...)
            # await result.save_to_db(db_session)   # wire in your session
            # ----------------------------------------------------------

            return PIIMaskResult(
                original_length=len(text),
                masked_text=anonymized.text,
                entities_detected=detected_types,
                entity_count=len(results),
                pii_detected=True,
                processing_time_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("PII masking error: %s", exc, exc_info=True)
            # Fail open: return original text so the request is not silently dropped.
            # Depending on your policy you may want to fail closed (raise / block).
            return PIIMaskResult(
                original_length=len(text),
                masked_text=text,
                processing_time_ms=elapsed,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Health check  (used by /health endpoint)
    # ------------------------------------------------------------------
    def health(self) -> dict:
        return {
            "ready":            self._ready,
            "score_threshold":  self.score_threshold,
            "engine":           "presidio",
        }


# ---------------------------------------------------------------------------
# Module-level singleton  (import-safe; initialised once at import time)
# ---------------------------------------------------------------------------
_pii_masker: Optional[PIIMasker] = None


def get_pii_masker() -> PIIMasker:
    """Return the module-level PIIMasker singleton, creating it if needed."""
    global _pii_masker
    if _pii_masker is None:
        _pii_masker = PIIMasker()
    return _pii_masker


def mask_pii(text: str) -> str:
    """
    Convenience wrapper — masks PII and returns the masked string.

    Suitable for synchronous call sites (blocking). For async contexts
    use: await asyncio.to_thread(mask_pii, text)
    """
    result = get_pii_masker().mask(text)
    if result.error:
        logger.warning("mask_pii: error during masking: %s", result.error)
    return result.masked_text


def mask_pii_detailed(text: str) -> PIIMaskResult:
    """
    Like mask_pii() but returns the full PIIMaskResult for callers that
    need metadata (entity types, timing, etc.).
    """
    return get_pii_masker().mask(text)