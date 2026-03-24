"""
Step F – Privacy Detection & Redaction

Uses Microsoft Presidio (AnalyzerEngine + AnonymizerEngine) for PII
detection and redaction.  Optionally loads a custom HuggingFace NER model
(default: Meddies/meddies-pii) as an additional recognizer.

Output → privacy_checked.jsonl
"""

from __future__ import annotations

import logging
from typing import Optional

from text.config.settings import Settings
from text.models.schemas import DedupDocument, PIIEntity, PrivacyResult

logger = logging.getLogger(__name__)

# Module-level singletons (lazy-loaded)
_analyzer = None
_anonymizer = None


def _get_analyzer(settings: Settings):
    """Lazy-init the Presidio AnalyzerEngine with optional custom NER model."""
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Configure spaCy NLP engine for supported languages
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_sm"},
        ],
    }

    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()
    _analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=settings.presidio_languages,
    )

    # Optionally add a HuggingFace Transformers-based recognizer
    if settings.pii_model_name:
        try:
            _register_transformers_recognizer(_analyzer, settings)
        except Exception as e:
            logger.warning(
                "Failed to load custom PII model '%s': %s. "
                "Continuing with built-in recognizers only.",
                settings.pii_model_name, e,
            )

    logger.info("Presidio AnalyzerEngine initialised")
    return _analyzer


def _register_transformers_recognizer(analyzer, settings: Settings):
    """
    Register a HuggingFace NER pipeline as a Presidio RecognizerRegistry entry.
    Uses the TransformersRecognizer pattern if available, otherwise wraps manually.
    """
    try:
        from presidio_analyzer.predefined_recognizers import TransformersRecognizer

        transformers_recognizer = TransformersRecognizer(
            model_path=settings.pii_model_name,
            supported_entities=[
                "PERSON", "LOCATION", "ORGANIZATION",
                "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                "ID", "DATE_TIME",
            ],
            supported_language="en",
        )
        analyzer.registry.add_recognizer(transformers_recognizer)
        logger.info("Registered TransformersRecognizer with model: %s", settings.pii_model_name)
    except (ImportError, Exception) as e:
        logger.info(
            "TransformersRecognizer not available (%s), "
            "registering manual NER wrapper", e
        )
        _register_manual_ner(analyzer, settings)


def _register_manual_ner(analyzer, settings: Settings):
    """Manual fallback: wrap a HuggingFace pipeline as a Presidio EntityRecognizer."""
    from presidio_analyzer import EntityRecognizer, RecognizerResult

    class HFNerRecognizer(EntityRecognizer):
        def __init__(self, model_name: str):
            super().__init__(
                supported_entities=["PERSON", "LOCATION", "ORGANIZATION"],
                supported_language="en",
                name="HFNerRecognizer",
            )
            from transformers import pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
            )

        def load(self):
            pass

        def analyze(self, text: str, entities, nlp_artifacts=None):
            results = []
            ner_results = self.ner_pipeline(text[:5000])
            for ent in ner_results:
                entity_type = ent["entity_group"].upper()
                if entity_type in {"PER", "PERSON"}:
                    entity_type = "PERSON"
                elif entity_type in {"LOC", "LOCATION", "GPE"}:
                    entity_type = "LOCATION"
                elif entity_type in {"ORG", "ORGANIZATION"}:
                    entity_type = "ORGANIZATION"
                else:
                    continue
                if entities and entity_type not in entities:
                    continue
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=ent["start"],
                        end=ent["end"],
                        score=float(ent["score"]),
                    )
                )
            return results

    try:
        recognizer = HFNerRecognizer(settings.pii_model_name)
        analyzer.registry.add_recognizer(recognizer)
        logger.info("Registered manual HF NER recognizer: %s", settings.pii_model_name)
    except Exception as e:
        logger.warning("Failed to register HF NER: %s", e)


def _get_anonymizer():
    """Lazy-init the Presidio AnonymizerEngine."""
    global _anonymizer
    if _anonymizer is not None:
        return _anonymizer
    from presidio_anonymizer import AnonymizerEngine
    _anonymizer = AnonymizerEngine()
    return _anonymizer


def _analyze_and_redact(
    text: str,
    analyzer,
    anonymizer,
    languages: list[str],
    score_threshold: float,
) -> tuple[str, list[PIIEntity]]:
    """Run analysis + anonymisation on a single text block."""
    from presidio_anonymizer.entities import OperatorConfig

    all_results = []
    for lang in languages:
        try:
            results = analyzer.analyze(
                text=text,
                language=lang,
                score_threshold=score_threshold,
            )
            all_results.extend(results)
        except Exception as e:
            logger.debug("Analyzer failed for lang=%s: %s", lang, e)

    # Deduplicate overlapping results (keep higher-score)
    all_results.sort(key=lambda r: (-r.score, r.start))
    deduped = []
    used_ranges: list[tuple[int, int]] = []
    for r in all_results:
        overlap = False
        for start, end in used_ranges:
            if r.start < end and r.end > start:
                overlap = True
                break
        if not overlap:
            deduped.append(r)
            used_ranges.append((r.start, r.end))

    # Anonymise
    if deduped:
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=deduped,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
                "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
                "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            },
        )
        redacted_text = anonymized.text
    else:
        redacted_text = text

    pii_entities = [
        PIIEntity(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=round(r.score, 4),
            original_text=text[r.start : r.end][:100],
        )
        for r in deduped
    ]

    return redacted_text, pii_entities


# ────────────────────────────────────────────────────────────
# Fallback (no Presidio)
# ────────────────────────────────────────────────────────────

def _fallback_scan(documents: list[DedupDocument]) -> list[PrivacyResult]:
    """Simple pass-through when Presidio is not installed."""
    logger.warning("Presidio not available – passing documents through without PII redaction")
    return [
        PrivacyResult(
            doc_id=doc.doc_id,
            original_text=doc.text,
            redacted_text=doc.text,
            pii_entities=[],
            pii_count=0,
        )
        for doc in documents
        if not doc.is_duplicate
    ]


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def run(
    documents: list[DedupDocument],
    settings: Settings | None = None,
) -> list[PrivacyResult]:
    """
    Execute privacy detection and redaction.

    Parameters
    ----------
    documents : list[DedupDocument]
    settings : Settings, optional

    Returns
    -------
    list[PrivacyResult]
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    try:
        analyzer = _get_analyzer(settings)
        anonymizer = _get_anonymizer()
    except ImportError:
        return _fallback_scan(documents)

    results: list[PrivacyResult] = []
    for doc in documents:
        if doc.is_duplicate:
            continue
        redacted_text, pii_entities = _analyze_and_redact(
            doc.text,
            analyzer,
            anonymizer,
            settings.presidio_languages,
            settings.pii_score_threshold,
        )
        results.append(
            PrivacyResult(
                doc_id=doc.doc_id,
                original_text=doc.text,
                redacted_text=redacted_text,
                pii_entities=pii_entities,
                pii_count=len(pii_entities),
            )
        )
        if pii_entities:
            logger.debug(
                "Doc %s: %d PII entities detected", doc.doc_id, len(pii_entities)
            )

    total_pii = sum(r.pii_count for r in results)
    logger.info(
        "Privacy detection complete: %d PII entities across %d documents",
        total_pii, len(results),
    )
    return results
