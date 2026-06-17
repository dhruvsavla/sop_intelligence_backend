"""
Confidence scoring and escalation logic for retrieval results.
"""
from dataclasses import dataclass
from typing import Optional

from retrieval.retriever import RetrievalResult


@dataclass
class ConfidenceResult:
    score: float
    level: str              # "HIGH", "MEDIUM", "LOW"
    should_escalate: bool
    escalation_reason: Optional[str]
    top_similarity: float
    result_count: int


class ConfidenceScorer:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def score(self, results: list[RetrievalResult], query: str) -> ConfidenceResult:
        result_count = len(results)

        if result_count == 0:
            return ConfidenceResult(
                score=0.0,
                level="LOW",
                should_escalate=True,
                escalation_reason="No relevant SOP sections found for this query",
                top_similarity=0.0,
                result_count=0,
            )

        top_similarity = max(r.similarity_score for r in results)
        base_score = top_similarity

        # Add a penalty if the top match is very weak (typical of out-of-scope queries)
        if top_similarity < 0.65:
            base_score -= 0.10  # Force it down into the LOW tier

        # Penalty: too few results signals poor coverage
        if result_count < 2:
            base_score -= 0.05

        # Penalty: all results from same SOP section (low coverage diversity)
        unique_sections = {f"{r.sop_number}_{r.section_number}" for r in results}
        if len(unique_sections) == 1:
            base_score -= 0.03

        score = round(max(0.0, min(1.0, base_score)), 4)

        if score >= 0.75:
            level = "HIGH"
        elif score >= 0.60:
            level = "MEDIUM"
        else:
            level = "LOW"

        should_escalate = score < self.threshold

        escalation_reason: Optional[str] = None
        if should_escalate:
            # INCREASED THRESHOLD: 0.65 is a better cutoff for "out-of-scope" with MiniLM
            if top_similarity < 0.65: 
                escalation_reason = "Query may reference a topic not covered in the current SOP library"
            elif result_count < 2:
                escalation_reason = "Insufficient matching SOP sections found; Quality review recommended"
            else:
                escalation_reason = "Multiple possible interpretations; Quality review recommended"

        return ConfidenceResult(
            score=score,
            level=level,
            should_escalate=should_escalate,
            escalation_reason=escalation_reason,
            top_similarity=top_similarity,
            result_count=result_count,
        )
