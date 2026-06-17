"""
RAG Q&A Agent: retrieve → score → generate with citations.
"""
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from agent.prompts import (
    CONTEXT_BLOCK_TEMPLATE,
    ESCALATION_SYSTEM_PROMPT,
    QUERY_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from retrieval.confidence_scorer import ConfidenceResult, ConfidenceScorer
from retrieval.retriever import RetrievalResult, VersionAwareRetriever

CITATION_PATTERN = re.compile(r"\[SOP-[A-Z]+-\d+\s+v[\d.]+,\s+Section\s+[\d.]+\]")


@dataclass
class QAResponse:
    answer: str
    citations: list[str]
    confidence: ConfidenceResult
    retrieved_chunks: list[RetrievalResult]
    query: str
    escalated: bool
    escalation_reason: Optional[str]
    model_used: str
    processing_time_ms: int


class SOPQAAgent:
    def __init__(
        self,
        retriever: VersionAwareRetriever,
        confidence_scorer: ConfidenceScorer,
        anthropic_client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
    ):
        self.retriever = retriever
        self.confidence_scorer = confidence_scorer
        self.client = anthropic_client
        self.model = model

    async def answer(self, query: str, domain_filter: Optional[str] = None) -> QAResponse:
        start = time.perf_counter()

        # 1. Retrieve relevant chunks
        results = self.retriever.retrieve(query, n_results=8, domain_filter=domain_filter)

        # 2. Score confidence
        confidence = self.confidence_scorer.score(results, query)

        # 3. Build context string
        context_blocks = self._build_context(results)

        # 4. Choose system prompt
        system_prompt = ESCALATION_SYSTEM_PROMPT if confidence.should_escalate else SYSTEM_PROMPT

        # 5. Format user prompt
        user_prompt = QUERY_PROMPT_TEMPLATE.format(
            query=query,
            context_blocks=context_blocks if context_blocks else "(No relevant SOP sections found.)",
        )

        # 6. Call Claude
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        answer_text = message.content[0].text

        # 7. Parse citations — only keep ones grounded in retrieved chunks
        retrieved_citations = {r.citation for r in results}
        raw_citations = list(dict.fromkeys(CITATION_PATTERN.findall(answer_text)))
        citations = [c for c in raw_citations if c in retrieved_citations]

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return QAResponse(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            retrieved_chunks=results,
            query=query,
            escalated=confidence.should_escalate,
            escalation_reason=confidence.escalation_reason,
            model_used=self.model,
            processing_time_ms=elapsed_ms,
        )

    def _build_context(self, results: list[RetrievalResult]) -> str:
        blocks = []
        for r in results:
            block = CONTEXT_BLOCK_TEMPLATE.format(
                citation=r.citation,
                sop_title=r.sop_title,
                effective_date=r.effective_date,
                approver=r.approver if hasattr(r, "approver") else "",
                text=r.text,
            )
            blocks.append(block)
        return "\n\n".join(blocks)
