"""
SOPAssist Evaluation Framework.
Runs 60 Q&A pairs, computes metrics, saves report.
"""
import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.qa_agent import QAResponse, SOPQAAgent
from config import settings
from evaluation.eval_dataset import EVALUATION_DATASET
from ingestion.chroma_store import ChromaSOPStore
from ingestion.embedder import SOPEmbedder
from retrieval.confidence_scorer import ConfidenceScorer
from retrieval.retriever import VersionAwareRetriever


@dataclass
class EvaluationReport:
    timestamp: str
    total_questions: int
    answerable_questions: int
    oos_questions: int
    citation_accuracy: float
    escalation_accuracy: float
    false_escalation_rate: float
    average_confidence_score: float
    average_processing_time_ms: float
    domain_breakdown: dict
    per_question_results: list
    summary: str


class SOPEvaluator:
    def __init__(self, qa_agent: SOPQAAgent):
        self.agent = qa_agent

    def _check_citation_present(self, response: QAResponse, expected_pattern: Optional[str]) -> bool:
        if not expected_pattern:
            return False
        for citation in response.citations:
            if re.search(expected_pattern, citation):
                return True
        # Also search answer text directly
        if re.search(expected_pattern, response.answer):
            return True
        return False

    def _check_correct_escalation(self, response: QAResponse, is_answerable: bool) -> bool:
        if not is_answerable:
            # Out-of-scope should escalate
            return response.escalated
        else:
            # Answerable should NOT escalate (or at least produce a citation)
            return not response.escalated or len(response.citations) > 0

    async def run_evaluation(self) -> EvaluationReport:
        print("\n" + "═" * 50)
        print("  Running SOPAssist Evaluation")
        print("═" * 50)

        per_question_results = []
        total = len(EVALUATION_DATASET)

        answerable = [q for q in EVALUATION_DATASET if q["is_answerable"]]
        oos = [q for q in EVALUATION_DATASET if not q["is_answerable"]]

        citation_correct = 0
        escalation_correct_oos = 0
        false_escalations = 0
        all_confidence_scores = []
        all_processing_times = []
        domain_results: dict[str, list[bool]] = {}

        for i, question in enumerate(EVALUATION_DATASET, start=1):
            q_id = question["question_id"]
            q_text = question["question"]
            is_answerable = question["is_answerable"]
            domain = question.get("domain")

            print(f"[{i}/{total}] {q_id}: {q_text[:70]}...")

            try:
                response = await self.agent.answer_async(q_text, domain_filter=domain)
            except Exception as exc:
                print(f"  ✗ Error: {exc}")
                per_question_results.append({
                    "question_id": q_id,
                    "question": q_text,
                    "is_answerable": is_answerable,
                    "domain": domain,
                    "error": str(exc),
                    "citation_correct": False,
                    "escalation_correct": False,
                    "confidence_score": 0.0,
                    "escalated": False,
                    "processing_time_ms": 0,
                })
                continue

            citation_ok = self._check_citation_present(response, question.get("expected_citation_pattern"))
            escalation_ok = self._check_correct_escalation(response, is_answerable)

            if is_answerable and citation_ok:
                citation_correct += 1
            if not is_answerable and response.escalated:
                escalation_correct_oos += 1
            if is_answerable and response.escalated and not response.citations:
                false_escalations += 1

            all_confidence_scores.append(response.confidence.score)
            all_processing_times.append(response.processing_time_ms)

            if domain:
                domain_results.setdefault(domain, []).append(citation_ok)

            per_question_results.append({
                "question_id": q_id,
                "question": q_text,
                "is_answerable": is_answerable,
                "domain": domain,
                "citation_correct": citation_ok,
                "escalation_correct": escalation_ok,
                "confidence_score": response.confidence.score,
                "confidence_level": response.confidence.level,
                "escalated": response.escalated,
                "escalation_reason": response.escalation_reason,
                "citations_found": response.citations,
                "processing_time_ms": response.processing_time_ms,
            })

            print(f"  Citation: {'✅' if citation_ok else '❌'} | "
                  f"Escalated: {response.escalated} | "
                  f"Confidence: {response.confidence.score:.2f} ({response.confidence.level})")

            await asyncio.sleep(0.3)

        # Compute final metrics
        n_answerable = len(answerable)
        n_oos = len(oos)

        citation_accuracy = citation_correct / n_answerable if n_answerable else 0.0
        escalation_accuracy = escalation_correct_oos / n_oos if n_oos else 0.0
        false_escalation_rate = false_escalations / n_answerable if n_answerable else 0.0
        avg_confidence = sum(all_confidence_scores) / len(all_confidence_scores) if all_confidence_scores else 0.0
        avg_time = sum(all_processing_times) / len(all_processing_times) if all_processing_times else 0.0

        domain_breakdown = {}
        for domain_key, results in domain_results.items():
            domain_breakdown[domain_key] = round(sum(results) / len(results) * 100, 1) if results else 0.0

        passes = citation_accuracy >= 0.90 and escalation_accuracy >= 0.80
        summary = (
            "✅ PASS — Meets production readiness criteria"
            if passes
            else "⚠️  NEEDS IMPROVEMENT — Review citation accuracy or escalation handling"
        )

        report = EvaluationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_questions=total,
            answerable_questions=n_answerable,
            oos_questions=n_oos,
            citation_accuracy=round(citation_accuracy * 100, 1),
            escalation_accuracy=round(escalation_accuracy * 100, 1),
            false_escalation_rate=round(false_escalation_rate * 100, 1),
            average_confidence_score=round(avg_confidence, 4),
            average_processing_time_ms=round(avg_time, 1),
            domain_breakdown=domain_breakdown,
            per_question_results=per_question_results,
            summary=summary,
        )

        self._print_report(report)
        self._save_report(report)
        return report

    def _print_report(self, report: EvaluationReport):
        print("\n" + "═" * 50)
        print("  SOPAssist Evaluation Report")
        print("═" * 50)
        print(f"  Citation Accuracy:        {report.citation_accuracy:.1f}% (target: ≥90%)")
        print(f"  Escalation Accuracy:      {report.escalation_accuracy:.1f}%")
        print(f"  False Escalation Rate:    {report.false_escalation_rate:.1f}%")
        print(f"  Avg Confidence Score:     {report.average_confidence_score:.2f}")
        print(f"  Avg Processing Time:      {report.average_processing_time_ms:,.0f} ms")
        print("═" * 50)
        print("  Domain Breakdown:")
        for domain, acc in sorted(report.domain_breakdown.items()):
            print(f"    {domain}:  {acc:.1f}% citation accuracy")
        print("═" * 50)
        print(f"  STATUS: {report.summary}")
        print("═" * 50 + "\n")

    def _save_report(self, report: EvaluationReport):
        report_path = Path(__file__).parent.parent / "reports" / "evaluation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"✅ Report saved to {report_path}")


async def main():
    embedder = SOPEmbedder(settings.EMBEDDING_MODEL)
    store = ChromaSOPStore(
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )

    if not store.collection_exists_and_populated():
        print("❌ ChromaDB collection is empty. Run ingestion/run_ingestion.py first.")
        sys.exit(1)

    retriever = VersionAwareRetriever(chroma_store=store, embedder=embedder)
    confidence_scorer = ConfidenceScorer(threshold=settings.CONFIDENCE_THRESHOLD)
    anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    agent = SOPQAAgent(
        retriever=retriever,
        confidence_scorer=confidence_scorer,
        anthropic_client=anthropic_client,
        model=settings.CLAUDE_MODEL,
    )

    evaluator = SOPEvaluator(qa_agent=agent)
    await evaluator.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
