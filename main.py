"""
SOPAssist FastAPI backend.
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import anthropic
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.qa_agent import SOPQAAgent
from config import settings
from ingestion.chroma_store import ChromaSOPStore
from ingestion.embedder import SOPEmbedder
from retrieval.confidence_scorer import ConfidenceScorer
from retrieval.retriever import VersionAwareRetriever

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    domain_filter: Optional[str] = None


class ChunkSchema(BaseModel):
    sop_number: str
    sop_title: str
    version: str
    effective_date: str
    section_number: str
    section_title: str
    citation: str
    similarity_score: float
    is_current_version: bool
    text: str


class QAResponseSchema(BaseModel):
    answer: str
    citations: list[str]
    confidence_score: float
    confidence_level: str
    escalated: bool
    escalation_reason: Optional[str]
    retrieved_chunks: list[ChunkSchema]
    query: str
    model_used: str
    processing_time_ms: int


# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting SOPAssist backend...")
    embedder = SOPEmbedder(settings.EMBEDDING_MODEL)
    store = ChromaSOPStore(
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )
    retriever = VersionAwareRetriever(chroma_store=store, embedder=embedder)
    confidence_scorer = ConfidenceScorer(threshold=settings.CONFIDENCE_THRESHOLD)
    anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    agent = SOPQAAgent(
        retriever=retriever,
        confidence_scorer=confidence_scorer,
        anthropic_client=anthropic_client,
        model=settings.CLAUDE_MODEL,
    )

    app.state.embedder = embedder
    app.state.store = store
    app.state.retriever = retriever
    app.state.confidence_scorer = confidence_scorer
    app.state.agent = agent

    print("✅ SOPAssist backend ready.")
    yield
    print("Shutting down SOPAssist backend.")


app = FastAPI(title="SOPAssist", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

async def _health_response():
    stats = app.state.store.get_collection_stats()
    return {
        "status": "healthy",
        "collection_stats": stats,
        "model": settings.CLAUDE_MODEL,
    }

@app.get("/health")
async def health():
    return await _health_response()

@app.get("/api/health")
async def api_health():
    return await _health_response()


@app.post("/api/query", response_model=QAResponseSchema)
async def query_sops(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    agent: SOPQAAgent = app.state.agent
    response = await agent.answer(
        query=request.query,
        domain_filter=request.domain_filter,
    )

    chunks = [
        ChunkSchema(
            sop_number=r.sop_number,
            sop_title=r.sop_title,
            version=r.version,
            effective_date=r.effective_date,
            section_number=r.section_number,
            section_title=r.section_title,
            citation=r.citation,
            similarity_score=r.similarity_score,
            is_current_version=r.is_current_version,
            text=r.text,
        )
        for r in response.retrieved_chunks
    ]

    return QAResponseSchema(
        answer=response.answer,
        citations=response.citations,
        confidence_score=response.confidence.score,
        confidence_level=response.confidence.level,
        escalated=response.escalated,
        escalation_reason=response.escalation_reason,
        retrieved_chunks=chunks,
        query=response.query,
        model_used=response.model_used,
        processing_time_ms=response.processing_time_ms,
    )


@app.get("/api/sops")
async def list_sops():
    metadata_path = Path(__file__).parent / "sop_generator" / "sop_metadata.json"
    if not metadata_path.exists():
        return {"sops": [], "total": 0}

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    sops = []
    for m in metadata:
        if m.get("status") in ("error", None) and "sop_number" not in m:
            continue
        sops.append({
            "sop_number": m.get("sop_number", ""),
            "title": m.get("title", ""),
            "version": m.get("version", ""),
            "domain": m.get("domain", ""),
            "effective_date": m.get("effective_date", ""),
            "section_count": m.get("section_count", 0),
            "keywords": m.get("keywords", []),
        })

    return {"sops": sops, "total": len(sops)}


@app.get("/api/sops/{sop_number}")
async def get_sop(sop_number: str):
    metadata_path = Path(__file__).parent / "sop_generator" / "sop_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    sop = next((m for m in metadata if m.get("sop_number", "").upper() == sop_number.upper()), None)
    if not sop:
        raise HTTPException(status_code=404, detail=f"SOP {sop_number} not found")

    # Get sections from ChromaDB
    store: ChromaSOPStore = app.state.store
    results = store.collection.get(
        where={"sop_number": sop_number.upper()},
        include=["metadatas"],
    )
    sections = []
    for m in results.get("metadatas", []):
        if m:
            sections.append({
                "section_number": m.get("section_number", ""),
                "section_title": m.get("section_title", ""),
            })

    return {**sop, "sections": sections}


@app.get("/api/sops/{sop_number}/content")
async def get_sop_content(sop_number: str):
    metadata_path = Path(__file__).parent / "sop_generator" / "sop_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    sop = next((m for m in metadata if m.get("sop_number", "").upper() == sop_number.upper()), None)
    if not sop:
        raise HTTPException(status_code=404, detail=f"SOP {sop_number} not found")

    file_path = sop.get("file_path")
    if not file_path:
        file_path = str(Path(settings.SOP_DATA_PATH) / f"{sop_number.upper()}_v{sop['version']}.txt")

    sop_file = Path(file_path)
    if not sop_file.exists():
        raise HTTPException(status_code=404, detail=f"SOP file not found on disk: {file_path}")

    content = sop_file.read_text(encoding="utf-8", errors="replace")
    return {
        "sop_number": sop.get("sop_number"),
        "title": sop.get("title"),
        "version": sop.get("version"),
        "domain": sop.get("domain"),
        "effective_date": sop.get("effective_date"),
        "approver": sop.get("approver"),
        "content": content,
    }


@app.get("/api/domains")
async def list_domains():
    stats = app.state.store.get_collection_stats()
    domain_labels = {
        "GMP": "Good Manufacturing Practice",
        "GCP": "Good Clinical Practice",
        "GLP": "Good Laboratory Practice",
        "PV": "Pharmacovigilance",
        "DI": "Data Integrity",
    }
    domains = stats.get("domains", [])
    counts: dict[str, int] = {}

    if domains:
        metadata_path = Path(__file__).parent / "sop_generator" / "sop_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            for m in metadata:
                d = m.get("domain", "")
                if d:
                    counts[d] = counts.get(d, 0) + 1

    return {
        "domains": domains,
        "labels": domain_labels,
        "counts": counts,
    }


_ingestion_status = {"running": False, "last_result": None}
_eval_status = {"running": False, "last_result": None}


def _run_ingestion_background():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "ingestion/run_ingestion.py"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent,
    )
    _ingestion_status["running"] = False
    _ingestion_status["last_result"] = result.stdout[-2000:] if result.stdout else result.stderr[-2000:]


@app.post("/api/ingest")
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    x_admin_key: Optional[str] = Header(None),
):
    admin_key = os.environ.get("ADMIN_KEY", "")
    if admin_key and x_admin_key != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    if _ingestion_status["running"]:
        raise HTTPException(status_code=409, detail="Ingestion already running")

    _ingestion_status["running"] = True
    background_tasks.add_task(_run_ingestion_background)
    return {"status": "started", "message": "Ingestion running in background"}


def _run_evaluation_background():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "evaluation/evaluator.py"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent,
    )
    _eval_status["running"] = False
    _eval_status["last_result"] = result.stdout[-2000:] if result.stdout else result.stderr[-2000:]


@app.post("/api/evaluate")
async def trigger_evaluation(background_tasks: BackgroundTasks):
    if _eval_status["running"]:
        raise HTTPException(status_code=409, detail="Evaluation already running")
    _eval_status["running"] = True
    background_tasks.add_task(_run_evaluation_background)
    return {"status": "started"}


@app.get("/api/evaluate/status")
async def eval_status():
    return {
        "running": _eval_status["running"],
        "last_result": _eval_status["last_result"],
    }


@app.get("/api/evaluate/report")
async def eval_report():
    report_path = Path(__file__).parent / "reports" / "evaluation_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation report not yet generated. Run /api/evaluate first.")
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)
