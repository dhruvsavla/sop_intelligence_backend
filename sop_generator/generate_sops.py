"""
Generate 50 synthetic GxP SOPs via Claude Sonnet API.
Usage:
    python generate_sops.py
    python generate_sops.py --skip-existing
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings
from sop_generator.sop_templates import SOP_CATALOG

METADATA_PATH = Path(__file__).parent / "sop_metadata.json"
DATA_DIR = Path(settings.SOP_DATA_PATH)


def _build_generation_prompt(sop: dict) -> str:
    keywords_str = ", ".join(sop["keywords"])
    supersedes = sop["supersedes_version"] if sop["supersedes_version"] else "N/A (initial approved version)"

    return f"""Generate a complete, realistic Standard Operating Procedure (SOP) document for a pharmaceutical/life sciences GxP environment.

SOP SPECIFICATIONS:
  SOP Number:       {sop["sop_number"]}
  Title:            {sop["title"]}
  Domain:           {sop["domain"]} ({_domain_fullname(sop["domain"])})
  Version:          {sop["version"]}
  Effective Date:   {sop["effective_date"]}
  Supersedes:       {supersedes}
  Approved By:      {sop["approver"]}
  Section Count:    {sop["section_count"]} sections
  Page Estimate:    {sop["page_estimate"]} pages
  Key Terms:        {keywords_str}

DOCUMENT STRUCTURE REQUIREMENTS:

Produce the document in this exact order:

═══════════════════════════════════════════════
  HEADER BLOCK (use this exact formatting)
═══════════════════════════════════════════════
SOP Number: {sop["sop_number"]}
Title: {sop["title"]}
Version: {sop["version"]}
Effective Date: {sop["effective_date"]}
Supersedes: {supersedes}
Department: {sop["domain"]}
Approved By: {sop["approver"]}
Review Cycle: 2 years
Pages: {sop["page_estimate"]}
Classification: GxP Critical

═══════════════════════════════════════════════
  TABLE OF CONTENTS
═══════════════════════════════════════════════
List all section numbers and titles with page numbers.

═══════════════════════════════════════════════
  BODY — {sop["section_count"]} SECTIONS
═══════════════════════════════════════════════

Generate exactly {sop["section_count"]} numbered sections. Each section MUST contain:
- Section header: "X.0 SECTION TITLE IN ALL CAPS"
- 3–6 numbered subsections (X.1, X.2, X.3, etc.)
- Minimum 3–4 substantive paragraphs per section (not placeholders)
- At least one data table with realistic values, limits, or specifications
- At least one numbered procedure list with actionable steps
- Specific regulatory citations where appropriate (FDA 21 CFR, ICH guidelines, EMA guidelines)
- Concrete numerical parameters: acceptance criteria, time limits, temperature ranges, concentrations

MANDATORY SECTIONS (include in this order):
1.0  PURPOSE AND SCOPE
2.0  DEFINITIONS AND ABBREVIATIONS
3.0  ROLES AND RESPONSIBILITIES
4.0  [CORE PROCEDURE — the primary operational section for this SOP topic]
5.0+ [ADDITIONAL PROCEDURE SECTIONS — expand the procedural depth for {sop["domain"]} operations]
(N-2).0  DEVIATION AND EXCEPTION HANDLING
(N-1).0  REFERENCES AND RELATED DOCUMENTS
(N).0    REVISION HISTORY

CONTENT REQUIREMENTS:
- Do NOT use placeholder text like "[Insert X here]" or "TBD"
- Write as a real regulatory document a QA professional would approve
- For {sop["domain"]} operations: include scientifically accurate parameters, acceptance criteria, and regulatory framework
- Reference ICH, FDA 21 CFR Parts, EMA/EU guidelines as appropriate to the domain
- Include realistic role names: QA Specialist, Manufacturing Operator, Laboratory Analyst, etc.
- For tables: include column headers and ≥3 data rows
- For procedure lists: include ≥5 specific numbered steps

The SOP should be comprehensive enough that a trained GxP professional could use it as a working document.
"""


def _domain_fullname(domain: str) -> str:
    mapping = {
        "GMP": "Good Manufacturing Practice",
        "GCP": "Good Clinical Practice",
        "GLP": "Good Laboratory Practice",
        "PV": "Pharmacovigilance",
        "DI": "Data Integrity",
    }
    return mapping.get(domain, domain)


async def generate_sop(client: anthropic.Anthropic, sop: dict, skip_existing: bool) -> dict:
    version_safe = sop["version"].replace(".", "_")
    filename = f"{sop['sop_number']}_v{sop['version']}.txt"
    filepath = DATA_DIR / filename

    if skip_existing and filepath.exists() and filepath.stat().st_size > 500:
        return {**sop, "file_path": str(filepath), "status": "skipped"}

    prompt = _build_generation_prompt(sop)

    message = client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    content = message.content[0].text
    filepath.write_text(content, encoding="utf-8")

    return {**sop, "file_path": str(filepath), "status": "generated"}


async def generate_all_sops(skip_existing: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    results = []
    total = len(SOP_CATALOG)

    for i, sop in enumerate(SOP_CATALOG, start=1):
        status_word = "Checking" if skip_existing else "Generating"
        print(f"[{i}/{total}] {status_word} {sop['sop_number']}: {sop['title']}...")

        try:
            result = await generate_sop(client, sop, skip_existing)
            status = result.get("status", "generated")
            if status == "skipped":
                print(f"  → Skipped (already exists)")
            else:
                print(f"  → Saved to {result['file_path']}")
            results.append(result)
        except Exception as exc:
            print(f"  ✗ ERROR: {exc}")
            results.append({**sop, "status": "error", "error": str(exc)})

        if i < total:
            await asyncio.sleep(0.5)

    METADATA_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    generated = sum(1 for r in results if r.get("status") == "generated")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    errors = sum(1 for r in results if r.get("status") == "error")
    print(f"\n✅ Done. Generated: {generated} | Skipped: {skipped} | Errors: {errors}")
    print(f"   Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 50 synthetic GxP SOPs via Claude")
    parser.add_argument("--skip-existing", action="store_true", help="Skip SOPs that already have output files")
    args = parser.parse_args()
    asyncio.run(generate_all_sops(skip_existing=args.skip_existing))
