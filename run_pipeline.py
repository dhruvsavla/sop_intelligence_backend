"""
Master pipeline runner.
Usage:
    python run_pipeline.py --all
    python run_pipeline.py --generate
    python run_pipeline.py --ingest
    python run_pipeline.py --evaluate
"""
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent


def run_step(name: str, script: str, extra_args: list[str] = None):
    print(f"\n{'═'*50}")
    print(f"  STEP: {name}")
    print(f"{'═'*50}")
    cmd = [sys.executable, str(BASE / script)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"\n❌ Step '{name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n✅ Step '{name}' completed")


def main():
    parser = argparse.ArgumentParser(description="SOPAssist pipeline runner")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: generate → ingest")
    parser.add_argument("--generate", action="store_true", help="Generate SOPs")
    parser.add_argument("--ingest", action="store_true", help="Ingest into ChromaDB")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing SOP files during generation")
    args = parser.parse_args()

    if not any([args.all, args.generate, args.ingest, args.evaluate]):
        parser.print_help()
        sys.exit(0)

    print("\n╔══════════════════════════════════╗")
    print("║  SOPAssist Pipeline Runner       ║")
    print("╚══════════════════════════════════╝")

    if args.all or args.generate:
        extra = ["--skip-existing"] if args.skip_existing else []
        run_step("Generate 50 SOPs", "sop_generator/generate_sops.py", extra)

    if args.all or args.ingest:
        run_step("Ingest into ChromaDB", "ingestion/run_ingestion.py")

    if args.evaluate:
        run_step("Run Evaluation (60 questions)", "evaluation/evaluator.py")

    print("\n✅ Pipeline complete.")

    if args.all or args.ingest:
        print("\nNext step: Start the API server:")
        print("  uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
