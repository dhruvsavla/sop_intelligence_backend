"""
SOPAssist setup checker.
Run: python setup.py
"""
import os
import shutil
from pathlib import Path

BASE = Path(__file__).parent

DIRS = [
    BASE / "data" / "sops",
    BASE / "data" / "chroma_db",
    BASE / "reports",
]


def main():
    print("═" * 50)
    print("  SOPAssist Setup Check")
    print("═" * 50)

    # 1. Directories
    all_dirs_ok = True
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directories created/verified")

    # 2. .env file
    env_path = BASE / ".env"
    env_example = BASE / ".env.example"
    if not env_path.exists():
        if env_example.exists():
            shutil.copy(env_example, env_path)
            print(f"✅ .env file created from .env.example (please set your API key)")
        else:
            print(f"❌ .env.example not found — cannot create .env")
    else:
        print(f"✅ .env file present")

    # 3. API key
    from dotenv import load_dotenv
    load_dotenv(env_path)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key and api_key != "your_anthropic_api_key_here":
        print(f"✅ ANTHROPIC_API_KEY detected")
    else:
        print(f"❌ ANTHROPIC_API_KEY not set — open .env and add your key")

    # 4. SOP files
    sop_dir = BASE / "data" / "sops"
    sop_files = list(sop_dir.glob("*.txt"))
    if sop_files:
        print(f"✅ {len(sop_files)} SOP files found in data/sops/")
    else:
        print(f"⬜ SOPs not yet generated — run: python sop_generator/generate_sops.py")

    # 5. ChromaDB
    chroma_dir = BASE / "data" / "chroma_db"
    chroma_files = list(chroma_dir.rglob("*.sqlite3")) + list(chroma_dir.rglob("*.bin"))
    if chroma_files:
        print(f"✅ ChromaDB data found in data/chroma_db/")
    else:
        print(f"⬜ ChromaDB not yet populated — run: python ingestion/run_ingestion.py")

    print("═" * 50)


if __name__ == "__main__":
    main()
