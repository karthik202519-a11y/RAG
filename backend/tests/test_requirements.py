"""Every third-party import in the backend must be declared in requirements.txt (a missing one only shows up in a clean install)."""
import ast
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
DIST = {"jwt": "pyjwt", "dotenv": "python-dotenv", "bs4": "beautifulsoup4", "docx": "python-docx", "pptx": "python-pptx",
        "qdrant_client": "qdrant-client", "rank_bm25": "rank-bm25", "multipart": "python-multipart", "botocore": "boto3", "PIL": "pillow", "faiss": "faiss-cpu"}


def test_every_third_party_import_is_in_requirements():
    imported = set()
    for f in (BACKEND / "studio").rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                imported |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                imported.add(n.module.split(".")[0])
    third = {m for m in imported if m not in sys.stdlib_module_names and m != "studio"}
    req = (BACKEND / "requirements.txt").read_text(encoding="utf-8").lower()
    missing = sorted(m for m in third if not re.search(r"^" + re.escape(DIST.get(m, m).lower()).replace(r"\-", "[-_]") + r"\b", req, re.M))
    assert not missing, f"imported but not in backend/requirements.txt: {missing}"
