"""Download the ObliQA data from pinned commits and verify every file by SHA-256.

QA splits come from the Hugging Face dataset RegNLP/ObliQA (Apache-2.0). The 40 structured
regulation documents and the DocumentID -> title map are not on Hugging Face; they come from
the authors' GitHub repo RegNLP/ObliQADataset. Usage: `uv run python -m regrag.download`.
"""

import hashlib
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from regrag.config import DOCS_DIR, RAW_DIR

HF_REV = "53d87b5b72f9450c56e90cbde216885c8021dbf7"
GH_REV = "c1b2c3c04aee1f6ce75f76106ce13af672d48e02"
HF = f"https://huggingface.co/datasets/RegNLP/ObliQA/resolve/{HF_REV}/ObliQA"
GH = f"https://raw.githubusercontent.com/RegNLP/ObliQADataset/{GH_REV}"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str


SOURCES = [
    Source(
        "ObliQA_train.json",
        f"{HF}/ObliQA_train.json",
        "0bf0fdb4ead84faa8b6865022cfd4fb3caaa4617a82c718e9978bb951f5fb64f",
    ),
    Source(
        "ObliQA_dev.json",
        f"{HF}/ObliQA_dev.json",
        "cf6584f892c2ddde2da380fadfadea2008be7667f50085c27135b2dd29c14e20",
    ),
    Source(
        "ObliQA_test.json",
        f"{HF}/ObliQA_test.json",
        "cdca623b02f248ca736c2c2d98e9b6bebe4a82708f7984ef8f80f9fcd797e1b1",
    ),
    Source(
        "StructuredRegulatoryDocuments.zip",
        f"{GH}/StructuredRegulatoryDocuments.zip",
        "7f31af52e9bfacabfbe1f73c5bbd860afd93dd2aab72f05cb0608dd39ea1205d",
    ),
    Source(
        "DocumentMap.rtf",
        f"{GH}/scripts/DocumentMap.rtf",
        "e0ded4b23b65af6e7252c6c8cfbd627f90ae0bbd561a75b0442f39cb3532427c",
    ),
]

_DOC_MEMBER = re.compile(r"^StructuredRegulatoryDocuments/(\d+)\.json$")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(src: Source, dest_dir: Path) -> Path:
    dest = dest_dir / src.name
    if dest.exists() and sha256_of(dest) == src.sha256:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(src.url, timeout=120) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    got = sha256_of(tmp)
    if got != src.sha256:
        tmp.unlink()
        raise RuntimeError(f"{src.name}: sha256 mismatch (expected {src.sha256}, got {got})")
    tmp.replace(dest)
    return dest


def extract_documents(zip_path: Path, out_dir: Path) -> int:
    """Extract only StructuredRegulatoryDocuments/<n>.json; skip __MACOSX and anything else."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            m = _DOC_MEMBER.match(member)
            if m:
                (out_dir / f"{m.group(1)}.json").write_bytes(z.read(member))
                n += 1
    return n


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        print(f"ok  {fetch(src, RAW_DIR).relative_to(RAW_DIR.parent)}")
    n = extract_documents(RAW_DIR / "StructuredRegulatoryDocuments.zip", DOCS_DIR)
    print(f"extracted {n} documents to {DOCS_DIR}")


if __name__ == "__main__":
    main()
