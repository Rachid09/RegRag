import zipfile

import pytest

from regrag.download import Source, extract_documents, fetch, sha256_of


def test_extract_only_document_jsons(tmp_path):
    z = tmp_path / "docs.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("StructuredRegulatoryDocuments/1.json", "[]")
        f.writestr("StructuredRegulatoryDocuments/40.json", "[]")
        f.writestr("__MACOSX/StructuredRegulatoryDocuments/._1.json", "junk")
        f.writestr("StructuredRegulatoryDocuments/../../evil.json", "x")
    out = tmp_path / "out"
    assert extract_documents(z, out) == 2
    assert sorted(p.name for p in out.iterdir()) == ["1.json", "40.json"]


def test_fetch_rejects_hash_mismatch(tmp_path):
    src_file = tmp_path / "src.txt"
    src_file.write_text("hello")
    dest = tmp_path / "dest"
    dest.mkdir()
    bad = Source("src.txt", src_file.as_uri(), "0" * 64)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        fetch(bad, dest)
    assert not any(dest.iterdir())

    good = Source("src.txt", src_file.as_uri(), sha256_of(src_file))
    assert fetch(good, dest).read_text() == "hello"
