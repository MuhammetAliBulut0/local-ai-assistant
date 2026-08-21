"""Arayüz olmadan, terminalden belge indeksleme scripti.

Kullanım:
    python scripts/ingest_cli.py belgeler/klasoru
    python scripts/ingest_cli.py dosya1.pdf dosya2.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import SUPPORTED_SUFFIXES, ingest_files  # noqa: E402


def collect_paths(args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            paths.extend(
                f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif p.is_file():
            paths.append(p)
        else:
            print(f"Uyarı: bulunamadı, atlanıyor -> {p}")
    return paths


def main() -> None:
    if len(sys.argv) < 2:
        print("Kullanım: python scripts/ingest_cli.py <dosya_veya_klasor> [...]")
        sys.exit(1)

    paths = collect_paths(sys.argv[1:])
    if not paths:
        print("İşlenecek desteklenen dosya bulunamadı (.pdf, .txt, .md).")
        sys.exit(1)

    print(f"{len(paths)} dosya bulundu, indeksleniyor...")
    chunk_count = ingest_files(paths)
    print(f"Tamamlandı: {chunk_count} parça vektör veritabanına eklendi.")


if __name__ == "__main__":
    main()
