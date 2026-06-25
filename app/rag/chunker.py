"""Doc-type-aware chunker that respects natural document boundaries."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:

    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text.split())


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = _count_tokens(self.content)


class SmartChunker:
    # Hard upper limit — recursive splitter uses this as its own default too
    MAX_TOKENS = 512

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def chunk(self, file_path: Path) -> list[Chunk]:
        """Detect doc type from path and dispatch to the right chunker."""
        content = file_path.read_text(encoding="utf-8")
        source = self._relative_source(file_path)
        parent = file_path.parent.name

        dispatch = {
            "faqs": self._chunk_faq,
            "policies": self._chunk_policy,
            "tickets": self._chunk_ticket,
            "api_docs": self._chunk_api_doc,
        }
        fn = dispatch.get(parent)
        if fn:
            return fn(content, source)
        return self._chunk_recursive(content, source)

    def chunk_directory(self, dir_path: Path) -> list[Chunk]:
        """Chunk every .md file found under dir_path."""
        chunks: list[Chunk] = []
        for md_file in sorted(dir_path.rglob("*.md")):
            chunks.extend(self.chunk(md_file))
        return chunks

    # ------------------------------------------------------------------ #
    # Doc-type chunkers                                                    #
    # ------------------------------------------------------------------ #

    def _chunk_faq(self, content: str, source: str) -> list[Chunk]:
        """One chunk per Q&A pair; question stored in metadata for BM25."""
        last_updated = self._extract_last_updated(content)
        title = self._extract_h1(content)
        stem = Path(source).stem

        # Split on level-2 headings that start with Q (## Q1. ...)
        parts = re.split(r"^(## Q\d+\.[^\n]*)", content, flags=re.MULTILINE)
        # parts: [preamble, heading1, body1, heading2, body2, ...]

        chunks: list[Chunk] = []
        idx = 1
        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            # Strip horizontal rules that separate Q&As
            body = re.sub(r"\n\s*---\s*\n", "\n\n", body).strip()

            q_match = re.match(r"## Q\d+\.\s*(.+)", heading)
            question = q_match.group(1).strip() if q_match else heading

            chunk_content = f"Q: {question}\n\nA: {body}"
            chunks.append(
                Chunk(
                    id=f"faq_{stem}_{idx:03d}",
                    content=chunk_content,
                    metadata={
                        "doc_type": "faq",
                        "source_file": source,
                        "category": stem,
                        "last_updated": last_updated,
                        "title": title,
                        "question": question,
                    },
                )
            )
            idx += 1

        return chunks

    def _chunk_policy(self, content: str, source: str) -> list[Chunk]:
        """One chunk per ## section; oversized sections are split recursively."""
        effective_date = self._extract_effective_date(content)
        title = self._extract_h1(content)
        stem = Path(source).stem

        # Split on level-2 headings
        parts = re.split(r"^(## .+)", content, flags=re.MULTILINE)
        # parts: [preamble, heading1, body1, heading2, body2, ...]

        chunks: list[Chunk] = []
        idx = 1

        # Include a meaningful preamble as the first chunk
        preamble = parts[0].strip()
        if _count_tokens(preamble) > 20:
            chunks.append(
                Chunk(
                    id=f"policy_{stem}_{idx:03d}",
                    content=preamble,
                    metadata={
                        "doc_type": "policy",
                        "source_file": source,
                        "category": stem,
                        "effective_date": effective_date,
                        "section_title": "Preamble",
                        "title": title,
                    },
                )
            )
            idx += 1

        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            body = (parts[i + 1].strip() if i + 1 < len(parts) else "")

            h_match = re.match(r"##\s+(.+)", heading)
            section_title = h_match.group(1).strip() if h_match else heading

            base_metadata = {
                "doc_type": "policy",
                "source_file": source,
                "category": stem,
                "effective_date": effective_date,
                "section_title": section_title,
                "title": title,
            }

            section_text = f"{heading}\n\n{body}".strip()
            if _count_tokens(section_text) <= self.MAX_TOKENS:
                chunks.append(
                    Chunk(id=f"policy_{stem}_{idx:03d}", content=section_text, metadata=base_metadata)
                )
                idx += 1
            else:
                # Recursively split the body, prepend heading to each sub-chunk
                sub_texts = self._split_text(body, max_tokens=self.MAX_TOKENS - _count_tokens(heading) - 4)
                for sub_text in sub_texts:
                    chunks.append(
                        Chunk(
                            id=f"policy_{stem}_{idx:03d}",
                            content=f"{heading}\n\n{sub_text}",
                            metadata=base_metadata,
                        )
                    )
                    idx += 1

        return chunks

    def _chunk_ticket(self, content: str, source: str) -> list[Chunk]:
        """Entire ticket = one chunk; metadata extracted from the header block."""
        stem = Path(source).stem

        def _extract(pattern: str) -> str:
            m = re.search(pattern, content, re.MULTILINE)
            return m.group(1).strip() if m else ""

        ticket_id = _extract(r"^#\s+Ticket:\s+(\S+)")
        status = _extract(r"^\*\*Status:\*\*\s*(.+)")
        category = _extract(r"^\*\*Category:\*\*\s*(.+)")
        resolution = _extract(r"^\*\*Resolution:\*\*\s*(.+)")
        created = _extract(r"^\*\*Created:\*\*\s*(.+)")

        return [
            Chunk(
                id=stem,
                content=content.strip(),
                metadata={
                    "doc_type": "ticket",
                    "source_file": source,
                    "ticket_id": ticket_id,
                    "status": status,
                    "category": category,
                    "resolution": resolution,
                    "created": created,
                },
            )
        ]

    def _chunk_api_doc(self, content: str, source: str) -> list[Chunk]:
        """One chunk per API endpoint block (### N. METHOD /path)."""
        stem = Path(source).stem

        endpoint_re = r"^(###\s+\d+\.\s+(?:GET|POST|PUT|DELETE|PATCH)\s+\S+)"
        parts = re.split(endpoint_re, content, flags=re.MULTILINE)
        # parts: [pre-endpoint content, heading1, body1, heading2, body2, ...]

        chunks: list[Chunk] = []
        idx = 1

        # Overview / intro section (everything before first ### endpoint)
        overview = parts[0].strip()
        if overview:
            chunks.append(
                Chunk(
                    id=f"api_doc_{stem}_{idx:03d}",
                    content=overview,
                    metadata={
                        "doc_type": "api_doc",
                        "source_file": source,
                        "category": stem,
                        "section_title": "Overview",
                        "endpoint": None,
                        "method": None,
                        "path": None,
                    },
                )
            )
            idx += 1

        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            body = (parts[i + 1].strip() if i + 1 < len(parts) else "")

            h_match = re.match(r"###\s+\d+\.\s+(GET|POST|PUT|DELETE|PATCH)\s+(\S+)", heading)
            method = h_match.group(1) if h_match else ""
            path = h_match.group(2) if h_match else ""

            chunk_content = f"{heading}\n\n{body}".strip()
            chunks.append(
                Chunk(
                    id=f"api_doc_{stem}_{idx:03d}",
                    content=chunk_content,
                    metadata={
                        "doc_type": "api_doc",
                        "source_file": source,
                        "category": stem,
                        "section_title": heading,
                        "endpoint": f"{method} {path}" if method else "",
                        "method": method,
                        "path": path,
                    },
                )
            )
            idx += 1

        return chunks

    def _chunk_recursive(
        self,
        content: str,
        source: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[Chunk]:
        """Fallback paragraph → sentence recursive splitter with token overlap."""
        stem = Path(source).stem
        texts = self._split_text(content, max_tokens=chunk_size, overlap=overlap)
        return [
            Chunk(
                id=f"recursive_{stem}_{idx:03d}",
                content=text,
                metadata={"doc_type": "text", "source_file": source, "category": stem},
            )
            for idx, text in enumerate(texts, start=1)
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _split_text(self, content: str, max_tokens: int = 512, overlap: int = 50) -> list[str]:
        """
        Split content into token-bounded pieces.
        Strategy: split on double newlines (paragraphs) first; if a paragraph
        still exceeds max_tokens, split on sentence boundaries.
        Overlap is applied as a trailing word-window carried into the next chunk.
        """
        # Atomic units: paragraphs, subdivided into sentences when oversized
        paragraphs = re.split(r"\n{2,}", content)
        units: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if _count_tokens(para) > max_tokens:
                # Split on sentence endings
                sentences = re.split(r"(?<=[.!?])\s+", para)
                units.extend(s.strip() for s in sentences if s.strip())
            else:
                units.append(para)

        texts: list[str] = []
        current: list[str] = []
        current_tokens = 0
        overlap_tail: list[str] = []  # words carried from previous chunk

        def _flush() -> None:
            nonlocal current, current_tokens, overlap_tail
            if not current:
                return
            text = "\n\n".join(current)
            texts.append(text)
            # Carry last `overlap` words into the next chunk
            all_words = text.split()
            overlap_tail = all_words[-overlap:] if len(all_words) > overlap else all_words
            current = []
            current_tokens = 0

        for unit in units:
            unit_tokens = _count_tokens(unit)
            if current_tokens + unit_tokens > max_tokens:
                _flush()
                # Start next chunk with overlap from previous
                if overlap_tail:
                    overlap_text = " ".join(overlap_tail)
                    current = [overlap_text]
                    current_tokens = _count_tokens(overlap_text)
            current.append(unit)
            current_tokens += unit_tokens

        _flush()
        return texts

    @staticmethod
    def _relative_source(file_path: Path) -> str:
        """Return path relative to knowledge_base/, or just the filename."""
        parts = file_path.parts
        try:
            kb_idx = parts.index("knowledge_base")
            return str(Path(*parts[kb_idx + 1 :]))
        except ValueError:
            return file_path.name

    @staticmethod
    def _extract_h1(content: str) -> str:
        m = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_last_updated(content: str) -> str:
        m = re.search(r"_Last updated:\s*([^_\n]+)_", content)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_effective_date(content: str) -> str:
        m = re.search(r"\*\*Effective Date:\s*([^*\n]+)\*\*", content)
        return m.group(1).strip() if m else ""
