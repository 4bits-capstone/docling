"""Heading detection and DoclingDocument enrichment for PDFs.

Two entry points:

* ``HeadingEnricher.detect(pdf_path) -> HeadingDetectionResult``  — standalone
  heading detection that uses only PyMuPDF (style + OCR) plus the optional
  DeepSeek API.  Returns a typed result dataclass with per-heading levels
  and provenance.
* ``HeadingEnricher.enrich(document, pdf_path) -> DoclingDocument``  —
  thin wrapper that runs ``detect()`` and merges the levels back onto an
  already-converted ``DoclingDocument`` (backward-compatible with the
  original API).

Detection signals, in order of priority:

1. Table-of-contents analysis (PyMuPDF only).  The embedded PDF outline
   is consulted first; if missing, the first N pages are scanned for a
   "Contents" page (with PyMuPDF OCR fallback for scanned PDFs).
2. DeepSeek style-level detection (``deepseek-V4-Flash`` by default,
   OpenAI-compatible chat completions).  Responses are cached on disk
   keyed by a hash of the model + heading inputs + TOC signature +
   PDF hash, so repeated runs are free.
3. Numbering-pattern analysis.  Decimal numbering in heading text
   (e.g. ``1.2.3``) is parsed and mapped to heading depth.
4. PyMuPDF font-size tier clustering.  Unique font sizes are bucketed
   into percentile tiers; bold/non-bold rules map each tier to a
   level.  Used as the final fallback.

The final level per heading is resolved with priority
``TOC > DeepSeek > numbering > tier`` and recorded on each ``HeadingRecord``
together with the candidate levels from all four sources.

Tier rules
----------
    tier 0 (largest),  non-bold  ->  Title (level 0)
    tier <= 1,         bold      ->  H1
    tier = 2,          bold      ->  H2
    tier >= 3,         bold      ->  H3
    tier = 3,          non-bold  ->  H4
    tier >= 4,         non-bold  ->  H5
    all other combinations      ->  fallback by tier index

Usage
-----
    # Standalone
    result = HeadingEnricher().detect("report.pdf")
    result.save_json("./headers/report_headers.json")

    # With a pre-converted DoclingDocument
    result = converter.convert("report.pdf")
    HeadingEnricher(write_headers_json=True).enrich(result.document, "report.pdf")

    # CLI
    python heading_enricher.py report.pdf
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import fitz
import httpx
import numpy as np
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import (
    DoclingDocument,
    Formatting,
    SectionHeaderItem,
    TitleItem,
)
from rapidfuzz import fuzz, process


API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MAX_SCAN_PAGES = 200
DEFAULT_HEADING_SIZE_RATIO = 1.15
DEFAULT_BODY_SAMPLE_PAGES = 5
LEVEL_TITLE = 0
LEVEL_MIN = 1
LEVEL_MAX = 5

@dataclass
class TocEntry:
    level: int
    title: str
    source: str  # "embedded" | "scanned"


@dataclass
class HeadingRecord:
    """One detected heading, with all level candidates and provenance."""

    text: str
    font_size: float
    is_bold: bool
    is_italic: bool
    final_level: int
    source: str  # "toc" | "deepseek" | "numbering" | "tier" | "consistency" | "ocr" | "none"
    tier: Optional[int] = None
    tier_level: Optional[int] = None
    toc_level: Optional[int] = None
    toc_score: Optional[float] = None
    deepseek_level: Optional[int] = None
    numbering_level: Optional[int] = None
    bbox: Optional[dict] = None  # serialisable BoundingBox

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HeadingDetectionResult:
    """All headings detected in a single PDF, plus its TOC."""

    pdf_path: Path
    toc_entries: list[TocEntry] = field(default_factory=list)
    headings: list[HeadingRecord] = field(default_factory=list)

    def _toc_to_dict(self, e: TocEntry) -> dict:
        d = asdict(e)
        d.pop("page_no", None)
        return d

    def to_dict(self) -> dict:
        return {
            "pdf_path": str(self.pdf_path),
            "toc_entries": [self._toc_to_dict(e) for e in self.toc_entries],
            "headings": [h.to_dict() for h in self.headings],
        }

    def save_json(self, out: Union[str, Path]) -> Path:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return out_path

    def to_docling_items(self) -> list[dict]:
        """Serialize headings in a schema compatible with Docling JSON exports."""
        return [
            {
                "text": h.text,
                "level": h.final_level,
                "font_size": round(h.font_size, 3),
                "is_bold": h.is_bold,
                "is_italic": h.is_italic,
                "source": h.source,
            }
            for h in self.headings
        ]

    @classmethod
    def from_dict(cls, data: dict) -> "HeadingDetectionResult":
        def _filter_fields(cls_, items: list[dict]) -> list:
            fields = set(cls_.__dataclass_fields__)
            return [cls_(**{k: v for k, v in item.items() if k in fields}) for item in items]

        toc_entries = _filter_fields(TocEntry, data.get("toc_entries", []))
        headings = _filter_fields(HeadingRecord, data.get("headings", []))
        return cls(
            pdf_path=Path(data["pdf_path"]),
            toc_entries=toc_entries,
            headings=headings,
        )


def _load_api_key(
    explicit: Optional[str] = None,
    env_path: Optional[Union[str, Path]] = None,
) -> str:
    """Resolve the DeepSeek API key.

    Order:
        1. ``explicit`` argument (if provided)
        2. ``DEEPSEEK_API_KEY`` environment variable
        3. ``DEEPSEEK_API_KEY = ...`` line in ``.env`` (default: project root)
    """
    if explicit:
        return explicit.strip()
    env_val = os.getenv(API_KEY_ENV)
    if env_val:
        return env_val.strip()
    if env_path is None:
        env_path = Path(".env")
    env_file = Path(env_path)
    if not env_file.exists():
        raise RuntimeError(
            "DeepSeek API key not found. Set the DEEPSEEK_API_KEY environment "
            "variable or add `DEEPSEEK_API_KEY = ...` to your .env file."
        )
    pattern = re.compile(rf"^\s*{re.escape(API_KEY_ENV)}\s*=\s*(.+?)\s*$", re.IGNORECASE)
    for line in env_file.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            value = match.group(1).strip().strip('"').strip("'")
            if value:
                return value
    raise RuntimeError(
        f"Could not find {API_KEY_ENV} in {env_file}. Set the environment "
        f"variable or add `{API_KEY_ENV} = ...` to the file."
    )


def normalize_title(text: str) -> str:
    """Normalize a heading or TOC title for matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b\u2032]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f\u2033]", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.:;,!?'\"()[]{}<>-")


class DeepSeekClient:
    """Thin OpenAI-compatible client for the DeepSeek chat-completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.api_key = _load_api_key(api_key)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> tuple[dict, str]:
        """Send a chat-completion request and return the parsed JSON object + raw content string."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server error {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"]
                return json.loads(raw_content), raw_content
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(1.5 ** attempt)
        raise RuntimeError(
            f"DeepSeek request failed after {self.max_retries + 1} attempts: {last_exc}"
        )


class HeadingEnricher:
    """Standalone heading detection and optional DoclingDocument enrichment.

    Signals, in order of priority:

    1. PDF table of contents (embedded outline or scanned "Contents" page)
    2. DeepSeek style-level classification
    3. PyMuPDF font-size tier clustering (fallback)
    """

    def __init__(
        self,
        n_tiers: int = 5,
        write_headers_json: bool = False,
        headers_output_dir: Union[str, Path] = "./headers",
        use_deepseek: bool = True,
        deepseek_api_key: Optional[str] = None,
        deepseek_model: str = DEFAULT_DEEPSEEK_MODEL,
        deepseek_timeout: float = 30.0,
        deepseek_cache_dir: Union[str, Path] = "./.cache/deepseek",
        fuzzy_match_threshold: float = 0.85,
        scan_contents_pages: int = 20,
        contents_keywords: Optional[list[str]] = None,
        max_scan_pages: int = DEFAULT_MAX_SCAN_PAGES,
        heading_size_ratio: float = DEFAULT_HEADING_SIZE_RATIO,
        body_sample_pages: int = DEFAULT_BODY_SAMPLE_PAGES,
    ):
        if n_tiers < 1 or n_tiers > 6:
            raise ValueError("n_tiers must be between 1 and 6")
        if not 0.0 <= fuzzy_match_threshold <= 1.0:
            raise ValueError("fuzzy_match_threshold must be in [0, 1]")
        if scan_contents_pages < 0:
            raise ValueError("scan_contents_pages must be >= 0")
        if max_scan_pages < 1:
            raise ValueError("max_scan_pages must be >= 1")
        if heading_size_ratio <= 1.0:
            raise ValueError("heading_size_ratio must be > 1.0")
        if body_sample_pages < 1:
            raise ValueError("body_sample_pages must be >= 1")
        self.n_tiers = n_tiers
        self.write_headers_json = write_headers_json
        self.headers_output_dir = Path(headers_output_dir)
        self.use_deepseek = use_deepseek
        self.deepseek_model = deepseek_model
        self.deepseek_timeout = deepseek_timeout
        self.deepseek_cache_dir = Path(deepseek_cache_dir)
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.scan_contents_pages = scan_contents_pages
        self.contents_keywords = contents_keywords or [
            "Table of Contents",
            "Contents",
            "TABLE OF CONTENTS",
            "CONTENTS",
        ]
        self.max_scan_pages = max_scan_pages
        self.heading_size_ratio = heading_size_ratio
        self.body_sample_pages = body_sample_pages

        self._client: Optional[DeepSeekClient] = None
        self._api_key = deepseek_api_key
        self._deepseek_disabled_reason: Optional[str] = None

    def detect(self, pdf_path: Union[str, Path]) -> HeadingDetectionResult:
        """Detect headings in a PDF and return a typed result.

        The result includes the TOC, every detected heading with its final
        level (0=Title, 1..5=H1..H5), and provenance for each level
        decision.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        try:
            toc_entries, toc_pages = self._extract_toc(doc)
            candidates = self._extract_candidate_headings(doc, toc_entries, toc_pages)
            if not candidates:
                return HeadingDetectionResult(pdf_path=pdf_path, toc_entries=toc_entries)

            heading_styles = self._extract_styles_from_candidates(candidates, doc)
            if not heading_styles:
                return HeadingDetectionResult(pdf_path=pdf_path, toc_entries=toc_entries)

            self._assign_tier_levels(heading_styles)
            self._assign_numbering_levels(heading_styles)
            deepseek_levels: dict[int, int] = {}
            if self.use_deepseek:
                deepseek_levels = self._deepseek_assign_levels(
                    heading_styles, toc_entries, pdf_path
                )
            self._reconcile_levels(heading_styles, toc_entries, deepseek_levels)
            self._validate_hierarchy(heading_styles)
            records = [self._to_record(hs) for hs in heading_styles]
        finally:
            doc.close()

        return HeadingDetectionResult(
            pdf_path=pdf_path,
            toc_entries=toc_entries,
            headings=records,
        )

    def enrich(
        self,
        document: DoclingDocument,
        pdf_path: Union[str, Path],
    ) -> DoclingDocument:
        """Detect headings and merge the levels onto a DoclingDocument.

        Args:
            document: A DoclingDocument produced by docling conversion.
            pdf_path: Path to the original PDF file.

        Returns:
            The same document with heading levels and formatting populated.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        result = self.detect(pdf_path)

        if result.headings:
            self._merge_into_document(document, result)

        if self.write_headers_json:
            self.headers_output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.headers_output_dir / f"{pdf_path.stem}_headers.json"
            result.save_json(out_path)

        return document

    def to_json(self, pdf_path: Union[str, Path], out_path: Union[str, Path]) -> Path:
        """Convenience: detect and save JSON in one call."""
        result = self.detect(pdf_path)
        return result.save_json(out_path)

    @classmethod
    def from_dict(cls, data: dict) -> HeadingDetectionResult:
        return HeadingDetectionResult.from_dict(data)

    # ------------------------------------------------------------------ TOC

    def _extract_toc(self, doc: fitz.Document) -> tuple[list[TocEntry], list[int]]:
        """Extract TOC entries via PyMuPDF only.

        Returns (toc_entries, toc_pages) where toc_pages are the page numbers
        for each entry, kept separate from the output dataclass.

        Strategy:
            1. Try the embedded PDF outline (``doc.get_toc()``).
            2. Otherwise, scan the first N pages for a "Contents" page
               and parse subsequent pages line-by-line (with OCR fallback).
        """
        entries: list[TocEntry] = []
        pages: list[int] = []
        try:
            raw = doc.get_toc(simple=False) or []
        except Exception:  # noqa: BLE001
            raw = []
        for row in raw:
            # PyMuPDF returns: [level, title, page, dest?]
            if len(row) >= 3:
                level, title, page = row[0], row[1], row[2]
                if title and page:
                    entries.append(
                        TocEntry(int(level), str(title).strip(), "embedded")
                    )
                    pages.append(int(page))
        if entries:
            return entries, pages

        if self.scan_contents_pages <= 0:
            return entries, pages

        n = min(self.scan_contents_pages, doc.page_count)
        contents_idx: Optional[int] = None
        for i in range(n):
            page_text = self._page_text_via_ocr(doc[i])
            for kw in self.contents_keywords:
                if kw in page_text:
                    contents_idx = i
                    break
            if contents_idx is not None:
                break
        if contents_idx is None:
            return entries, pages

        parse_span = min(contents_idx + 3, doc.page_count)
        combined_lines: list[str] = []
        for i in range(contents_idx, parse_span):
            combined_lines.extend(self._page_text_via_ocr(doc[i]).splitlines())

        dot_pattern = re.compile(
            r"""^
                (?P<num>\d+(?:\.\d+)*)?\s*
                (?P<title>.+?)\s*
                [.\u2024\u2025\u2026]{2,}\s*
                (?P<page>\d{1,4})\s*$
            """,
            re.VERBOSE,
        )
        trailing_pattern = re.compile(
            r"""^
                (?P<title>.+?)\s+
                (?P<page>\d{1,4})\s*$
            """,
            re.VERBOSE,
        )
        numbered_pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")

        for line in combined_lines:
            line = line.strip()
            if not line:
                continue
            m = dot_pattern.match(line)
            if not m:
                m = trailing_pattern.match(line)
            if not m:
                continue
            title = m.group("title").strip()
            try:
                page_no = int(m.group("page"))
            except (TypeError, ValueError):
                continue
            num_match = numbered_pattern.match(title)
            if num_match:
                level = num_match.group(1).count(".") + 1
                title = num_match.group(2).strip()
            else:
                level = 1
            if not title:
                continue
            entries.append(TocEntry(level, title, "scanned"))
            pages.append(page_no)

        # De-duplicate while preserving order
        seen: set[tuple[str, int]] = set()
        unique_entries: list[TocEntry] = []
        unique_pages: list[int] = []
        for e, p in zip(entries, pages):
            key = (e.title.lower(), e.level)
            if key in seen:
                continue
            seen.add(key)
            unique_entries.append(e)
            unique_pages.append(p)
        return unique_entries, unique_pages

    # ------------------------------------------------- Candidate detection

    def _extract_candidate_headings(
        self,
        doc: fitz.Document,
        toc_entries: list[TocEntry],
        toc_pages: Optional[list[int]] = None,
    ) -> list[dict]:
        """Return candidate heading dicts ready for style extraction.

        When the TOC is non-empty, candidates are derived from TOC entries
        (text + page).  Otherwise the first ``max_scan_pages`` pages are
        scanned for large-font text spans using a median-based threshold.
        """
        if toc_entries:
            return self._candidates_from_toc(toc_entries, toc_pages or [])
        return self._candidates_from_full_scan(doc)

    def _candidates_from_toc(self, toc_entries: list[TocEntry], toc_pages: list[int]) -> list[dict]:
        candidates: list[dict] = []
        for e, p in zip(toc_entries, toc_pages):
            candidates.append(
                {
                    "text": e.title,
                    "page_no": p,
                    "bbox": None,
                    "from_outline": True,
                }
            )
        return candidates

    def _candidates_from_full_scan(self, doc: fitz.Document) -> list[dict]:
        """Median-based candidate detection across pages."""
        body_size = self._estimate_body_size(doc)
        if body_size <= 0:
            return []
        threshold = body_size * self.heading_size_ratio

        max_pages = min(self.max_scan_pages, doc.page_count)
        candidates: list[dict] = []
        seen: set[tuple[int, str]] = set()
        for i in range(max_pages):
            page = doc[i]
            blocks = self._page_dict_with_ocr(page)
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text, line_size, is_bold, is_italic, line_bbox = self._summarize_line(
                        line
                    )
                    if not line_text or line_size <= 0:
                        continue
                    if line_size < threshold:
                        continue
                    if len(line_text) < 2 or len(line_text) > 200:
                        continue
                    norm = normalize_title(line_text)
                    key = (i + 1, norm)
                    if not norm or key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "text": line_text,
                            "page_no": i + 1,
                            "bbox": self._bbox_to_dict(line_bbox) if line_bbox else None,
                            "from_outline": False,
                        }
                    )
        return candidates

    @staticmethod
    def _summarize_line(line: dict) -> tuple[str, float, bool, bool, Optional[tuple]]:
        """Return (text, max_size, bold, italic, bbox) for a PyMuPDF line."""
        spans = line.get("spans", []) or []
        if not spans:
            return "", 0.0, False, False, None
        text = "".join(s.get("text", "") for s in spans).strip()
        if not text:
            return "", 0.0, False, False, None
        max_size = 0.0
        bold = False
        italic = False
        for s in spans:
            sz = float(s.get("size", 0.0))
            if sz > max_size:
                max_size = sz
                font = s.get("font", "")
                flags = s.get("flags", 0)
                bold = HeadingEnricher._is_bold(font, flags)
                italic = HeadingEnricher._is_italic(font, flags)
        bbox = line.get("bbox")
        return text, max_size, bold, italic, bbox

    def _estimate_body_size(self, doc: fitz.Document) -> float:
        """Median text size of long lines on a few early pages."""
        n = min(self.body_sample_pages, doc.page_count)
        sizes: list[float] = []
        for i in range(n):
            blocks = self._page_dict_with_ocr(doc[i])
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    text, size, _b, _i, _bb = self._summarize_line(line)
                    if size <= 0 or not text:
                        continue
                    if len(text) < 30:  # body lines tend to be long
                        continue
                    sizes.append(size)
        if not sizes:
            return 0.0
        return float(np.median(sizes))

    @staticmethod
    def _bbox_to_dict(bbox) -> Optional[dict]:
        if bbox is None:
            return None
        try:
            l, t, r, b = (float(x) for x in bbox)
            return {"l": l, "t": t, "r": r, "b": b}
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------ Style extraction

    def _extract_styles_from_candidates(
        self,
        candidates: list[dict],
        doc: fitz.Document,
    ) -> list[dict]:
        """Resolve font size, bold, italic for each candidate via PyMuPDF."""
        styles: list[dict] = []
        for c in candidates:
            style = self._extract_candidate_style(c, doc)
            if style is not None:
                styles.append(style)
        return styles

    def _extract_candidate_style(
        self,
        candidate: dict,
        doc: fitz.Document,
    ) -> Optional[dict]:
        """Resolve font properties for a candidate heading.

        For outline-derived candidates (no bbox), falls back to the largest
        non-empty span on the candidate's page that contains the candidate
        text.  For span-derived candidates, reuses the bbox.
        """
        page_no = int(candidate["page_no"])
        text = candidate.get("text", "") or ""
        if page_no < 1 or page_no > doc.page_count:
            return None

        page = doc[page_no - 1]
        blocks = self._page_dict_with_ocr(page)
        if not blocks:
            return None

        spans = self._collect_spans(blocks)
        if not spans:
            return None

        if candidate.get("from_outline"):
            best = self._find_span_by_text(spans, text)
        else:
            best = self._find_span_by_bbox(spans, candidate.get("bbox"))

        if best is None:
            best = self._find_largest_span(spans)
        if best is None:
            return None

        font_name = best.get("font", "")
        flags = best.get("flags", 0)
        return {
            "text": text,
            "page_no": page_no,
            "font_size": float(best.get("size", 0.0)) or 0.0,
            "is_bold": self._is_bold(font_name, flags),
            "is_italic": self._is_italic(font_name, flags),
            "bbox": candidate.get("bbox"),
        }

    @staticmethod
    def _collect_spans(blocks: list[dict]) -> list[dict]:
        spans: list[dict] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text"):
                        spans.append(span)
        return spans

    def _find_span_by_text(
        self,
        spans: list[dict],
        text: str,
    ) -> Optional[dict]:
        """Find the span whose text best matches ``text`` (largest first)."""
        norm_target = normalize_title(text)
        if not norm_target:
            return None
        best: Optional[dict] = None
        best_score = 0.0
        for span in spans:
            span_text = span.get("text", "")
            if not span_text:
                continue
            score = fuzz.token_set_ratio(norm_target, normalize_title(span_text))
            if score > best_score:
                best = span
                best_score = score
                if score >= 99.0:
                    return span
        if best_score < 60.0:
            return None
        return best

    def _find_span_by_bbox(
        self,
        spans: list[dict],
        bbox: Optional[dict],
    ) -> Optional[dict]:
        if not bbox:
            return None
        try:
            l, t, r, b = bbox["l"], bbox["t"], bbox["r"], bbox["b"]
        except (KeyError, TypeError):
            return None
        best: Optional[dict] = None
        best_overlap = 0.0
        for span in spans:
            sb = span.get("bbox")
            if not sb:
                continue
            sx0, sy0, sx1, sy1 = sb
            overlap_x = max(0.0, min(sx1, r) - max(sx0, l))
            overlap_y = max(0.0, min(sy1, b) - max(sy0, t))
            overlap = overlap_x * overlap_y
            if overlap > best_overlap:
                best_overlap = overlap
                best = span
        return best

    @staticmethod
    def _find_largest_span(spans: list[dict]) -> Optional[dict]:
        if not spans:
            return None
        return max(spans, key=lambda s: float(s.get("size", 0.0)))

    # --------------------------------------------------- PyMuPDF OCR helpers

    def _page_text_via_ocr(self, page: fitz.Page) -> str:
        text = page.get_text("text") or ""
        if text.strip():
            return text
        try:
            tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
        except Exception:  # noqa: BLE001
            return ""
        try:
            return tp.getText("text") or ""
        finally:
            try:
                tp = None
            except Exception:  # noqa: BLE001
                pass

    def _page_dict_with_ocr(self, page: fitz.Page) -> list[dict]:
        blocks = page.get_text("dict").get("blocks", [])
        if blocks:
            return blocks
        try:
            tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
        except Exception:  # noqa: BLE001
            return []
        try:
            return tp.getText("dict").get("blocks", [])
        finally:
            try:
                tp = None
            except Exception:  # noqa: BLE001
                pass

    # -------------------------------------------------- Font property helpers

    @staticmethod
    def _is_bold(font_name: str, flags: int) -> bool:
        bold_keywords = ["Bold", "Heavy", "Black", "Demi", "ExtraBold"]
        name_has_bold = any(kw in font_name for kw in bold_keywords)
        flag_has_bold = bool(flags & 16)
        return name_has_bold or flag_has_bold

    @staticmethod
    def _is_italic(font_name: str, flags: int) -> bool:
        italic_keywords = ["Italic", "Oblique", "It"]
        name_has_italic = any(kw in font_name for kw in italic_keywords)
        flag_has_italic = bool(flags & 2)
        return name_has_italic or flag_has_italic

    # -------------------------------------------------------- Tier levels

    def _assign_tier_levels(self, heading_styles: list[dict]) -> None:
        sizes = sorted(set(s["font_size"] for s in heading_styles if s["font_size"] > 0), reverse=True)
        if not sizes:
            for hs in heading_styles:
                hs["tier"] = None
                hs["tier_level"] = None
            return

        tier_of_size: dict[float, int] = {}
        if len(sizes) >= self.n_tiers:
            percentiles = [100 * (i + 1) / self.n_tiers for i in range(self.n_tiers - 1)]
            boundaries = np.percentile(sizes, percentiles)
            for size in sizes:
                tier = 0
                for bound in boundaries:
                    if size <= bound:
                        tier += 1
                tier_of_size[size] = min(tier, self.n_tiers - 1)
        else:
            for i, size in enumerate(sizes):
                tier_of_size[size] = i

        for hs in heading_styles:
            size = hs["font_size"]
            if size <= 0:
                hs["tier"] = None
                hs["tier_level"] = None
                continue
            tier = tier_of_size[size]
            is_bold = hs["is_bold"]

            if tier == 0 and not is_bold:
                tier_level = LEVEL_TITLE
            elif tier <= 1 and is_bold:
                tier_level = 1
            elif tier == 2 and is_bold:
                tier_level = 2
            elif tier >= 3 and is_bold:
                tier_level = 3
            elif tier == 3 and not is_bold:
                tier_level = 4
            elif tier >= 4 and not is_bold:
                tier_level = 5
            else:
                tier_level = min(tier + 1, self.n_tiers)

            hs["tier"] = tier
            hs["tier_level"] = tier_level

    # ---------------------------------------------------- Numbering levels

    @staticmethod
    def _extract_numbering_level(text: str) -> Optional[int]:
        if not text:
            return None
        m = re.match(r"^(\d+(?:\.\d+)*)\s*[.)]?\s+", text)
        if not m:
            return None
        segments = m.group(1).split(".")
        return len(segments)

    def _assign_numbering_levels(self, heading_styles: list[dict]) -> None:
        for hs in heading_styles:
            hs["numbering_level"] = self._extract_numbering_level(hs.get("text", ""))

    # --------------------------------------------------------- DeepSeek

    def _ensure_client(self) -> Optional[DeepSeekClient]:
        if self._client is not None:
            return self._client
        if self._deepseek_disabled_reason is not None:
            return None
        try:
            self._client = DeepSeekClient(
                api_key=self._api_key,
                model=self.deepseek_model,
                timeout=self.deepseek_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            self._deepseek_disabled_reason = str(exc)
            print(f"[HeadingEnricher] DeepSeek disabled: {exc}")
            return None
        return self._client

    @staticmethod
    def _toc_signature(toc_entries: list[TocEntry]) -> str:
        if not toc_entries:
            return "empty"
        items = [(e.level, e.title, e.source) for e in toc_entries]
        blob = json.dumps(items, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _pdf_signature(pdf_path: Path) -> str:
        try:
            h = hashlib.sha256()
            with open(pdf_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return "unreadable"

    def _cache_key(
        self,
        heading_styles: list[dict],
        toc_sig: str,
        pdf_sig: str,
    ) -> str:
        payload = {
            "model": self.deepseek_model,
            "pdf": pdf_sig,
            "toc": toc_sig,
            "items": [
                {
                    "text": hs["text"],
                    "font_size": round(hs["font_size"], 3),
                    "is_bold": bool(hs["is_bold"]),
                    "is_italic": bool(hs["is_italic"]),
                    "page_no": int(hs["page_no"]),
                }
                for hs in heading_styles
            ],
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _deepseek_assign_levels(
        self,
        heading_styles: list[dict],
        toc_entries: list[TocEntry],
        pdf_path: Path,
    ) -> dict[int, int]:
        client = self._ensure_client()
        if client is None:
            return {}

        self.deepseek_cache_dir.mkdir(parents=True, exist_ok=True)
        toc_sig = self._toc_signature(toc_entries)
        pdf_sig = self._pdf_signature(pdf_path)
        key = self._cache_key(heading_styles, toc_sig, pdf_sig)
        cache_path = self.deepseek_cache_dir / f"{key}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                levels = cached.get("levels", {})
                return {
                    int(k): int(v)
                    for k, v in levels.items()
                    if self._is_valid_level(v)
                }
            except (json.JSONDecodeError, ValueError):
                pass

        def _numbering_prefix(text: str) -> Optional[str]:
            m = re.match(r"^(\d+(?:\.\d+)*)", text)
            return m.group(1) if m else None

        items = [
            {
                "idx": i,
                "text": hs["text"],
                "font_size": round(hs["font_size"], 2),
                "bold": bool(hs["is_bold"]),
                "italic": bool(hs["is_italic"]),
                "page": int(hs["page_no"]),
                "numbering": _numbering_prefix(hs.get("text", "")),
            }
            for i, hs in enumerate(heading_styles)
        ]
        toc_hint = [
            {"level": e.level, "title": e.title}
            for e in toc_entries[:200]
        ]
        system = (
            "You are a PDF heading-hierarchy classifier. You receive a JSON "
            "object with `items` (an ordered list of candidate headings) and "
            "an optional `toc` array describing the document's table of "
            "contents. Return JSON of the form "
            "`{\"levels\": {<idx>: <level>, ...}}` where level is one of "
            "0=Title, 1=H1, 2=H2, 3=H3, 4=H4, 5=H5.\n\n"
            "Use these signals, in priority order:\n"
            "1. Numbering pattern (e.g. '1.2.3' → H3). Items with deeper numbering "
            "should have deeper levels.\n"
            "2. TOC entries — use the TOC level when a heading clearly matches.\n"
            "3. Visual style (font size, bold) as a tiebreaker.\n\n"
            "The output hierarchy must be logically consistent. For example, an H1 "
            "should not be directly followed by H3 without an H2 in between. "
            "Items at the same numbering depth (e.g. '1.1' and '1.2') should have "
            "the same level. Output only the JSON object."
        )
        user = json.dumps({"items": items, "toc": toc_hint}, ensure_ascii=False)

        try:
            data, raw_content = client.chat_json(system=system, user=user)
        except Exception as exc:  # noqa: BLE001
            print(f"[HeadingEnricher] DeepSeek call failed: {exc}")
            return {}

        levels_raw = data.get("levels") if isinstance(data, dict) else None
        if not isinstance(levels_raw, dict):
            return {}
        result: dict[int, int] = {}
        for k, v in levels_raw.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if not self._is_valid_level(v):
                continue
            if 0 <= idx < len(heading_styles):
                result[idx] = int(v)
        if result:
            try:
                cache_path.write_text(
                    json.dumps(
                        {
                            "levels": {str(k): v for k, v in result.items()},
                            "raw_response": raw_content,
                            "prompt": user,
                            "model": self.deepseek_model,
                            "timestamp": time.time(),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass
        return result

    @staticmethod
    def _is_valid_level(v: Any) -> bool:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return False
        return 0 <= iv <= 5

    # ---------------------------------------------------------- Reconcile

    def _reconcile_levels(
        self,
        heading_styles: list[dict],
        toc_entries: list[TocEntry],
        deepseek_levels: dict[int, int],
    ) -> None:
        """Resolve the final level per heading:  TOC > DeepSeek > tier."""
        toc_index: dict[str, TocEntry] = {}
        toc_titles: list[str] = []
        for e in toc_entries:
            key = normalize_title(e.title)
            if not key:
                continue
            toc_index.setdefault(key, e)
            toc_titles.append(key)

        for i, hs in enumerate(heading_styles):
            tier_level = hs.get("tier_level")
            numbering_level = hs.get("numbering_level")

            norm = normalize_title(hs.get("text", ""))
            toc_match: Optional[TocEntry] = None
            match_score: float = 0.0
            if norm and toc_titles:
                if norm in toc_index:
                    toc_match = toc_index[norm]
                    match_score = 100.0
                else:
                    best = process.extractOne(
                        norm,
                        toc_titles,
                        scorer=fuzz.token_set_ratio,
                    )
                    if best is not None:
                        candidate_key, score, _ = best
                        if score >= self.fuzzy_match_threshold * 100.0:
                            toc_match = toc_index.get(candidate_key)
                            match_score = float(score)

            deepseek_level = deepseek_levels.get(i)
            hs["toc_level"] = toc_match.level if toc_match is not None else None
            hs["toc_score"] = round(match_score, 2) if toc_match is not None else None
            hs["deepseek_level"] = deepseek_level
            hs["numbering_level"] = numbering_level

            if toc_match is not None:
                final = toc_match.level
                source = "toc"
            elif deepseek_level is not None:
                final = deepseek_level
                source = "deepseek"
            elif numbering_level is not None:
                final = numbering_level
                source = "numbering"
            elif tier_level is not None:
                final = tier_level
                source = "tier"
            else:
                final = 1
                source = "none"

            hs["final_level"] = final
            hs["source"] = source

    # ------------------------------------------------------- Consistency pass

    def _validate_hierarchy(self, heading_styles: list[dict]) -> None:
        """Post-reconciliation validation that fixes illogical sequences.

        1. Level-skip clamping: a heading cannot jump more than one level
           deeper than the previous heading (e.g. H1 → H3 → H2).
        2. Numbering-group consistency: headings that share the same numbered
           parent (e.g. ``1.1`` and ``1.2`` under parent ``1``) must have
           the same level.
        """
        if not heading_styles:
            return

        # --- Rule 1: level-skip clamping ---
        prev_level = heading_styles[0]["final_level"]
        for hs in heading_styles[1:]:
            level = hs["final_level"]
            if level > prev_level + 1:
                hs["final_level"] = prev_level + 1
                hs["source"] = "consistency"
                level = prev_level + 1
            prev_level = level

        # --- Rule 2: numbering-group consistency ---
        groups: dict[str, list[dict]] = {}
        for hs in heading_styles:
            prefix = hs.get("numbering_level")
            if prefix is None:
                continue
            segments = None
            m = re.match(r"^(\d+(?:\.\d+)*)", hs.get("text", ""))
            if m:
                segments = m.group(1)
            if not segments:
                continue
            dot_count = segments.count(".")
            if dot_count < 1:
                continue
            parent = segments.rsplit(".", 1)[0]
            groups.setdefault(parent, []).append(hs)

        for parent, items in groups.items():
            if len(items) <= 1:
                continue
            level_counts = collections.Counter(
                item["final_level"] for item in items
            )
            majority_level = level_counts.most_common(1)[0][0]
            for item in items:
                if item["final_level"] != majority_level:
                    item["final_level"] = majority_level
                    item["source"] = "consistency"

    def _to_record(self, hs: dict) -> HeadingRecord:
        return HeadingRecord(
            text=hs.get("text", ""),
            font_size=float(hs.get("font_size", 0.0)),
            is_bold=bool(hs.get("is_bold", False)),
            is_italic=bool(hs.get("is_italic", False)),
            final_level=int(hs.get("final_level", 1)),
            source=str(hs.get("source", "none")),
            tier=hs.get("tier"),
            tier_level=hs.get("tier_level"),
            toc_level=hs.get("toc_level"),
            toc_score=hs.get("toc_score"),
            deepseek_level=hs.get("deepseek_level"),
            numbering_level=hs.get("numbering_level"),
            bbox=hs.get("bbox"),
        )

    # --------------------------------------------- DoclingDocument merge

    def _merge_into_document(
        self,
        document: DoclingDocument,
        result: HeadingDetectionResult,
    ) -> None:
        """Apply detected levels and formatting back to SectionHeaderItems."""
        existing = self._collect_headings(document)
        if not existing:
            return
        # Match by normalized title; fall back to page-order alignment.
        existing_norm = [normalize_title(getattr(h, "text", "") or "") for h in existing]
        rec_for_heading: dict[int, HeadingRecord] = {}
        for i, h in enumerate(existing):
            rec = self._match_record_to_heading(h, result, fallback_index=i)
            if rec is not None:
                rec_for_heading[i] = rec

        for i, heading in enumerate(existing):
            rec = rec_for_heading.get(i)
            if rec is None:
                continue
            level = rec.final_level
            if level == LEVEL_TITLE:
                heading.level = 1
            else:
                heading.level = level
            heading.formatting = Formatting(bold=rec.is_bold, italic=rec.is_italic)

        self._convert_title_items(document, list(rec_for_heading.values()))

    def _match_record_to_heading(
        self,
        heading,
        result: HeadingDetectionResult,
        fallback_index: int,
    ) -> Optional[HeadingRecord]:
        text_norm = normalize_title(getattr(heading, "text", "") or "")
        if text_norm and result.headings:
            best = process.extractOne(
                text_norm,
                [normalize_title(h.text) for h in result.headings],
                scorer=fuzz.token_set_ratio,
            )
            if best is not None and best[1] >= self.fuzzy_match_threshold * 100.0:
                return result.headings[best[2]]
        if 0 <= fallback_index < len(result.headings):
            return result.headings[fallback_index]
        return None

    @staticmethod
    def _collect_headings(document: DoclingDocument) -> list:
        return [
            item
            for item, _level in document.iterate_items()
            if hasattr(item, "level") and hasattr(item, "prov")
        ]

    @staticmethod
    def _convert_title_items(
        document: DoclingDocument,
        records: list[HeadingRecord],
    ) -> None:
        for rec in records:
            if rec.final_level != LEVEL_TITLE:
                continue
            # Re-resolve the heading to swap it for a TitleItem
            for item, _level in document.iterate_items():
                if not isinstance(item, SectionHeaderItem):
                    continue
                if normalize_title(getattr(item, "text", "") or "") == normalize_title(
                    rec.text
                ):
                    title_item = TitleItem(
                        text=item.text,
                        self_ref=item.self_ref,
                        orig=item.orig,
                        prov=item.prov,
                        formatting=item.formatting,
                        hyperlink=item.hyperlink,
                    )
                    document.replace_item(new_item=title_item, old_item=item)
                    break


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect heading hierarchy in a PDF (PyMuPDF + optional DeepSeek).",
    )
    parser.add_argument("pdf", type=str, help="Path to the input PDF.")
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=None,
        help="Output JSON path (default: <pdf_stem>_headers.json next to the PDF).",
    )
    parser.add_argument(
        "--no-deepseek",
        action="store_true",
        help="Disable DeepSeek-based heading-level classification.",
    )
    parser.add_argument(
        "--deepseek-model",
        type=str,
        default=DEFAULT_DEEPSEEK_MODEL,
        help=f"DeepSeek model id (default: {DEFAULT_DEEPSEEK_MODEL}).",
    )
    parser.add_argument(
        "--max-scan-pages",
        type=int,
        default=DEFAULT_MAX_SCAN_PAGES,
        help=f"Max pages to full-page scan when no TOC is present (default: {DEFAULT_MAX_SCAN_PAGES}).",
    )
    parser.add_argument(
        "--heading-size-ratio",
        type=float,
        default=DEFAULT_HEADING_SIZE_RATIO,
        help=f"Font-size multiplier over median body size to qualify as heading (default: {DEFAULT_HEADING_SIZE_RATIO}).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    enricher = HeadingEnricher(
        use_deepseek=not args.no_deepseek,
        deepseek_model=args.deepseek_model,
        max_scan_pages=args.max_scan_pages,
        heading_size_ratio=args.heading_size_ratio,
    )
    result = enricher.detect(pdf_path)

    out_path = Path(args.out) if args.out else pdf_path.with_name(f"{pdf_path.stem}_headers.json")
    saved = result.save_json(out_path)

    counts: dict[str, int] = {}
    for h in result.headings:
        counts[h.source] = counts.get(h.source, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"{pdf_path.name}: {len(result.headings)} headings ({summary}) -> {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
