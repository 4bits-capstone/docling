"""Post-process a DoclingDocument to enrich headings with font-based classification.

Uses PyMuPDF to extract font size and styling from the original PDF, matches
spans to Docling TextItems via bounding box overlap, and classifies heading
levels using configurable relative font-size rules.
"""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Sequence

import fitz
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.document import BaseMeta, DoclingDocument, Formatting


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FontSpan:
    """A single text span extracted from PyMuPDF."""

    text: str
    font: str
    size: float
    bold: bool
    italic: bool
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1), top-left
    page_no: int  # 1-indexed, matching docling convention


@dataclasses.dataclass
class AggregatedFontInfo:
    """Aggregated font info for one TextItem."""

    font_size: float
    font_name: str
    bold: bool
    italic: bool
    span_count: int
    char_count: int


# ---------------------------------------------------------------------------
# Default rules
# ---------------------------------------------------------------------------

DEFAULT_RULES: list[dict] = [
    # H1: biggest bold heading (title is handled separately by docling layout)
    {"level": 1, "min_scale": 1.4, "require_bold": True},
    # H2: next bold heading
    {"level": 2, "min_scale": 1.2, "require_bold": True},
    # H3: smaller bold heading
    {"level": 3, "min_scale": 1.05, "require_bold": True},
    # H4: non-bold, between heading and body size (bold text already caught above)
    {"level": 4, "min_scale": 1.0},
    # H5: non-bold, slightly above body
    {"level": 5, "min_scale": 0.9},
]


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_spans(pdf_path: str) -> list[FontSpan]:
    """Extract all text spans from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    spans: list[FontSpan] = []

    for page_no in range(len(doc)):
        page = doc[page_no]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] != 0:  # skip images
                continue
            for line in block.get("lines", []):
                for span in line["spans"]:
                    flags = span["flags"]
                    spans.append(
                        FontSpan(
                            text=span["text"],
                            font=span["font"],
                            size=span["size"],
                            bold=bool(flags & 2**4),     # bit 4
                            italic=bool(flags & 2**1),   # bit 1
                            bbox=span["bbox"],
                            page_no=page_no + 1,  # 1-indexed
                        )
                    )
    return spans


def _compute_body_size(spans: list[FontSpan]) -> float:
    """Detect the dominant body font size across the whole document.

    Uses iterative trimming: compute the median of all sizes (≥6pt, weighted
    by character count), then filter to the cluster within 0.8×–1.2× of it.
    """
    sizes: list[float] = []
    for s in spans:
        if s.size >= 6:
            sizes.extend([s.size] * len(s.text))

    if not sizes:
        return 11.0  # safe fallback

    median = statistics.median(sizes)
    filtered = [s for s in sizes if 0.8 * median <= s <= 1.2 * median]

    return statistics.mean(filtered) if filtered else median


# ---------------------------------------------------------------------------
# Span–item matching
# ---------------------------------------------------------------------------

def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection over minimum-area (IoU-like using smaller bbox as denominator)."""
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return intersection / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0


def _get_item_bbox(
    item,
    page_height: float,
) -> tuple[float, float, float, float]:
    """Extract (x0, y0, x1, y1) from a TextItem's provenance bboxes.

    When multiple provenance entries exist (e.g. a wrapped paragraph) the
    union of all bboxes is returned.  Coordinates are converted from
    docling's origin to the top-left origin used by PyMuPDF.
    """
    if not item.prov:
        return (0, 0, 0, 0)

    xs: list[float] = []
    ys: list[float] = []

    for p in item.prov:
        bbox = p.bbox
        if bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
            xs.extend([bbox.l, bbox.r])
            ys.extend([page_height - bbox.t, page_height - bbox.b])
        else:
            xs.extend([bbox.l, bbox.r])
            ys.extend([bbox.t, bbox.b])

    return (min(xs), min(ys), max(xs), max(ys))


def _match_spans_to_items(
    spans: list[FontSpan],
    doc: DoclingDocument,
    iou_threshold: float = 0.3,
) -> dict[str, AggregatedFontInfo]:
    """Match PyMuPDF spans to docling TextItems by bbox overlap.

    Returns a dict keyed by ``item.self_ref``.
    """
    # Index spans by page
    spans_by_page: dict[int, list[FontSpan]] = {}
    for s in spans:
        spans_by_page.setdefault(s.page_no, []).append(s)

    # Build page-height lookup
    page_heights: dict[int, float] = {
        p.page_no: p.size.height
        for p in (doc.pages or {}).values()
        if p.size is not None
    }

    result: dict[str, AggregatedFontInfo] = {}

    for item in doc.texts:
        if not item.prov:
            continue

        page_no = item.prov[0].page_no
        page_height = page_heights.get(page_no, 792.0)

        item_bbox = _get_item_bbox(item, page_height)
        page_spans = spans_by_page.get(page_no, [])
        matching = [
            s for s in page_spans
            if _bbox_iou(item_bbox, s.bbox) > iou_threshold
        ]

        if not matching:
            continue

        total_chars = sum(len(s.text) for s in matching)
        if total_chars == 0:
            continue

        size_sum = sum(s.size * len(s.text) for s in matching)
        bold_chars = sum(len(s.text) for s in matching if s.bold)
        italic_chars = sum(len(s.text) for s in matching if s.italic)

        # Pick the most-representative font name (by char count)
        font_counts: dict[str, int] = {}
        for s in matching:
            font_counts[s.font] = font_counts.get(s.font, 0) + len(s.text)
        best_font = max(font_counts, key=font_counts.get)

        result[item.self_ref] = AggregatedFontInfo(
            font_size=size_sum / total_chars,
            font_name=best_font,
            bold=(bold_chars / total_chars) > 0.5,
            italic=(italic_chars / total_chars) > 0.5,
            span_count=len(matching),
            char_count=total_chars,
        )

    return result


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify_item(
    info: AggregatedFontInfo,
    body_size: float,
    rules: Sequence[dict],
) -> int:
    """Determine heading level from font info and rules.

    Returns heading *level* (1-5), or 0 for non-headings.
    """
    if body_size <= 0:
        return 0

    ratio = info.font_size / body_size

    for rule in rules:
        if ratio >= rule["min_scale"]:
            if rule.get("require_bold", False) and not info.bold:
                continue
            return rule["level"]

    return 0


# ---------------------------------------------------------------------------
# Main enricher
# ---------------------------------------------------------------------------

class FontHeadingEnricher:
    """Post-process a DoclingDocument to classify headings from font data.

    Parameters
    ----------
    rules :
        Ordered list of heading rules. Each rule is a dict with keys:
        ``level`` (int), ``label`` (DocItemLabel), ``min_scale`` (float),
        and optionally ``require_bold`` (bool). Rules are checked in order;
        the first match wins.
    iou_threshold :
        Minimum intersection-over-min-area ratio for matching a PyMuPDF span
        to a docling TextItem.
    meta_namespace :
        Namespace prefix for extra fields stored in ``BaseMeta``.
    """

    def __init__(
        self,
        rules: Sequence[dict] | None = None,
        iou_threshold: float = 0.3,
        meta_namespace: str = "hf",
    ):
        self.rules = list(rules or DEFAULT_RULES)
        self.iou_threshold = iou_threshold
        self.ns = meta_namespace

    def enrich(self, doc: DoclingDocument, pdf_path: str) -> DoclingDocument:
        """Run font-based heading enrichment on *doc*.

        The document is modified *in-place* and also returned for convenience.

        Parameters
        ----------
        doc :
            The docling document to enrich.
        pdf_path :
            Path to the original PDF file.
        """
        spans = _extract_spans(pdf_path)
        body_size = _compute_body_size(spans)
        font_map = _match_spans_to_items(spans, doc, self.iou_threshold)

        for item in doc.texts:
            info = font_map.get(item.self_ref)
            if info is None:
                continue

            # Set formatting (bold / italic)
            item.formatting = Formatting(
                bold=info.bold,
                italic=info.italic,
            )

            # Classify heading level
            level = _classify_item(info, body_size, self.rules)

            # Build extra metadata (stored in meta so downstream code
            # like convert_to_ls.py's map_label() can read it)
            extra: dict[str, object] = {
                f"{self.ns}__font_size": round(info.font_size, 1),
                f"{self.ns}__font_name": info.font_name,
                f"{self.ns}__body_size": round(body_size, 1),
                f"{self.ns}__heading_level": level,
            }

            if item.meta is None:
                item.meta = BaseMeta(**extra)
            else:
                item.meta.__pydantic_extra__.update(extra)

        return doc
