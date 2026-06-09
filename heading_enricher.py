"""Heading level enricher for DoclingDocument.

Uses PyMuPDF to extract font size and bold/italic properties from
SectionHeaderItem bounding boxes, then assigns heading levels via
rule-based font size rank analysis with bold/non-bold overrides.

Rules (applied to font-size ranks from largest to smallest):
    rank 0 (largest), non-bold  ->  TitleItem
    rank 1 (second),  bold      ->  H1
    rank 2 (third),   bold      ->  H2
    rank 3 (fourth),  bold      ->  H3
    rank 3 (fourth),  non-bold  ->  H4
    all other ranks             ->  fallback by rank alone

Usage:
    result = converter.convert("report.pdf")
    HeadingEnricher().enrich(result.document, "report.pdf")
"""

from pathlib import Path
from typing import Optional, Union

import fitz
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import (
    DoclingDocument,
    Formatting,
    SectionHeaderItem,
    TitleItem,
)


class HeadingEnricher:
    """Assign heading levels (Title/H1-H4) to SectionHeaderItems using
    PyMuPDF font size and bold analysis.

    Runs as a post-processing step after docling conversion.
    Only operates on items already classified as SECTION_HEADER by docling's
    layout model — it does not detect new headings.
    """

    def __init__(self, n_tiers: int = 5):
        if n_tiers < 1 or n_tiers > 6:
            raise ValueError("n_tiers must be between 1 and 6")
        self.n_tiers = n_tiers

    def enrich(
        self,
        document: DoclingDocument,
        pdf_path: Union[str, Path],
    ) -> DoclingDocument:
        """Assign heading levels to SectionHeaderItems based on font properties.

        Args:
            document: A DoclingDocument produced by docling conversion.
            pdf_path: Path to the original PDF file.

        Returns:
            The same document with heading levels and formatting populated.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        headings = self._collect_headings(document)
        if not headings:
            return document

        heading_styles = self._extract_styles(headings, pdf_path)
        if not heading_styles:
            return document

        self._assign_levels(heading_styles)
        self._apply_to_document(heading_styles)
        self._convert_title_items(document, heading_styles)

        return document

    def _collect_headings(self, document: DoclingDocument) -> list:
        """Collect all SectionHeaderItem references from the document."""
        return [
            item
            for item, _level in document.iterate_items()
            if hasattr(item, "level") and hasattr(item, "prov")
        ]

    def _extract_styles(self, headings, pdf_path: Path) -> list[dict]:
        """Extract font size and bold for each heading using PyMuPDF."""
        doc = fitz.open(pdf_path)
        try:
            styles = []
            for h in headings:
                style = self._extract_single_heading_style(h, doc)
                if style is not None:
                    styles.append(style)
            return styles
        finally:
            doc.close()

    def _get_page_height(self, doc: fitz.Document, page_no: int) -> Optional[float]:
        """Get page height for coordinate conversion.

        Args:
            doc: Open PyMuPDF document.
            page_no: 1-based page number.

        Returns:
            Page height in points, or None if page is invalid.
        """
        if page_no < 1 or page_no > doc.page_count:
            return None
        return doc[page_no - 1].rect.height

    def _extract_single_heading_style(self, heading, doc: fitz.Document) -> Optional[dict]:
        """Extract font properties for a single heading from the PDF.

        Steps:
            1. Get heading bounding box from provenance
            2. Convert to top-left origin (PyMuPDF coordinate system)
            3. Find overlapping text spans via page.get_text("dict")
            4. Use the maximum font size as the heading's representative size

        Args:
            heading: A SectionHeaderItem from the document.
            doc: Open PyMuPDF document.

        Returns:
            Dict with heading, page_no, font_size, is_bold, is_italic
            or None if no text could be matched.
        """
        if not heading.prov:
            return None

        prov = heading.prov[0]
        page_no = prov.page_no
        bbox: BoundingBox = prov.bbox

        page_height = self._get_page_height(doc, page_no)
        if page_height is None:
            return None

        page = doc[page_no - 1]
        tl_bbox = bbox.to_top_left_origin(page_height)

        blocks = page.get_text("dict")["blocks"]

        max_size = 0.0
        is_bold = False
        is_italic = False

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if self._span_overlaps_bbox(span["bbox"], tl_bbox):
                        span_size = span["size"]
                        if span_size > max_size:
                            max_size = span_size
                            font_name = span.get("font", "")
                            flags = span.get("flags", 0)
                            is_bold = self._is_bold(font_name, flags)
                            is_italic = self._is_italic(font_name, flags)

        if max_size == 0.0:
            return None

        return {
            "heading": heading,
            "page_no": page_no,
            "font_size": max_size,
            "is_bold": is_bold,
            "is_italic": is_italic,
        }

    @staticmethod
    def _span_overlaps_bbox(span_bbox: tuple[float, float, float, float], heading_bbox: BoundingBox) -> bool:
        """Check if a PyMuPDF text span overlaps with the heading bbox.

        Uses bounding box intersection to accommodate minor misalignments
        between docling's layout model and PyMuPDF's text extraction.

        Args:
            span_bbox: (x0, y0, x1, y1) from PyMuPDF in top-left origin.
            heading_bbox: BoundingBox in top-left origin.

        Returns:
            True if the span and heading bboxes intersect.
        """
        sx0, sy0, sx1, sy1 = span_bbox
        overlap_x = max(0, min(sx1, heading_bbox.r) - max(sx0, heading_bbox.l))
        overlap_y = max(0, min(sy1, heading_bbox.b) - max(sy0, heading_bbox.t))
        return overlap_x > 0 and overlap_y > 0

    @staticmethod
    def _is_bold(font_name: str, flags: int) -> bool:
        """Determine if a font span is bold.

        Checks both the font name for keywords like "Bold", "Heavy", "Black",
        and the PyMuPDF flags bit 4.

        Args:
            font_name: Font name from the PDF (e.g., "FrutigerCE-Bold").
            flags: PyMuPDF span flags integer.

        Returns:
            True if the font appears to be bold.
        """
        bold_keywords = ["Bold", "Heavy", "Black", "Demi", "ExtraBold"]
        name_has_bold = any(kw in font_name for kw in bold_keywords)
        # PyMuPDF bit 4 indicates bold
        flag_has_bold = bool(flags & 16)
        return name_has_bold or flag_has_bold

    @staticmethod
    def _is_italic(font_name: str, flags: int) -> bool:
        """Determine if a font span is italic.

        Args:
            font_name: Font name from the PDF.
            flags: PyMuPDF span flags integer.

        Returns:
            True if the font appears to be italic.
        """
        italic_keywords = ["Italic", "Oblique", "It"]
        name_has_italic = any(kw in font_name for kw in italic_keywords)
        # PyMuPDF bit 1 indicates italic
        flag_has_italic = bool(flags & 2)
        return name_has_italic or flag_has_italic

    def _assign_levels(self, heading_styles: list[dict]) -> None:
        """Assign heading levels by font-size rank with bold/non-bold rules.

        Rules:
            rank 0 (largest), non-bold  ->  title (level=0)
            rank 3 (fourth),  bold      ->  H3   (level=3)
            rank 3 (fourth),  non-bold  ->  H4   (level=4)
            all others                  ->  level = min(rank + 1, n_tiers)
        """
        sizes = sorted(set(s["font_size"] for s in heading_styles), reverse=True)

        for hs in heading_styles:
            rank = sizes.index(hs["font_size"])
            if rank == 0 and not hs["is_bold"]:
                hs["level"] = 0
            elif rank == 3 and hs["is_bold"]:
                hs["level"] = 3
            elif rank == 3 and not hs["is_bold"]:
                hs["level"] = 4
            else:
                hs["level"] = min(rank + 1, self.n_tiers)

    @staticmethod
    def _apply_to_document(heading_styles: list[dict]) -> None:
        """Write level and formatting back onto document heading items."""
        for hs in heading_styles:
            heading = hs["heading"]
            if hs["level"] == 0:
                heading.level = 1
            else:
                heading.level = hs["level"]
            heading.formatting = Formatting(
                bold=hs["is_bold"],
                italic=hs["is_italic"],
            )

    @staticmethod
    def _convert_title_items(
        document: DoclingDocument,
        heading_styles: list[dict],
    ) -> None:
        """Replace level-0 SectionHeaderItems with TitleItems."""
        for hs in heading_styles:
            if hs.get("level") != 0:
                continue
            heading = hs["heading"]
            if not isinstance(heading, SectionHeaderItem):
                continue
            title_item = TitleItem(
                self_ref=heading.self_ref,
                text=heading.text,
                orig=heading.orig,
                prov=heading.prov,
                formatting=heading.formatting,
                hyperlink=heading.hyperlink,
            )
            document.replace_item(new_item=title_item, old_item=heading)
