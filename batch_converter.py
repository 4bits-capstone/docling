"""
batch_converter.py:
    Create a folder called `inputs`, filled with your source PDFs.
    This will convert it into JSON, at the `outputs/` folder. 

"""

import argparse
import json
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from heading_enricher import HeadingEnricher

def _iter_input_files(INPUT_DIR: Path) -> list[Path]:
    return sorted(path for path in INPUT_DIR.rglob("*") if path.is_file())

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert PDFs to Docling JSON.")
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        help="Single PDF filepath to process (skips inputs/ directory).",
    )
    args = parser.parse_args()

    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        input_files = [Path(args.pdf)]
    else:
        INPUT_DIR = Path("./inputs")
        input_files = _iter_input_files(INPUT_DIR)
        if not input_files:
            print(f"No files found in {INPUT_DIR}.")
            return

    start_time = time.perf_counter()
    # Edit the pipeline here
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False                  # We run out of memory when using OCR
    pipeline_options.do_table_structure = True
    pipeline_options.do_code_enrichment = False
    
    # Fast table processing
    pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=False)

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption (
                pipeline_options=pipeline_options, 
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    for pdf in input_files:
        print(f"Processing {pdf.stem}")
        source = pdf
        result = doc_converter.convert(source)

        # Enrich section headers with PyMuPDF-based heading levels (H1/H2/H3)
        HeadingEnricher(n_tiers=5).enrich(result.document, str(pdf))

        # Export the result to docling JSON
        with open(OUTPUT_DIR / f"{pdf.stem}.json", "w", encoding="utf-8") as f:
            json.dump(result.document.export_to_dict(), f, indent=2)

    elapsed_seconds = time.perf_counter() - start_time


    print(f"Converted {len(input_files)} file(s) in {elapsed_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()