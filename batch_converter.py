"""
batch_converter.py:
    Create a folder called `inputs`, filled with your source PDFs.
    This will convert it into JSON, at the `outputs/` folder. 

"""

import json
import logging
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# Import a custom model
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
# from docling.datamodel.pipeline_options import TesseractOcrOptions, TesseractCliOcrOptions, OcrMacOptions

_log = logging.getLogger(__name__)

def _iter_input_files(INPUT_DIR: Path) -> list[Path]:
    return sorted(path for path in INPUT_DIR.rglob("*") if path.is_file())

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    INPUT_DIR = Path("./inputs")
    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = _iter_input_files(INPUT_DIR)
    if not input_files:
        _log.warning("No files found in %s", INPUT_DIR)
        return

    # Edit the pipeline here
    pipeline_options = PdfPipelineOptions()

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA,
    )

    pipeline_options.do_ocr = False # We run out of memory when using OCR
    pipeline_options.do_table_structure = True
    

    # --- Not exactly sure about this one right now
    pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=False)

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption (
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )
        }
    )

    start_time = time.perf_counter()
    for input_file in input_files:
        _log.info("Converting %s", input_file)
        result = doc_converter.convert(input_file)

        # Prepare loggign information (source file, conversion method, and document in dict format)
        output_file = OUTPUT_DIR / f"{input_file.stem}.json"
        payload = {
            "source_file": str(input_file),
            "document": result.document.export_to_dict(),
        }

        with output_file.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)

    elapsed_seconds = time.perf_counter() - start_time
    stats_file = OUTPUT_DIR / "conversion_stats.json"
    stats = {
        "INPUT_DIRectory": str(INPUT_DIR),
        "OUTPUT_DIRectory": str(OUTPUT_DIR),
        "files_converted": len(input_files),
        "elapsed_seconds": elapsed_seconds,
    }

    with stats_file.open("w", encoding="utf-8") as file_handle:
        json.dump(stats, file_handle, ensure_ascii=False, indent=2)

    print(f"Converted {len(input_files)} file(s) in {elapsed_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()