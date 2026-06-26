# About this repo

Currently, the repo contains two files:

1. [`batch_converter.py`](batch_converter.py): this script allows you to convert multiple PDFs from an `inputs/` directory with the same configuration.
2. [`convert_to_ls.py`](convert_to_ls.py): this script converts Docling exported conversions into Label Studio format.

## Docling Setup

1. Run `pip install docling` 😄

> optional: run `pip install --upgrade label-studio-sdk` for integration with label studio **experimental: we have to verify the label studio version first**

### 1. Docling CLI

Run (processing is faster if you provide a Hugging Face token):

```powershell
$env:HF_TOKEN="your_token_here"
docling [--from 'pdf' --to 'output (json, html)'] source_file
```

[Full list of Docling CLI options](https://docling-project.github.io/docling/reference/cli/).

For our tests, we added the `--` options:

- `pdf-backend pypdfium2`: which makes the overall process a lot faster
- (optional) `no-ocr`: forces it to run without OCR

This is better for quick testing with a single document.

### 2. Code Approach

Docling's default config will fail with large PDFs. Follow these steps to make sure it runs smoothly.

We can approach this by:

1. Modifying the pipeline's settings (current approach)
**or**
2. Chunking the data before processing it

I would recommend adding a HuggingFace token before starting. This ensures your models get downloaded fast. The below is for powershell.

```powershell
    $env:HF_TOKEN="your_token_here"
```

Then run `batch_converter.py`! Below is some code explanation if you'd like to understand how it works.

```python
# Import Docling's required libraries
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# Import a custom model - we can change this later on
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
```

To configure the pipeline directly, edit the following

```python
# Edit the pipeline here
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False # We run out of memory when using OCR
pipeline_options.do_table_structure = True

# --- Not exactly sure about this one right now
pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=False)

# Instantiate DocumentConverter, and pass pipeline options + PyPdfium2
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption (
            pipeline_options=pipeline_options, 
            backend=PyPdfiumDocumentBackend
        )
    }
)
```


---

### Docling Architecture

Simply put, the 'document converter' can employ a specific pipeline for use based on specific formats.

It returns a 'Docling document', which can then be used to call export methods in markdown, dictionary, or document tokens. Alternatively, it can be serialised or chunked.

---

### Heading enrichment

`heading_enricher.py` enriches `SectionHeaderItem`s in a converted `DoclingDocument` with heading levels (Title / H1–H5). It combines three signals, in priority order:

1. **Table-of-contents analysis** (PyMuPDF only) — reads the embedded PDF outline, or scans the first pages for a `Contents` page (with PyMuPDF OCR fallback for scanned PDFs).
2. **DeepSeek style-level detection** — sends the candidate headings (text + font size + bold/italic) to the DeepSeek chat completions API (`deepseek-V4-Flash` by default) and asks for a JSON level assignment. Responses are cached on disk in `./.cache/deepseek/` keyed by a hash of the input, so repeated runs of the same PDF are free.
3. **PyMuPDF font-size tier clustering** — the original heuristic that groups unique font sizes into percentile tiers and maps them to levels using bold/non-bold rules. Used as the final fallback.

The resolved level, the source that won (`toc` / `deepseek` / `tier`), and the candidate levels from all three signals are written to `./headers/{pdf_stem}_headers.json`.

#### Setup

Install the additional dependency used for TOC matching:

```powershell
pip install rapidfuzz
```

Set the DeepSeek API key — either as an environment variable, or in the existing `.env` file (renamed to the standard name):

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
```

```ini
# .env
DEEPSEEK_API_KEY = sk-...
```

Run with the new options:

```powershell
python batch_converter.py                       # uses DeepSeek by default
python batch_converter.py --no-deepseek         # tier-only fallback
python batch_converter.py --deepseek-model deepseek-chat
python batch_converter.py -p .\inputs\report.pdf
```

#### Standalone detection

`heading_enricher.py` can also run on its own — no `DoclingDocument` is
required. It opens the PDF, detects candidates, applies the same
TOC / DeepSeek / tier pipeline, and returns a typed `HeadingDetectionResult`:

```python
from heading_enricher import HeadingEnricher

enricher = HeadingEnricher()
result = enricher.detect("report.pdf")

for h in result.headings:
    print(f"p{h.page_no:>3}  L{h.final_level}  ({h.source:<8})  {h.text}")

result.save_json("./headers/report_headers.json")
```

A small CLI is included:

```powershell
python heading_enricher.py report.pdf
python heading_enricher.py report.pdf --out .\headers\report.json --no-deepseek
python heading_enricher.py report.pdf --max-scan-pages 50 --heading-size-ratio 1.2
```

Re-load a previously saved result:

```python
from heading_enricher import HeadingEnricher
result = HeadingEnricher.from_dict(json.load(open("report_headers.json")))
```
