# Install dependencies
# %pip install docling
# %pip install --upgrade label-studio-sdk

from __future__ import annotations
from label_studio_sdk.client import LabelStudio

import argparse
import json
import time
from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

LABEL_STUDIO = "http://localhost:8080"
LS_API_KEY = "a81a470adcac997a1fc177fe9d09aec21a84e48f"
IMAGE_SERVER = "http://localhost:9090"
PREDICTIONS_DIR = "outputs"
OUTPUT_DIR = "label-studio-output"


# Helper Functions

def _convert_single_bbox(bbox, page_width, page_height):
    l = float(bbox["l"])
    t = float(bbox["t"])
    r = float(bbox["r"])
    b = float(bbox["b"])
    origin = str(bbox.get("coord_origin", "TOPLEFT"))

    x1 = min(l, r)
    x2 = max(l, r)

    if origin.endswith("BOTTOMLEFT"):
        y1 = page_height - max(t, b)
        y2 = page_height - min(t, b)
    else:
        y1 = min(t, b)
        y2 = max(t, b)

    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    if page_width <= 0 or page_height <= 0 or w <= 0 or h <= 0:
        return None

    return {
        "x": (x1 / page_width) * 100.0,
        "y": (y1 / page_height) * 100.0,
        "width": (w / page_width) * 100.0,
        "height": (h / page_height) * 100.0,
        "rotation": 0,
    }


def convert_bbox_to_ls(source_file: str | Path) -> list[dict]:
    """
    Load a Docling JSON annotation file and convert all bboxes to Label Studio format.

    Handles two formats:
      - Docling native: a dict with a "pages" key
      - Label Studio export: a list of tasks with annotations

    Args:
        source_file: path to a .json file

    Returns: a list of Label Studio region dicts with rectanglelabels, ready to upload.
    """
    with open(source_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Label Studio export format — list of tasks with annotations
    if isinstance(data, list):
        regions = []
        for task in data:
            for ann in task.get("annotations", []):
                for result in ann.get("result", []):
                    value = dict(result["value"])
                    value.setdefault("rectanglelabels", ["unspecified"])
                    regions.append({
                        "id": result.get("id", f"region_{len(regions)}"),
                        "from_name": result.get("from_name", "layout_label"),
                        "to_name": result.get("to_name", "pdf"),
                        "original_width": result.get("original_width", 0),
                        "original_height": result.get("original_height", 0),
                        "type": result.get("type", "rectanglelabels"),
                        "value": value,
                        "item_index": result.get("item_index", 0),
                        "origin": result.get("origin", "manual"),
                    })
        return regions

    # Docling native format — dict with "pages" (keyed by page number str)
    # Items live in top-level lists: texts, pictures, tables, form_items, key_value_items
    if isinstance(data, dict):
        page_map = {}
        for page_key, page_val in data.get("pages", {}).items():
            size = page_val.get("size", {})
            page_map[page_val.get("page_no", int(page_key))] = {
                "width": size.get("width", 0),
                "height": size.get("height", 0),
            }

        regions = []
        item_lists = ["texts", "pictures", "tables", "form_items", "key_value_items"]
        for list_key in item_lists:
            for item in data.get(list_key, []):
                label = map_label(item)
                for prov in item.get("prov", []):
                    page_no = prov.get("page_no", 1)
                    page_info = page_map.get(page_no, {})
                    pw = page_info.get("width", 0)
                    ph = page_info.get("height", 0)
                    bbox = prov.get("bbox")
                    if bbox is None:
                        continue
                    ls_bbox = _convert_single_bbox(bbox, pw, ph)
                    if ls_bbox is None:
                        continue
                    # Label Studio multi-page indexing is 0-based.
                    item_index = max(int(page_no) - 1, 0)
                    regions.append({
                        "id": f"region_{list_key}_{len(regions)}",
                        "from_name": "layout_label",
                        "to_name": "pdf",
                        "original_width": pw,
                        "original_height": ph,
                        "type": "rectanglelabels",
                        "value": {**ls_bbox, "rectanglelabels": [label]},
                        "item_index": item_index,
                    })
        return regions

    raise ValueError(f"Unsupported JSON structure in {source_file}")


def map_label(item):
    """
    Docling's processing background may use a different labeling structure than what we expect. Therefore, we standardise each item's given label and convert it to one our system expects.

    Args:
        item: A result object from Docling's predictions

    Returns:
        The matching label for it, if one hasn't been set it defaults to unspecified
    """
    raw = str(item.get("label", "unspecified"))

    if raw == "section_header":
        meta = item.get("meta", {}) or {}
        level = int(meta.get("hf__heading_level", item.get("level", 1)))
        if level < 1:
            level = 1
        if level > 5:
            level = 5
        return f"H{level}"

    label_map = {
        "caption": "caption",
        "checkbox_unselected": "form",
        "checkbox_selected": "form",
        "document_index": "list",
        "footnote": "footnote",
        "formula": "formula",
        "list": "list",
        "list_item": "list",
        "page_footer": "text",
        "page_header": "text",
        "picture": "picture",
        "table": "table",
        "title": "title",
        "text": "text",
        "unspecified": "unspecified",
    }

    return label_map.get(raw, "unspecified")


def do_ocr(source_file: str | Path) -> list[dict]:
    """
    Process a PDF through the Docling pipeline and return predictions in Label Studio format.

    Args:
        source_file: path to a .pdf file

    Returns:
        A list of prediction dicts with item_index, label, bbox_value, original_width/height, etc.
    """
    start_time = time.time()
    predictions = []

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=False)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    source_path = Path(source_file)
    print(f"processing {source_path}")

    result = converter.convert(source_path)

    doc = result.document.export_to_dict()
    pages = doc.get("pages", {})

    for collection_name in ("texts", "pictures", "tables", "form_items", "key_value_items"):
        for item in doc.get(collection_name, []):
            mapped_label = map_label(item)

            for prov in item.get("prov", []):
                page_no = prov.get("page_no")
                bbox = prov.get("bbox")
                if page_no is None or bbox is None:
                    continue

                page_meta = pages.get(str(page_no)) or pages.get(page_no)
                if not page_meta:
                    continue

                size = page_meta.get("size", {})
                page_width = float(size.get("width", 0))
                page_height = float(size.get("height", 0))

                bbox_ls = _convert_single_bbox(bbox, page_width, page_height)
                if not bbox_ls:
                    continue

                # Label Studio multi-page indexing is 0-based.
                item_index = max(int(page_no) - 1, 0)
                predictions.append({
                    "item_index": item_index,
                    "page_no": int(page_no),
                    "label": mapped_label,
                    "bbox_value": bbox_ls,
                    "original_width": page_width,
                    "original_height": page_height,
                })

    end_time = time.time() - start_time
    print(f"Done. Processed in {end_time:.2f}s")
    return predictions


def _predictions_to_regions(predictions: list[dict]) -> list[dict]:
    """Convert do_ocr() prediction dicts into Label Studio result regions."""
    regions = []
    for i, p in enumerate(predictions):
        value = dict(p["bbox_value"])
        value["rectanglelabels"] = [p["label"]]
        regions.append({
            "id": f"region{i}",
            "from_name": "layout_label",
            "to_name": "pdf",
            "original_width": p["original_width"],
            "original_height": p["original_height"],
            "type": "rectanglelabels",
            "value": value,
            "item_index": p["item_index"],
        })
    return regions


# Labeling Config for OCR using Multi-page document annotation
labeling_config = """
<View style="display:flex;align-items:start;gap:8px;flex-direction:row">
  <Image name="pdf" valueList="$pages" zoom="true" zoomControl="true" rotateControl="true"/>
  <RectangleLabels name="layout_label" toName="pdf" showInline="false">
  
    <Label value="H1" background="#2ca02c"/>
    <Label value="H2" background="#98df8a"/>
    <Label value="H3" background="#ff7f0e"/>
    <Label value="H4" background="#FFA39E"/>
    <Label value="H5" background="#fccfcc"/>

    <Label value="caption" background="#FFC069"/>
    <Label value="footnote" background="#1f77b4"/>
    <Label value="form" background="#bcbd22"/>
    <Label value="formula" background="#f9c1be"/>
    <Label value="list" background="#c49c94"/>
    <Label value="picture" background="#ff9896"/>
    <Label value="section_header" background="#393b79"/>
    <Label value="table" background="#D94545"/>
    <Label value="title" background="#940505"/>
    <Label value="text" background="#cccccc"/>
    <Label value="unspecified" background="#000000"/>

    </RectangleLabels>
  </View>

"""


def print_ls_output(pdf_dir=None):
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pdf_dir:
        src_dir = Path(pdf_dir)
        for src in sorted(src_dir.iterdir()):
            if src.suffix.lower() not in (".pdf",):
                continue
            predictions = do_ocr(src)
            regions = _predictions_to_regions(predictions)
            dst = out_dir / f"{src.stem}.json"
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(regions, f, indent=2)
            print(f"wrote {len(regions)} regions to {dst}")
    else:
        src_dir = Path(PREDICTIONS_DIR)
        for src in sorted(src_dir.iterdir()):
            if src.suffix != ".json":
                continue
            regions = convert_bbox_to_ls(src)
            dst = out_dir / src.name
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(regions, f, indent=2)
            print(f"wrote {len(regions)} regions to {dst}")


def main(project_id: int, model_version: str, pdf_dir=None):
    ls = LabelStudio(base_url=LABEL_STUDIO, api_key=LS_API_KEY)

    ls.projects.update(id=project_id, label_config=labeling_config)

    tasks = {t.id: t for t in ls.tasks.list(project=project_id)}

    if pdf_dir:
        src_dir = Path(pdf_dir)
        for src in sorted(src_dir.iterdir()):
            if src.suffix.lower() not in (".pdf",):
                continue
            predictions = do_ocr(src)
            regions = _predictions_to_regions(predictions)

            stem = src.stem
            task_id = next((tid for tid, t in tasks.items() if stem in str(t.data)), None)
            if task_id is None:
                print(f"skipping {src.name}: no matching task found")
                continue

            ls.predictions.create(
                task=task_id,
                result=regions,
                model_version=model_version,
            )
            print(f"uploaded {len(regions)} predictions for task {task_id} ({src.name})")
    else:
        docling_outputs = Path(PREDICTIONS_DIR)
        for output in sorted(docling_outputs.iterdir()):
            if output.suffix != ".json":
                continue
            regions = convert_bbox_to_ls(output)

            stem = output.stem
            task_id = next((tid for tid, t in tasks.items() if stem in str(t.data)), None)
            if task_id is None:
                print(f"skipping {output.name}: no matching task found")
                continue

            ls.predictions.create(
                task=task_id,
                result=regions,
                model_version=model_version,
            )
            print(f"uploaded {len(regions)} predictions for task {task_id} ({output.name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass exported Docling predictions into a single Label Studio Project")
    parser.add_argument("--project_id", type=int, help="Label Studio project ID")
    parser.add_argument("--model_version", type=str, help="Provide details of your PDF backend and OCR changes or Table Formatter (if any)")
    parser.add_argument("--local", action="store_true", help="Write to label-studio-output/ instead of uploading")
    parser.add_argument("--pdf-dir", type=str, help="Directory of PDF files to process through Docling pipeline (instead of reading pre-exported JSONs from outputs/)")
    args = parser.parse_args()

    if args.local or not args.project_id:
        print_ls_output(pdf_dir=args.pdf_dir)
    else:
        main(args.project_id, args.model_version, pdf_dir=args.pdf_dir)
