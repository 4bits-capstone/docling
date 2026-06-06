# Install dependencies
# %pip install docling
# %pip install --upgrade label-studio-sdk

from __future__ import annotations
from label_studio_sdk.client import LabelStudio

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

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
        "rotation": 0
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
            page_idx = task.get("id", 0)
            for ann in task.get("annotations", []):
                for result in ann.get("result", []):
                    value = dict(result["value"])
                    value.setdefault("rectanglelabels", ["unspecified"])
                    regions.append({
                        "id": result.get("id", f"region_{len(regions)}"),
                        "from_name": result.get("from_name", "layout_label"),
                        "to_name": result.get("to_name", "pdf"),
                        "type": result.get("type", "rectanglelabels"),
                        "value": value,
                        "page_index": result.get("item_index", 0),
                        "origin": result.get("origin", "manual"),
                    })
        return regions

    # Docling native format — dict with "pages"
        # Docling native format
    if isinstance(data, dict):
        regions = []

        document = data.get("document", data)
        pages = document.get("pages", {})

        def get_page(page_no):
            if isinstance(pages, dict):
                return pages.get(str(page_no)) or pages.get(page_no)
            if isinstance(pages, list):
                idx = int(page_no) - 1
                if 0 <= idx < len(pages):
                    return pages[idx]
            return None

        all_items = []
        for collection_name in ["texts", "pictures", "tables", "form_items"]:
            for item in document.get(collection_name, []):
                all_items.append(item)

        for item_idx, item in enumerate(all_items):
            prov = item.get("prov", [])
            if not prov:
                continue

            src = prov[0]
            bbox = src.get("bbox")
            page_no = src.get("page_no")

            if bbox is None or page_no is None:
                continue

            page = get_page(page_no)
            if page is None:
                print(f"Skipping item {item_idx}: missing page {page_no}")
                continue

            pw = (
                page.get("page_width")
                or page.get("width")
                or page.get("size", {}).get("width")
                or 0
            )
            ph = (
                page.get("page_height")
                or page.get("height")
                or page.get("size", {}).get("height")
                or 0
            )

            ls_bbox = _convert_single_bbox(bbox, pw, ph)
            if ls_bbox is None:
                continue

            label = map_label(item)
            page_index = int(page_no) - 1

            regions.append({
                "id": f"region_{page_index}_{item_idx}",
                "from_name": "layout_label",
                "to_name": "pdf",
                "type": "rectanglelabels",
                "value": {**ls_bbox, "rectanglelabels": [label]},
                "page_index": page_index,
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

    label_map = {
        "caption": "caption",
        "checkbox_unselected": "form",
        "checkbox_selected": "form",
        "document_index": "H1",
        "footnote": "footnote",
        "formula": "formula",
        "list": "list",
        "list_item": "list",
        "page_footer": "text",
        "page_header": "text",
        "picture": "picture",
        "section_header": "section_header",
        "table": "table",
        "title": "title",
        "text": "text",
        "unspecified": "unspecified",
    }

    return label_map.get(raw, "unspecified")

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


def print_ls_output():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(PREDICTIONS_DIR)
    for src in sorted(src_dir.iterdir()):
        if src.suffix != ".json":
            continue
        regions = convert_bbox_to_ls(src)
        dst = out_dir / src.name
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(regions, f, indent=2)
        print(f"wrote {len(regions)} regions to {dst}")


def main(project_id: int, model_version: str):
    # Initialise a label studio connection
    ls = LabelStudio(base_url=LABEL_STUDIO, api_key=LS_API_KEY)

    # Get a label studio project ID
    tasks = {t.id: t for t in ls.tasks.list(project=project_id)}

    # Retrieve docling outputs from the folder
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
            project=project_id,
            task=task_id,
            result=regions,
            model_version=model_version
        )

    print(f"uploaded {len(regions)} for predictions for task {task_id} ({output.name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass exported Docling predictions into a single Label Studio Project")
    parser.add_argument("--project_id", type=int, help="Label Studio project ID")
    parser.add_argument("--model_version", type=str, help="Provide details of your PDF backend and OCR changes or Table Formatter (if any)")
    parser.add_argument("--local", action="store_true", help="Write to label-studio-output/ instead of uploading")
    args = parser.parse_args()

    if args.local or not args.project_id:
        print_ls_output()
    else:
        main(args.project_id, args.model_version)