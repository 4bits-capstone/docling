import os
import json
import requests
from pathlib import Path

LS_URL = "http://localhost:8080"
API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

PROJECT_ID = 2
TASK_ID = 1
DOCLING_JSON = "outputs/Funeral_and_Burial_Instructions_Report_for_web.json"
MODEL_VERSION = "docling_json_upload"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


LABEL_MAP = {
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


def convert_bbox(bbox, page_width, page_height):
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

    w = x2 - x1
    h = y2 - y1

    if page_width <= 0 or page_height <= 0 or w <= 0 or h <= 0:
        return None

    return {
        "x": x1 / page_width * 100,
        "y": y1 / page_height * 100,
        "width": w / page_width * 100,
        "height": h / page_height * 100,
        "rotation": 0,
    }


def get_page(pages, page_no):
    if isinstance(pages, dict):
        return pages.get(str(page_no)) or pages.get(page_no)

    if isinstance(pages, list):
        idx = int(page_no) - 1
        if 0 <= idx < len(pages):
            return pages[idx]

    return None


def docling_to_ls_regions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    document = data.get("document", data)
    pages = document.get("pages", {})

    regions = []
    all_items = []

    for name in ["texts", "pictures", "tables", "form_items"]:
        all_items.extend(document.get(name, []))

    for item_idx, item in enumerate(all_items):
        prov = item.get("prov", [])
        if not prov:
            continue

        src = prov[0]
        bbox = src.get("bbox")
        page_no = src.get("page_no")

        if bbox is None or page_no is None:
            continue

        page = get_page(pages, page_no)
        if page is None:
            continue

        page_width = (
            page.get("page_width")
            or page.get("width")
            or page.get("size", {}).get("width")
            or 0
        )
        page_height = (
            page.get("page_height")
            or page.get("height")
            or page.get("size", {}).get("height")
            or 0
        )

        ls_bbox = convert_bbox(bbox, page_width, page_height)
        if ls_bbox is None:
            continue

        raw_label = str(item.get("label", "unspecified"))
        label = LABEL_MAP.get(raw_label, "unspecified")

        page_index = int(page_no) - 1

        regions.append({
            "id": f"region_{page_index}_{item_idx}",
            "from_name": "layout_label",
            "to_name": "pdf",
            "type": "rectanglelabels",
            "value": {
                **ls_bbox,
                "rectanglelabels": [label],
            },
            "item_index": page_index,
            "page_index": page_index,
        })

    return regions


if not API_KEY:
    raise RuntimeError(
        "LABEL_STUDIO_API_KEY is not set. "
        "In PowerShell run: $env:LABEL_STUDIO_API_KEY='your_token_here'"
    )

regions = docling_to_ls_regions(DOCLING_JSON)

payload = {
    "task": TASK_ID,
    "project": PROJECT_ID,
    "result": regions,
    "model_version": MODEL_VERSION,
}

resp = requests.post(
    f"{LS_URL}/api/predictions/",
    headers=HEADERS,
    json=payload,
)

if resp.ok:
    print(f"Uploaded {len(regions)} prediction regions to task {TASK_ID}")
else:
    print(f"ERROR {resp.status_code}")
    print(resp.text)