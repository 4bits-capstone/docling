"""Generate confidence scores comparing label-studio-output vs ground-truths."""

import json
from pathlib import Path
from collections import defaultdict

PRED_DIR = Path("label-studio-output")
GT_DIR = Path("ground-truths")
IOU_THRESHOLD = 0.5
EXCLUDE_LABELS = {"unspecified"}

FILENAME_MAP = {
    "Surveillance_final_report.json": "Surveillance_in_Public_Places_2010.json",
}


def load_predictions(path):
    """Load flat array of prediction regions."""
    data = json.loads(path.read_text(encoding="utf-8"))
    regions = []
    for item in data:
        if "value" not in item:
            continue
        label = item["value"]["rectanglelabels"][0]
        if label in EXCLUDE_LABELS:
            continue
        regions.append({
            "page": item["item_index"],
            "x": item["value"]["x"],
            "y": item["value"]["y"],
            "w": item["value"]["width"],
            "h": item["value"]["height"],
            "label": label,
        })
    return regions


def load_ground_truths(path):
    """Load ground truth regions from Label Studio export format."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = raw if isinstance(raw, list) else [raw]
    regions = []
    for task in tasks:
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if "value" not in result:
                    continue
                label = result["value"]["rectanglelabels"][0]
                if label in EXCLUDE_LABELS:
                    continue
                regions.append({
                    "page": result["item_index"],
                    "x": result["value"]["x"],
                    "y": result["value"]["y"],
                    "w": result["value"]["width"],
                    "h": result["value"]["height"],
                    "label": label,
                })
    return regions


def compute_iou(a, b):
    """IoU of two bounding boxes (x, y, w, h in percentage space)."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def match_regions(preds, gts):
    """Greedy match predictions to ground truths by IoU per page."""
    pages = sorted(set(r["page"] for r in preds) | set(r["page"] for r in gts))

    tp = []
    fp = []
    fn = []

    for page in pages:
        page_preds = [r for r in preds if r["page"] == page]
        page_gts = [r for r in gts if r["page"] == page]

        matches = []
        for pi, p in enumerate(page_preds):
            for gi, g in enumerate(page_gts):
                iou = compute_iou(p, g)
                if iou >= IOU_THRESHOLD:
                    matches.append((iou, pi, gi))

        matches.sort(key=lambda x: -x[0])
        used_preds = set()
        used_gts = set()

        for iou, pi, gi in matches:
            if pi not in used_preds and gi not in used_gts:
                tp.append({
                    "pred": page_preds[pi],
                    "gt": page_gts[gi],
                    "iou": iou,
                    "label_match": page_preds[pi]["label"] == page_gts[gi]["label"],
                })
                used_preds.add(pi)
                used_gts.add(gi)

        for pi, p in enumerate(page_preds):
            if pi not in used_preds:
                fp.append(p)

        for gi, g in enumerate(page_gts):
            if gi not in used_gts:
                fn.append(g)

    return tp, fp, fn


def compute_detection_metrics(tp, fp, fn):
    """Compute overall detection metrics."""
    n_tp = len(tp)
    n_fp = len(fp)
    n_fn = len(fn)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    n_label_match = sum(1 for m in tp if m["label_match"])
    label_acc = n_label_match / n_tp if n_tp > 0 else 0.0
    avg_iou = sum(m["iou"] for m in tp) / n_tp if n_tp > 0 else 0.0

    return {
        "n_pred": n_tp + n_fp,
        "n_gt": n_tp + n_fn,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "n_label_match": n_label_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_accuracy": label_acc,
        "avg_iou": avg_iou,
    }


def compute_per_class_metrics(tp, fp, fn):
    """Compute per-class precision, recall, F1 from match results."""
    all_labels = set()
    for m in tp:
        all_labels.add(m["pred"]["label"])
        all_labels.add(m["gt"]["label"])
    for r in fp:
        all_labels.add(r["label"])
    for r in fn:
        all_labels.add(r["label"])

    per_class = {}
    for label in sorted(all_labels):
        cls_tp = sum(1 for m in tp if m["label_match"] and m["gt"]["label"] == label)

        cls_fp = sum(1 for r in fp if r["label"] == label)
        cls_fp += sum(1 for m in tp if m["pred"]["label"] == label and not m["label_match"])

        cls_fn = sum(1 for r in fn if r["label"] == label)
        cls_fn += sum(1 for m in tp if m["gt"]["label"] == label and not m["label_match"])

        p = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
        r = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_class[label] = {
            "tp": cls_tp,
            "fp": cls_fp,
            "fn": cls_fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }

    return per_class


def print_doc_report(doc_name, metrics, per_class):
    """Print per-document metrics."""
    sep = "-" * 72
    print(f"\n  Document: {doc_name}")
    print(f"    Regions: {metrics['n_pred']} pred, {metrics['n_gt']} gt")
    print(f"    Matched: {metrics['n_tp']}  (IoU >= {IOU_THRESHOLD})")
    print(f"    Precision: {metrics['precision']:.4f}   Recall: {metrics['recall']:.4f}   "
          f"F1: {metrics['f1']:.4f}")
    print(f"    Label accuracy (matched): {metrics['label_accuracy']:.4f}")
    print(f"    Avg IoU (matched): {metrics['avg_iou']:.4f}")

    if per_class:
        print()
        header = f"    {'Label':<22} {'P':>8} {'R':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}"
        print(header)
        print(f"    {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5}")
        for label, pc in per_class.items():
            print(f"    {label:<22} {pc['precision']:>8.4f} {pc['recall']:>8.4f} "
                  f"{pc['f1']:>8.4f} {pc['tp']:>5} {pc['fp']:>5} {pc['fn']:>5}")
    print(sep)


def main():
    sep = "=" * 72
    print(sep)
    print("  Confidence Score Report")
    print(sep)

    pred_dir = Path(PRED_DIR)
    gt_dir = Path(GT_DIR)

    pred_files = {p.name: p for p in sorted(pred_dir.glob("*.json"))
                  if p.name != "conversion_stats.json"}
    gt_files = {p.name: p for p in sorted(gt_dir.glob("*.json"))}

    doc_results = []

    for pred_name, pred_path in sorted(pred_files.items()):
        gt_name = FILENAME_MAP.get(pred_name, pred_name)
        if gt_name not in gt_files:
            print(f"\n  WARNING: No ground truth found for {pred_name}")
            continue

        gt_path = gt_files[gt_name]
        preds = load_predictions(pred_path)
        gts = load_ground_truths(gt_path)

        tp, fp, fn = match_regions(preds, gts)
        metrics = compute_detection_metrics(tp, fp, fn)
        per_class = compute_per_class_metrics(tp, fp, fn)

        doc_results.append({
            "doc_name": pred_name,
            "metrics": metrics,
            "per_class": per_class,
        })

        print_doc_report(pred_name, metrics, per_class)

    # --- Overall summary ---
    if not doc_results:
        print("\n  No documents to evaluate.")
        return

    total_tp = sum(r["metrics"]["n_tp"] for r in doc_results)
    total_fp = sum(r["metrics"]["n_fp"] for r in doc_results)
    total_fn = sum(r["metrics"]["n_fn"] for r in doc_results)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    n = len(doc_results)
    macro_p = sum(r["metrics"]["precision"] for r in doc_results) / n
    macro_r = sum(r["metrics"]["recall"] for r in doc_results) / n
    macro_f1 = sum(r["metrics"]["f1"] for r in doc_results) / n

    print(f"\n  {'Overall Summary':^72}")
    print(f"  {'-'*72}")
    print(f"    Documents evaluated: {n}")
    print(f"    Total regions: {total_tp + total_fp} pred, {total_tp + total_fn} gt")
    print(f"    Total matched: {total_tp}")
    print()
    print(f"    {'Metric':<22} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print(f"    {'-'*22} {'-'*12} {'-'*12} {'-'*12}")
    print(f"    {'Micro avg':<22} {micro_p:>12.4f} {micro_r:>12.4f} {micro_f1:>12.4f}")
    print(f"    {'Macro avg':<22} {macro_p:>12.4f} {macro_r:>12.4f} {macro_f1:>12.4f}")
    print()

    # Per-class summary across all docs
    all_per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in doc_results:
        for label, pc in r["per_class"].items():
            all_per_class[label]["tp"] += pc["tp"]
            all_per_class[label]["fp"] += pc["fp"]
            all_per_class[label]["fn"] += pc["fn"]

    print(f"  {'Per-Class Summary (aggregated)':^72}")
    print(f"  {'-'*72}")
    header = f"    {'Label':<22} {'P':>8} {'R':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}"
    print(header)
    print(f"    {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5}")
    for label in sorted(all_per_class.keys()):
        v = all_per_class[label]
        p = v["tp"] / (v["tp"] + v["fp"]) if (v["tp"] + v["fp"]) > 0 else 0.0
        r = v["tp"] / (v["tp"] + v["fn"]) if (v["tp"] + v["fn"]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"    {label:<22} {p:>8.4f} {r:>8.4f} {f1:>8.4f} "
              f"{v['tp']:>5} {v['fp']:>5} {v['fn']:>5}")
    print(sep)


if __name__ == "__main__":
    main()
