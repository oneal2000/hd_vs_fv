import json
import math
import argparse
from collections import defaultdict
from sklearn.metrics import roc_auc_score

def safe_float(val):
    """Convert to float, handling list wrappers and invalid values."""
    try:
        if val is None:
            return None
        if isinstance(val, list):
            val = val[0] if val else None
        val = float(val)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None

def compute_auroc_per_field(data, field_key):
    """Compute AUROC per score key under a field like 'detection_scores'."""
    scores_by_key = defaultdict(list)
    labels_by_key = defaultdict(list)

    for item in data:
        verdict = item.get("judge_verdict", "").lower()
        if verdict not in ["accurate", "inaccurate"]:
            continue
        label = 1 if verdict == "inaccurate" else 0

        field_data = item.get(field_key, {})
        if not isinstance(field_data, dict):
            continue

        for score_key, score_val in field_data.items():
            val = safe_float(score_val)
            if val is not None:
                scores_by_key[score_key].append(val)
                labels_by_key[score_key].append(label)

    auroc_by_key = {}
    for score_key in scores_by_key:
        y_true = labels_by_key[score_key]
        y_score = scores_by_key[score_key]
        try:
            if len(set(y_true)) < 2:
                auroc_by_key[score_key] = None  # AUROC undefined
            else:
                auroc_by_key[score_key] = roc_auc_score(y_true, y_score)
        except Exception:
            auroc_by_key[score_key] = None

    return auroc_by_key

def main(jsonl_path):
    valid_data = []
    total_lines = 0
    skipped_lines = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                item = json.loads(line)
                if item.get("judge_verdict", "").lower() in ["accurate", "inaccurate"]:
                    valid_data.append(item)
                else:
                    skipped_lines += 1
            except Exception:
                skipped_lines += 1

    print(f"\nTotal items: {total_lines}, valid: {len(valid_data)}, skipped: {skipped_lines}")

    detection_aurocs = compute_auroc_per_field(valid_data, "detection_scores")
    fv_aurocs = compute_auroc_per_field(valid_data, "fv_scores")

    print("\n=== Detection AUROCs ===")
    for k, v in sorted(detection_aurocs.items()):
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: AUROC undefined")

    print("\n=== Fact Verification (FV) AUROCs ===")
    for k, v in sorted(fv_aurocs.items()):
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: AUROC undefined")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute AUROC for detection and FV scores.")
    parser.add_argument("jsonl_path", type=str, help="Path to the JSONL file containing prediction data.")
    args = parser.parse_args()
    main(args.jsonl_path)