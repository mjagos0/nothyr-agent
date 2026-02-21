"""
COCO to YOLOv8 Annotation Converter

Usage:
    Place this script in the same directory as _annotations.coco.json and your images.
    Run: python coco_to_yolo.py

Output:
    - labels/  folder with one .txt per image (YOLOv8 format)
    - data.yaml ready for YOLOv8 training
"""

import json
import os
from pathlib import Path

# --- Configuration ---
COCO_JSON = "_annotations.coco.json"
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
LABELS_DIR = SCRIPT_DIR / "labels"
YAML_PATH = SCRIPT_DIR / "data.yaml"

def convert():
    json_path = SCRIPT_DIR / COCO_JSON
    with open(json_path, "r") as f:
        coco = json.load(f)

    # Build lookup maps
    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Roboflow COCO categories can start at 0 and may include a parent "objects" class.
    # We remap to contiguous 0-based IDs, skipping any supercategory-only entries.
    # Filter out the generic parent category if it has no annotations
    annotated_cat_ids = set(ann["category_id"] for ann in coco["annotations"])
    sorted_cat_ids = sorted(cat_id for cat_id in categories if cat_id in annotated_cat_ids)
    cat_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    yolo_names = {idx: categories[cat_id] for cat_id, idx in cat_id_to_yolo.items()}

    # Group annotations by image_id
    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Create labels directory
    LABELS_DIR.mkdir(exist_ok=True)

    # Convert each image's annotations
    converted = 0
    for img_id, img_info in images.items():
        w, h = img_info["width"], img_info["height"]
        label_name = Path(img_info["file_name"]).stem + ".txt"
        label_path = LABELS_DIR / label_name

        anns = ann_by_image.get(img_id, [])
        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_yolo:
                continue  # skip unannotated parent categories

            yolo_class = cat_id_to_yolo[cat_id]

            # COCO bbox: [x_min, y_min, box_width, box_height]
            bx, by, bw, bh = [float(v) for v in ann["bbox"]]

            # Convert to YOLO: x_center, y_center, width, height (normalized)
            x_center = (bx + bw / 2) / w
            y_center = (by + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h

            lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        converted += 1

    # Write data.yaml
    names_list = [yolo_names[i] for i in range(len(yolo_names))]
    with open(YAML_PATH, "w") as f:
        f.write(f"# YOLOv8 dataset config\n")
        f.write(f"# Auto-generated from {COCO_JSON}\n\n")
        f.write(f"train: .  # images directory (adjust path as needed)\n")
        f.write(f"val: .    # set a separate val path if you have a val split\n\n")
        f.write(f"nc: {len(names_list)}\n")
        f.write(f"names: {names_list}\n")

    print(f"Done! Converted {converted} images.")
    print(f"  Labels:    {LABELS_DIR}")
    print(f"  data.yaml: {YAML_PATH}")
    print(f"  Classes ({len(names_list)}): {names_list}")

if __name__ == "__main__":
    convert()