"""
YOLOv8 Training Script — Metin2 Object Detection

Usage:
    python train_yolo.py
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import random
import os

# ========================
#  CONFIGURATION
# ========================

DATASET_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL = "yolov8n.pt"

# Patience starts at 20 and shrinks as epochs progress.
# Early on, give the model room to learn. Later, cut it off faster.
EPOCHS = 150
IMGSZ = 640
BATCH = 4
PATIENCE = 20          # starting patience — actual value set via callback
DEVICE = "cpu"
VAL_SPLIT = 0.15


def adaptive_patience_callback(trainer):
    """Shrink patience as training progresses: 20 → 8 linearly."""
    progress = trainer.epoch / trainer.epochs  # 0.0 → 1.0
    trainer.stopper.patience = int(20 - 12 * progress)  # 20 early → 8 late


# ========================
#  AUTO TRAIN/VAL SPLIT
# ========================

def create_split():
    images_dir = DATASET_DIR / "images"
    labels_dir = DATASET_DIR / "labels"
    train_img = DATASET_DIR / "train" / "images"
    val_img = DATASET_DIR / "val" / "images"

    if train_img.exists() and val_img.exists():
        print("Train/val split already exists, skipping...")
        return

    if images_dir.exists():
        img_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    else:
        img_files = list(DATASET_DIR.glob("*.png")) + list(DATASET_DIR.glob("*.jpg"))

    if not img_files:
        print(f"ERROR: No images found in {images_dir}")
        return
    if not labels_dir.exists():
        print("ERROR: No 'labels/' folder. Run coco_to_yolo.py first.")
        return

    random.seed(42)
    random.shuffle(img_files)
    val_count = max(1, int(len(img_files) * VAL_SPLIT))

    for subset, files in [
        ("train", img_files[val_count:]),
        ("val", img_files[:val_count]),
    ]:
        img_dst = DATASET_DIR / subset / "images"
        lbl_dst = DATASET_DIR / subset / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img_path in files:
            shutil.copy2(img_path, img_dst / img_path.name)
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

    print(f"Split: {len(img_files) - val_count} train, {val_count} val")


# ========================
#  TRAIN
# ========================

def train():
    create_split()

    yaml_path = DATASET_DIR / "data.yaml"
    train_path = (DATASET_DIR / "train" / "images").resolve()
    val_path = (DATASET_DIR / "val" / "images").resolve()

    with open(yaml_path, "w") as f:
        f.write(f"path: {DATASET_DIR.resolve()}\n")
        f.write(f"train: {train_path}\n")
        f.write(f"val: {val_path}\n\n")
        f.write("nc: 6\n")
        f.write("names:\n")
        for i, name in enumerate(["Boss", "Buff", "Enemy", "Me", "Statue", "boulder"]):
            f.write(f"  {i}: {name}\n")

    model = YOLO(MODEL)
    model.add_callback("on_train_epoch_end", adaptive_patience_callback)
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.2,
        translate=0.1, scale=0.3,
        mosaic=1.0, flipud=0.0, fliplr=0.3, mixup=0.1,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        warmup_epochs=3, cos_lr=True,
        project="metin2_detect", name="boulder", verbose=True,
    )

    metrics = model.val()
    print(f"\nmAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"Best model: metin2_detect/boulder/weights/best.pt")


if __name__ == "__main__":
    train()