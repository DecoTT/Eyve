"""
Pack the project dataset as a zip for external training.
Includes images, labels, data.yaml, project metadata, and instructions.
"""
from __future__ import annotations
import json
import zipfile
from datetime import datetime
from pathlib import Path

from eyve.core.project_manager import Project
from eyve.core.logger import log
from eyve.training.dataset_builder import build_dataset, validate_dataset


def detect_size_label(n_images: int) -> tuple[str, str]:
    """Return (key, label) for the package size tier."""
    if n_images <= 500:
        return "small", "pkg_size_small"
    if n_images <= 2000:
        return "medium", "pkg_size_medium"
    if n_images <= 5000:
        return "large", "pkg_size_large"
    return "complex", "pkg_size_complex"


def create_package(project: Project) -> Path:
    """
    Build dataset YAML then zip everything.
    Returns path to the created zip file.
    """
    # ensure dataset is fresh
    yaml_path = build_dataset(project)

    n_images = sum(1 for _ in project.paths.dataset.rglob("*.jpg"))
    size_key, _ = detect_size_label(n_images)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkg_name = f"{project.name}_training_package_{ts}.zip"
    pkg_dir = project.paths.packages
    pkg_dir.mkdir(parents=True, exist_ok=True)
    pkg_path = pkg_dir / pkg_name

    readme = _build_readme(project, size_key, n_images)

    with zipfile.ZipFile(pkg_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # dataset files
        ds = project.paths.dataset
        for f in ds.rglob("*"):
            if f.is_file():
                zf.write(f, f"dataset/{f.relative_to(ds)}")

        # project metadata
        meta = {
            "project": project.name,
            "target": project.target,
            "classes": [c.to_dict() for c in project.classes],
            "size_tier": size_key,
            "n_images": n_images,
            "packed_at": datetime.now().isoformat(),
        }
        zf.writestr("project_meta.json", json.dumps(meta, indent=2))
        zf.writestr("README.txt", readme)

    log.info(f"Package created: {pkg_path}  ({n_images} images, tier={size_key})")
    return pkg_path


def _build_readme(project: Project, size_key: str, n_images: int) -> str:
    return f"""Eyve Training Package
======================
Project  : {project.name}
Target   : {project.target}
Classes  : {', '.join(project.class_names)}
Images   : {n_images}
Size tier: {size_key}
Packed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Contents
--------
dataset/
    images/train/   — training images
    images/val/     — validation images
    labels/train/   — YOLO label files (train)
    labels/val/     — YOLO label files (val)
    data.yaml       — YOLO dataset config

project_meta.json   — project class definitions

How to train on another machine
--------------------------------
1. Copy this zip to the target machine.
2. Unzip it.
3. Install: pip install ultralytics
4. Run:
       yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640

5. Find your trained model in:
       runs/detect/train/weights/best.pt

6. Copy best.pt back into your Eyve project's models/ folder.
7. Open Eyve, go to Production, and load the model.

If you are sending this package to the Eyve team for training,
attach this zip to your support request at eyve.app/training
"""
