"""
Build a YOLO-compatible dataset from a project's tagged images.

Output structure:
    datasets/yolo_dataset/
        images/train/  images/val/
        labels/train/  labels/val/
        data.yaml
"""
from __future__ import annotations
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from eyve.core.project_manager import Project
from eyve.core.logger import log


@dataclass
class DatasetValidation:
    ok: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_images: int = 0
    labeled_images: int = 0
    train_count: int = 0
    val_count: int = 0
    total_labeled: int = 0          # unfiltered count (for "N nuevas de M")
    per_class: dict[str, int] = field(default_factory=dict)


def collect_pairs(project: Project, since: Optional[datetime] = None
                  ) -> tuple[list[tuple[Path, Path]], int]:
    """
    Return ([(image, label), ...], total_labeled).

    label_file() is the canonical resolver — the SAME one tagging_screen
    writes with.  Using img.stem here was BUG-01 (labels never found).

    since: keep only labels written/edited AFTER this moment (label mtime).
    Used for "solo etiquetas nuevas desde el último entrenamiento": the user
    keeps improving the project model with fresh material only.  total_labeled
    is always the unfiltered count so the UI can say "12 nuevas de 71".
    """
    p = project.paths
    cutoff = since.timestamp() if since else None
    pairs: list[tuple[Path, Path]] = []
    total = 0
    for img in sorted(p.raw_images.rglob("*.jpg")):
        label = p.label_file(img)
        try:
            st = label.stat()
        except OSError:
            continue
        if st.st_size <= 0:
            continue
        total += 1
        if cutoff is not None and st.st_mtime <= cutoff:
            continue
        pairs.append((img, label))
    return pairs, total


def validate_dataset(project: Project, since: Optional[datetime] = None) -> DatasetValidation:
    result = DatasetValidation()
    p = project.paths

    if not project.classes:
        result.errors.append("train_no_data")
        return result

    pairs, total_labeled = collect_pairs(project, since)

    result.total_images = sum(1 for _ in p.raw_images.rglob("*.jpg"))
    result.labeled_images = len(pairs)
    result.total_labeled = total_labeled

    if not pairs:
        result.errors.append("train_no_new_data" if (since and total_labeled) else "train_no_data")
        return result

    # count per class
    class_names = project.class_names
    counts: dict[str, int] = {n: 0 for n in class_names}
    for _, label in pairs:
        seen: set[int] = set()
        for line in label.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) >= 1:
                try:
                    ci = int(parts[0])
                    if ci < len(class_names) and ci not in seen:
                        counts[class_names[ci]] += 1
                        seen.add(ci)
                except ValueError:
                    pass
    result.per_class = counts

    for cls, cnt in counts.items():
        if cnt == 0:
            result.warnings.append(f"Class '{cls}' has 0 labeled images.")
        elif cnt < 10:
            result.warnings.append(f"Class '{cls}' has only {cnt} sample(s). Results may be weak.")

    n = len(pairs)
    val_n = max(1, int(n * 0.15))
    result.train_count = n - val_n
    result.val_count = val_n
    result.ok = True
    return result


def build_dataset(project: Project, val_split: float = 0.15, seed: int = 42,
                  since: Optional[datetime] = None) -> Path:
    """
    Copy tagged images+labels into the YOLO dataset structure.
    Returns the path to data.yaml.
    since: only labels newer than this (see collect_pairs).
    """
    p = project.paths
    ds = p.dataset

    # clean previous split
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = ds / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    pairs, _ = collect_pairs(project, since)
    if not pairs:
        raise ValueError("No labeled images found."
                         if since is None else "No new labels since last training.")

    random.seed(seed)
    random.shuffle(pairs)
    val_n = max(1, int(len(pairs) * val_split))
    val_set = set(range(len(pairs) - val_n, len(pairs)))

    for i, (img, lbl) in enumerate(pairs):
        split = "val" if i in val_set else "train"
        # YOLO pairs image↔label BY FILENAME, so the image must be copied
        # under the same unique stem as its label.  Copying img.name would
        # (a) mismatch the label name and (b) collide when two class folders
        # contain the same base filename.
        stem = p.label_stem(img)
        shutil.copy2(img, ds / "images" / split / f"{stem}{img.suffix}")
        shutil.copy2(lbl, ds / "labels" / split / f"{stem}.txt")

    # write data.yaml
    data_yaml = {
        "path": str(ds.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(project.classes),
        "names": {i: c.name for i, c in enumerate(project.classes)},
    }
    yaml_path = ds / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    log.info(f"Dataset built: {len(pairs)} images → {yaml_path}")
    return yaml_path
