"""
Create, load, save and validate Eyve inspection projects.

Project on disk:
    <root>/
        project.yaml
        classes.yaml
        raw/images/<class_name>/
        tagged/images/  tagged/labels/
        datasets/yolo_dataset/
        models/
        runs/
        production/sessions/  production/screenshots/
        packages/
        logs/
"""
from __future__ import annotations
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from eyve.core.paths import ProjectPaths
from eyve.core.logger import log

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


@dataclass
class ClassDef:
    name: str
    kind: str = "nok"   # "ok" | "nok" | "ignore"
    color: str = "#ff4444"

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "ClassDef":
        return cls(name=d["name"], kind=d.get("kind", "nok"), color=d.get("color", "#ff4444"))


@dataclass
class Project:
    name: str
    root: Path
    target: str = ""
    camera_source: int = 0
    classes: list[ClassDef] = field(default_factory=list)
    active_model: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def paths(self) -> ProjectPaths:
        return ProjectPaths(self.root)

    @property
    def ok_classes(self) -> list[str]:
        return [c.name for c in self.classes if c.kind == "ok"]

    @property
    def nok_classes(self) -> list[str]:
        return [c.name for c in self.classes if c.kind == "nok"]

    @property
    def class_names(self) -> list[str]:
        return [c.name for c in self.classes]

    # ── serialization ────────────────────────────────────────────────────────
    def to_project_dict(self) -> dict:
        return {
            "name": self.name,
            "target": self.target,
            "camera_source": self.camera_source,
            "active_model": self.active_model,
            "created_at": self.created_at,
            "last_modified": datetime.now().isoformat(),
        }

    def to_classes_dict(self) -> dict:
        return {"classes": [c.to_dict() for c in self.classes]}

    def save(self) -> None:
        p = self.paths
        with open(p.project_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_project_dict(), f, allow_unicode=True)
        with open(p.classes_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_classes_dict(), f, allow_unicode=True)
        log.info(f"Project saved: {self.root}")

    # ── statistics ───────────────────────────────────────────────────────────
    def raw_image_count(self) -> int:
        return sum(1 for _ in self.paths.raw_images.rglob("*.jpg"))

    def tagged_image_count(self) -> int:
        return sum(1 for _ in self.paths.tagged_images.glob("*.jpg"))

    def label_count(self) -> int:
        return sum(1 for _ in self.paths.tagged_labels.glob("*.txt"))

    def images_per_class(self) -> dict[str, int]:
        counts: dict[str, int] = {c.name: 0 for c in self.classes}
        for cls_dir in self.paths.raw_images.iterdir():
            if cls_dir.is_dir() and cls_dir.name in counts:
                counts[cls_dir.name] = sum(1 for _ in cls_dir.glob("*.jpg"))
        return counts


# ── module-level helpers ─────────────────────────────────────────────────────

def validate_name(name: str) -> Optional[str]:
    """Return error string or None if valid."""
    if not name or not name.strip():
        return "proj_name_required"
    if not _NAME_RE.match(name.strip()):
        return "proj_name_invalid"
    return None


def create_project(name: str, folder: Path, target: str = "", camera_source: int = 0) -> Project:
    name = name.strip()
    root = Path(folder) / name
    if root.exists():
        raise FileExistsError("proj_exists")
    proj = Project(name=name, root=root, target=target, camera_source=camera_source)
    proj.paths.create_all()
    proj.save()
    log.info(f"Project created: {root}")
    return proj


def load_project(root: Path) -> Project:
    root = Path(root)
    p = ProjectPaths(root)
    if not p.project_yaml.exists():
        raise FileNotFoundError(f"project.yaml not found in {root}")

    with open(p.project_yaml, encoding="utf-8") as f:
        pd = yaml.safe_load(f)

    classes: list[ClassDef] = []
    if p.classes_yaml.exists():
        with open(p.classes_yaml, encoding="utf-8") as f:
            cd = yaml.safe_load(f) or {}
        classes = [ClassDef.from_dict(c) for c in cd.get("classes", [])]

    proj = Project(
        name=pd.get("name", root.name),
        root=root,
        target=pd.get("target", ""),
        camera_source=pd.get("camera_source", 0),
        classes=classes,
        active_model=pd.get("active_model"),
        created_at=pd.get("created_at", ""),
        last_modified=pd.get("last_modified", ""),
    )
    log.info(f"Project loaded: {root}")
    return proj


def delete_project(root: Path) -> None:
    shutil.rmtree(root)
    log.info(f"Project deleted: {root}")
