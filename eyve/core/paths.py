"""Canonical path helper for a loaded project."""
from __future__ import annotations
from pathlib import Path


class ProjectPaths:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ── raw capture ──────────────────────────────────────────────────────────
    @property
    def raw_images(self) -> Path:
        return self.root / "raw" / "images"

    def raw_class_dir(self, class_name: str) -> Path:
        return self.raw_images / class_name

    # ── tagged ───────────────────────────────────────────────────────────────
    @property
    def tagged_images(self) -> Path:
        return self.root / "tagged" / "images"

    @property
    def tagged_labels(self) -> Path:
        return self.root / "tagged" / "labels"

    # ── label naming (canonical — BUG-01) ────────────────────────────────────
    def label_stem(self, img_path: Path) -> str:
        """
        Canonical unique stem for an image's label/tagged-copy filename.

        Derived from the image's path relative to raw/images so that the same
        base name in two class folders never collides:

            raw/images/rayon/img_001.jpg   → "rayon_img_001"
            raw/images/golpe/img_001.jpg   → "golpe_img_001"
            raw/images/img_001.jpg         → "img_001"        (root, no class)

        Every producer and consumer of label files MUST go through this
        function (tagging_screen writes, dataset_builder reads).  Having two
        conventions is exactly the bug that silently broke training.
        """
        img_path = Path(img_path)
        try:
            rel = img_path.relative_to(self.raw_images)
            return str(rel.with_suffix("")).replace("/", "_").replace("\\", "_")
        except ValueError:
            # image lives outside raw/images (shouldn't happen in normal flow)
            return img_path.stem

    def label_file(self, img_path: Path) -> Path:
        """Canonical label .txt path for an image (see label_stem)."""
        return self.tagged_labels / f"{self.label_stem(img_path)}.txt"

    # ── yolo dataset ─────────────────────────────────────────────────────────
    @property
    def dataset(self) -> Path:
        return self.root / "datasets" / "yolo_dataset"

    @property
    def dataset_yaml(self) -> Path:
        return self.dataset / "data.yaml"

    # ── training runs ────────────────────────────────────────────────────────
    @property
    def runs(self) -> Path:
        return self.root / "runs"

    # ── models ───────────────────────────────────────────────────────────────
    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def best_model(self) -> Path:
        return self.models / "best.pt"

    @property
    def last_model(self) -> Path:
        return self.models / "last.pt"

    @property
    def training_metadata(self) -> Path:
        return self.models / "training_metadata.yaml"

    # ── production ───────────────────────────────────────────────────────────
    @property
    def production(self) -> Path:
        return self.root / "production"

    @property
    def sessions(self) -> Path:
        return self.production / "sessions"

    @property
    def screenshots(self) -> Path:
        return self.production / "screenshots"

    # ── config files ─────────────────────────────────────────────────────────
    @property
    def project_yaml(self) -> Path:
        return self.root / "project.yaml"

    @property
    def classes_yaml(self) -> Path:
        return self.root / "classes.yaml"

    # ── packages ─────────────────────────────────────────────────────────────
    @property
    def packages(self) -> Path:
        return self.root / "packages"

    # ── logs ─────────────────────────────────────────────────────────────────
    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # ── helpers ──────────────────────────────────────────────────────────────
    def create_all(self) -> None:
        for d in [
            self.raw_images,
            self.tagged_images,
            self.tagged_labels,
            self.dataset / "images" / "train",
            self.dataset / "images" / "val",
            self.dataset / "labels" / "train",
            self.dataset / "labels" / "val",
            self.runs,
            self.models,
            self.sessions,
            self.screenshots,
            self.packages,
            self.logs,
        ]:
            d.mkdir(parents=True, exist_ok=True)
