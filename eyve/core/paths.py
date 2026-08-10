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
