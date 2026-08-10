"""
Model manager — download and manage YOLOv8 base weights for offline use.

Models are stored in ~/.eyve/models/ so they persist across projects.
When offline_mode is enabled in config, the app uses these local paths
and sets YOLO_OFFLINE=1 to prevent ultralytics from phoning home.
"""
from __future__ import annotations

import threading
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from eyve.core.logger import log
from eyve.core import config

# ── model catalogue ────────────────────────────────────────────────────────────
MODELS: dict[str, dict] = {
    "yolov8n.pt": {
        "label":   "YOLOv8 Nano",
        "size":    "6 MB",
        "bytes":   6_500_000,
        "desc":    "Fastest, lowest accuracy. Good for quick tests.",
        "recommended": True,
    },
    "yolov8s.pt": {
        "label":   "YOLOv8 Small",
        "size":    "22 MB",
        "bytes":   22_000_000,
        "desc":    "Good balance for edge devices.",
        "recommended": False,
    },
    "yolov8m.pt": {
        "label":   "YOLOv8 Medium",
        "size":    "50 MB",
        "bytes":   50_000_000,
        "desc":    "Better accuracy, needs more VRAM.",
        "recommended": False,
    },
    "yolov8l.pt": {
        "label":   "YOLOv8 Large",
        "size":    "84 MB",
        "bytes":   84_000_000,
        "desc":    "High accuracy, GPU recommended.",
        "recommended": False,
    },
}

_BASE_URL = "https://github.com/ultralytics/assets/releases/latest/download/"

# ── paths ──────────────────────────────────────────────────────────────────────

def models_dir() -> Path:
    """Return the local models directory (creates it if needed)."""
    d = Path(config.get("models_dir", str(Path.home() / ".eyve" / "models")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(name: str) -> Path:
    return models_dir() / name


def is_downloaded(name: str) -> bool:
    p = model_path(name)
    if not p.exists():
        return False
    # Sanity-check: file should be at least half the expected size
    expected = MODELS.get(name, {}).get("bytes", 0)
    return expected == 0 or p.stat().st_size >= expected // 2


def local_path_or_name(name: str) -> str:
    """
    If offline mode is on AND the model is downloaded, return the full local
    path so ultralytics uses it without touching the network.
    Otherwise return just the name (ultralytics downloads/caches it as usual).
    """
    if config.get("offline_mode", False) and is_downloaded(name):
        return str(model_path(name))
    return name


# ── downloading ────────────────────────────────────────────────────────────────

def download_model(
    name: str,
    on_progress: Optional[Callable[[float], None]] = None,
    on_done: Optional[Callable[[bool, str], None]] = None,
) -> threading.Thread:
    """
    Download *name* to models_dir() in a daemon thread.

    on_progress(fraction)  — called with 0.0–1.0 from the download thread
    on_done(success, msg)  — called on completion (still in download thread;
                             use .after() if you need to update Tkinter widgets)
    Returns the thread so callers can join or monitor it.
    """
    dest = model_path(name)
    url  = _BASE_URL + name

    def _run() -> None:
        try:
            log.info(f"Downloading {name} from {url} → {dest}")

            def _hook(count: int, block: int, total: int) -> None:
                if on_progress and total > 0:
                    on_progress(min(1.0, count * block / total))

            tmp = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(url, str(tmp), reporthook=_hook)
            tmp.rename(dest)

            log.info(f"Downloaded {name} ({dest.stat().st_size // 1024} KB)")
            if on_done:
                on_done(True, f"{name} saved to {dest}")
        except Exception as e:
            log.error(f"Download failed for {name}: {e}")
            # Clean up partial file
            tmp_path = dest.with_suffix(".tmp")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if on_done:
                on_done(False, str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ── offline env var management ─────────────────────────────────────────────────

def apply_offline_env() -> None:
    """
    Call once at app startup (and whenever offline_mode changes) to set or
    clear the YOLO_OFFLINE environment variable.
    ultralytics respects this flag to skip all hub / internet calls.
    """
    import os
    if config.get("offline_mode", False):
        os.environ["YOLO_OFFLINE"] = "1"
        log.debug("Offline mode ON  (YOLO_OFFLINE=1)")
    else:
        os.environ.pop("YOLO_OFFLINE", None)
        log.debug("Offline mode OFF")
