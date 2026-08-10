"""App-level config (language, recent projects, window state)."""
from __future__ import annotations
import json
from pathlib import Path

_APP_DIR = Path.home() / ".eyve"
_CONFIG_FILE = _APP_DIR / "config.json"

_DEFAULTS: dict = {
    "language": "en",
    "recent_projects": [],
    "window_width": 1280,
    "window_height": 880,   # +10 % — must match the fallback in app.py
    "theme": "dark",
    "offline_mode": False,
    "models_dir": str(_APP_DIR / "models"),
    # Production loop FPS cap.
    # 0  = uncapped (after(1, …) — camera & inference are the bottleneck)
    # n  = target frames per second → delay = 1000 // n  ms
    "prod_fps_cap": 30,
}


def _load() -> dict:
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, encoding="utf-8") as f:
                data = json.load(f)
            merged = dict(_DEFAULTS)
            merged.update(data)
            return merged
        except Exception:
            pass
    return dict(_DEFAULTS)


def _save(data: dict) -> None:
    _APP_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


_cfg: dict = _load()


def get(key: str, default=None):
    return _cfg.get(key, default)


def set(key: str, value) -> None:
    _cfg[key] = value
    _save(_cfg)


def add_recent_project(path: str) -> None:
    recent: list = _cfg.get("recent_projects", [])
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    _cfg["recent_projects"] = recent[:10]
    _save(_cfg)


def remove_recent_project(path: str) -> None:
    recent: list = _cfg.get("recent_projects", [])
    if path in recent:
        recent.remove(path)
    _cfg["recent_projects"] = recent
    _save(_cfg)


def get_recent_projects() -> list[str]:
    return [p for p in _cfg.get("recent_projects", []) if Path(p).exists()]
