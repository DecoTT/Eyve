"""
Eyve logger: stdout + a persistent app-level file + (optionally) project/logs/.

The app-level file is what a beta tester attaches to a bug report:
    %USERPROFILE%\\.eyve\\eyve_run.log      (rotates at 2 MB, keeps 3)
It lives in the user profile, not next to the app, so it is always
writable, survives side-by-side updates (2.1 -> 2.1.1) and exists even if
the crash happens before any project is opened.
"""
from __future__ import annotations
import logging
import logging.handlers
import sys
from pathlib import Path

_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"

APP_LOG_DIR  = Path.home() / ".eyve"
APP_LOG_FILE = APP_LOG_DIR / "eyve_run.log"

_app_logger = logging.getLogger("eyve")
_app_logger.setLevel(logging.DEBUG)
if not _app_logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(_FMT, _DATE))
    _app_logger.addHandler(_h)
    try:
        APP_LOG_DIR.mkdir(parents=True, exist_ok=True)
        _fh = logging.handlers.RotatingFileHandler(
            APP_LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        _fh.setFormatter(logging.Formatter(_FMT, _DATE))
        _fh.setLevel(logging.DEBUG)
        _app_logger.addHandler(_fh)
    except Exception:
        pass   # a read-only profile must never keep the app from starting


def install_excepthook() -> None:
    """
    Route uncaught exceptions (main thread AND Tk callbacks) into the log.
    Without this a tester's crash only ever exists in a console window that
    closes with the app — exactly the report we can't act on.
    """
    def _hook(exc_type, exc, tb):
        _app_logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _hook
    try:
        import tkinter as tk
        def _tk_hook(self, exc_type, exc, tb):
            _app_logger.error("Tk callback exception", exc_info=(exc_type, exc, tb))
        tk.Tk.report_callback_exception = _tk_hook
    except Exception:
        pass


def get_logger(name: str = "eyve") -> logging.Logger:
    return logging.getLogger(name)


def attach_project_log(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eyve.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, _DATE))
    fh.setLevel(logging.DEBUG)
    _app_logger.addHandler(fh)


log = _app_logger
