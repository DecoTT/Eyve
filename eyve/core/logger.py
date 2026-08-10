"""Project-scoped logger that writes to project/logs/ and to stdout."""
from __future__ import annotations
import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"

_app_logger = logging.getLogger("eyve")
_app_logger.setLevel(logging.DEBUG)
if not _app_logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(_FMT, _DATE))
    _app_logger.addHandler(_h)


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
