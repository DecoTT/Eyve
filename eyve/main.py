"""Eyve 2.1 — entry point."""
from __future__ import annotations
import sys
from pathlib import Path

# ensure project root is in path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from eyve.core import config
    from eyve.i18n import set_language
    from eyve.ui.app import EyveApp

    set_language(config.get("language", "en"))
    app = EyveApp()
    app.mainloop()


if __name__ == "__main__":
    main()
