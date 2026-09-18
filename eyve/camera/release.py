"""
COM-safe release of cv2.VideoCapture objects.

WHY THIS EXISTS
---------------
OpenCV's DirectShow backend (CAP_DSHOW) keeps a process-wide COM init
counter.  Whichever thread performs the release that drives that counter
to zero gets a CoUninitialize() call.  If that thread is the Tk main
thread, it tears down the COM apartment Tk uses for native dialogs, and
every later filedialog.askopenfilename() fails with:

    _tkinter.TclError: CoInitialize has not been called

Verified empirically (scratchpad/test_com.py): release on main -> the
FileOpenDialog CoCreateInstance returns CO_E_NOTINITIALIZED; release on
any other thread -> main stays healthy.

RULE: never call cap.release() on the Tk main thread.  Either let the
grab thread that reads from the capture release it on exit, or hand it
to release_async() for captures that never got a grab loop (pre-warm,
open failures, user switched camera mid-warm-up).
"""
from __future__ import annotations
import threading
from typing import Optional

import cv2


def release_async(cap: Optional[cv2.VideoCapture]) -> None:
    """Release *cap* on a throwaway daemon thread (never on the caller)."""
    if cap is None:
        return
    threading.Thread(target=_safe_release, args=(cap,),
                     daemon=True, name="cap-release").start()


def _safe_release(cap: cv2.VideoCapture) -> None:
    try:
        cap.release()
    except Exception:
        pass
