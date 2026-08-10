"""
Enumerate available cameras with human-readable names on Windows.

Name resolution order (first non-empty result wins):
1. MFEnumDeviceSources via ctypes  — exact same call OpenCV CAP_MSMF uses,
   so the returned list index == VideoCapture index.  100% correct order.
2. Registry …\\Class\\{ca3e7ab9-…}  modern Win10/11 camera class
3. Registry …\\Class\\{65E8773D-…}  legacy DirectShow class
4. SetupDI via ctypes                same underlying API as DirectShow
5. WMIC Win32_PnPEntity             fallback — order may differ

Background-thread safe: all functions are read-only / stateless.
"""
from __future__ import annotations
import ctypes
import ctypes.wintypes as wt
import subprocess
import winreg
from dataclasses import dataclass, field
from typing import Optional

import cv2
from eyve.core.logger import log


# ── data class ────────────────────────────────────────────────────────────────
@dataclass
class CameraDevice:
    index: int
    name: str
    backend: int = field(default_factory=lambda: cv2.CAP_MSMF)

    @property
    def label(self) -> str:
        return f"{self.name}  ({self.index})"

    @property
    def short_label(self) -> str:
        n = self.name if len(self.name) <= 28 else self.name[:26] + "…"
        return f"{n}  ({self.index})"


# ── COM / GUID helpers ────────────────────────────────────────────────────────
class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wt.DWORD),
        ("Data2", wt.WORD),
        ("Data3", wt.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(s: str) -> _GUID:
    g = s.strip("{}")
    p = g.split("-")
    d4 = bytes.fromhex(p[3] + p[4])
    return _GUID(int(p[0], 16), int(p[1], 16), int(p[2], 16),
                 (_GUID._fields_[3][1])(*d4))


def _vtfn(ptr, slot, restype, *argtypes):
    """Resolve COM vtable slot into a callable."""
    vtbl = ctypes.cast(
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0],
        ctypes.POINTER(ctypes.c_void_p)
    )
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[slot])


# ── Method 1: MFEnumDeviceSources ─────────────────────────────────────────────
def _mf_enum_names() -> list[str]:
    """
    Call Windows MFEnumDeviceSources via ctypes.
    OpenCV CAP_MSMF calls the same function internally, so the returned
    list is guaranteed to be in the same order as VideoCapture indices.
    """
    try:
        mfplat = ctypes.WinDLL("MFPlat.DLL")
        mf_dll = ctypes.WinDLL("MF.DLL")
        ole32  = ctypes.windll.ole32

        # CoInitializeEx — ignore RPC_E_CHANGED_MODE / S_FALSE
        ole32.CoInitializeEx(None, 0)

        # MFStartup(MF_VERSION=0x00020070, MFSTARTUP_NOSOCKET=1)
        mfplat.MFStartup.restype  = ctypes.HRESULT
        mfplat.MFStartup.argtypes = [wt.DWORD, wt.DWORD]
        if mfplat.MFStartup(0x00020070, 1) < 0:
            return []

        names: list[str] = []
        try:
            # Create IMFAttributes
            mfplat.MFCreateAttributes.restype  = ctypes.HRESULT
            mfplat.MFCreateAttributes.argtypes = [ctypes.POINTER(ctypes.c_void_p), wt.UINT]
            pAttr = ctypes.c_void_p()
            if mfplat.MFCreateAttributes(ctypes.byref(pAttr), 1) < 0 or not pAttr:
                return []
            try:
                # pAttr->SetGUID(SOURCE_TYPE, VIDCAP_GUID)  — slot 24
                MF_SRC_TYPE = _guid("C60AC5FE-252A-478F-A0EF-BC8FA5F7CAD3")
                MF_VIDCAP   = _guid("8AC3587A-4AE7-42D8-99E0-0A6013EEF90F")
                hr = _vtfn(pAttr, 24, ctypes.HRESULT,
                           ctypes.POINTER(_GUID), ctypes.POINTER(_GUID))(
                    pAttr, ctypes.byref(MF_SRC_TYPE), ctypes.byref(MF_VIDCAP))
                if hr < 0:
                    return []

                # MFEnumDeviceSources
                mf_dll.MFEnumDeviceSources.restype  = ctypes.HRESULT
                mf_dll.MFEnumDeviceSources.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(wt.UINT),
                ]
                ppAct  = ctypes.c_void_p()
                count  = wt.UINT(0)
                hr = mf_dll.MFEnumDeviceSources(pAttr,
                                                ctypes.byref(ppAct),
                                                ctypes.byref(count))
                if hr >= 0 and count.value > 0:
                    arr = ctypes.cast(ppAct, ctypes.POINTER(ctypes.c_void_p))
                    MF_FRIENDLY = _guid("60D0E559-52F8-4FA2-BBCE-ACDB34A8EC01")
                    for i in range(count.value):
                        pAct = arr[i]
                        if not pAct:
                            names.append(f"Camera {i}")
                            continue
                        # GetAllocatedString — slot 13
                        name_p = ctypes.c_wchar_p()
                        name_n = wt.UINT(0)
                        hr2 = _vtfn(pAct, 13, ctypes.HRESULT,
                                    ctypes.POINTER(_GUID),
                                    ctypes.POINTER(ctypes.c_wchar_p),
                                    ctypes.POINTER(wt.UINT))(
                            pAct, ctypes.byref(MF_FRIENDLY),
                            ctypes.byref(name_p), ctypes.byref(name_n))
                        names.append(name_p.value if hr2 >= 0 and name_p.value
                                     else f"Camera {i}")
                        if hr2 >= 0 and name_p.value:
                            ole32.CoTaskMemFree(name_p)
                        # Release IMFActivate — slot 2
                        _vtfn(pAct, 2, wt.ULONG)(pAct)
                    ole32.CoTaskMemFree(ppAct)
            finally:
                _vtfn(pAttr, 2, wt.ULONG)(pAttr)   # Release IMFAttributes
        finally:
            mfplat.MFShutdown()

        return names
    except Exception as e:
        log.debug(f"MFEnumDeviceSources failed: {e}")
        return []


# ── Method 2 & 3: registry ────────────────────────────────────────────────────
def _registry_names(key_path: str) -> list[str]:
    names: list[str] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
        i = 0
        while True:
            try:
                sub_name = winreg.EnumKey(root, i)
                i += 1
                if not sub_name.isdigit():
                    continue
                try:
                    sub = winreg.OpenKey(root, sub_name)
                    try:
                        val, _ = winreg.QueryValueEx(sub, "FriendlyName")
                        if val:
                            names.append(str(val))
                    except (FileNotFoundError, OSError):
                        pass
                    finally:
                        winreg.CloseKey(sub)
                except OSError:
                    pass
            except OSError:
                break
        winreg.CloseKey(root)
    except Exception as e:
        log.debug(f"Registry [{key_path[-38:]}]: {e}")
    return names


# ── Method 4: SetupDI ─────────────────────────────────────────────────────────
def _setupdi_names(guid_str: str) -> list[str]:
    try:
        guid = _guid(guid_str)

        class _SP(ctypes.Structure):
            _fields_ = [("cbSize", wt.DWORD), ("ClassGuid", _GUID),
                        ("DevInst", wt.DWORD), ("Reserved", ctypes.c_ulong)]

        sapi = ctypes.windll.setupapi
        DIGCF_PRESENT   = 0x2
        SPDRP_FRIENDLY  = 0xC
        SPDRP_DESC      = 0x0
        INVALID         = wt.HANDLE(-1).value

        hDev = sapi.SetupDiGetClassDevsW(
            ctypes.byref(guid), None, None, DIGCF_PRESENT)
        if hDev == INVALID:
            return []

        names: list[str] = []
        try:
            idx = 0
            while True:
                info = _SP()
                info.cbSize = ctypes.sizeof(_SP)
                if not sapi.SetupDiEnumDeviceInfo(hDev, idx, ctypes.byref(info)):
                    break
                idx += 1
                buf  = ctypes.create_unicode_buffer(512)
                bsz  = wt.DWORD(1024)
                rt   = wt.DWORD()
                got  = False
                for prop in (SPDRP_FRIENDLY, SPDRP_DESC):
                    if sapi.SetupDiGetDeviceRegistryPropertyW(
                        hDev, ctypes.byref(info), prop,
                        ctypes.byref(rt),
                        ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte)),
                        wt.DWORD(1024), ctypes.byref(bsz)
                    ):
                        names.append(buf.value)
                        got = True
                        break
                if not got:
                    names.append(f"Camera {idx - 1}")
        finally:
            sapi.SetupDiDestroyDeviceInfoList(hDev)
        return names
    except Exception as e:
        log.debug(f"SetupDI [{guid_str[:8]}…]: {e}")
        return []


# ── Method 5: WMIC fallback ───────────────────────────────────────────────────
def _wmic_names() -> list[str]:
    try:
        r = subprocess.run(
            ["wmic", "path", "Win32_PnPEntity",
             "where", "(PNPClass='Camera' or PNPClass='Image')",
             "get", "Name", "/format:list"],
            capture_output=True, text=True, timeout=6)
        return [l[5:].strip() for l in r.stdout.splitlines()
                if l.strip().lower().startswith("name=") and l.strip()[5:]]
    except Exception as e:
        log.debug(f"WMIC: {e}")
        return []


def _get_fallback_names() -> list[str]:
    """
    Try registry / SetupDI / WMIC methods in order.
    These don't guarantee correct index order but are better than nothing.
    """
    METHODS = [
        ("registry-camera-new",  lambda: _registry_names(
            r"SYSTEM\CurrentControlSet\Control\Class"
            r"\{ca3e7ab9-b4c3-4ae6-8251-579ef933890f}")),
        ("registry-ds-old",      lambda: _registry_names(
            r"SYSTEM\CurrentControlSet\Control\Class"
            r"\{65E8773D-8F56-11D0-A3B9-00A0C9223196}")),
        ("setupdi-camera-new",   lambda: _setupdi_names(
            "ca3e7ab9-b4c3-4ae6-8251-579ef933890f")),
        ("setupdi-ds-old",       lambda: _setupdi_names(
            "65E8773D-8F56-11D0-A3B9-00A0C9223196")),
        ("wmic",                 _wmic_names),
    ]
    for label, fn in METHODS:
        try:
            result = fn()
        except Exception:
            result = []
        if result:
            log.debug(f"Camera names [{label}]: {result}")
            return result
    return []


# ── probing ───────────────────────────────────────────────────────────────────
def _probe(index: int) -> Optional[int]:
    """Open and immediately close a VideoCapture to confirm a camera exists."""
    import time as _time
    t0 = _time.monotonic()
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.release()
            log.debug(f"  probe({index}) → backend={backend}  "
                      f"{(_time.monotonic()-t0)*1000:.0f} ms")
            return backend
    log.debug(f"  probe({index}) → not found  "
              f"{(_time.monotonic()-t0)*1000:.0f} ms")
    return None


# ── main enumeration ──────────────────────────────────────────────────────────
def enumerate_cameras(max_index: int = 8) -> list[CameraDevice]:
    """
    Return CameraDevice list.  Safe to call from a background thread.

    Fast path (MFEnumDeviceSources):
        If the MF API returns names, indices ARE the OpenCV indices and the
        backend IS CAP_MSMF — no probing needed.  This saves ~2 s per camera.

    Slow path (fallback methods):
        Use registry/SetupDI/WMIC for names, then probe each index to confirm
        the camera exists and find its preferred backend.
    """
    import time as _time
    t0 = _time.monotonic()

    # ── fast path ──────────────────────────────────────────────────────────────
    mf_names = _mf_enum_names()
    if mf_names:
        devices = [
            CameraDevice(index=i, name=mf_names[i], backend=cv2.CAP_MSMF)
            for i in range(min(len(mf_names), max_index))
        ]
        log.debug(f"Cameras [MFEnumDeviceSources, no probe, "
                  f"{(_time.monotonic()-t0)*1000:.0f} ms]: "
                  f"{[d.label for d in devices]}")
        return devices

    # ── slow path (probe required) ─────────────────────────────────────────────
    log.debug("MFEnumDeviceSources empty → probe-based fallback")
    ordered = _get_fallback_names()
    devices: list[CameraDevice] = []

    if ordered:
        for i in range(min(len(ordered), max_index)):
            backend = _probe(i)
            if backend is not None:
                devices.append(CameraDevice(index=i, name=ordered[i], backend=backend))
        if devices:
            log.debug(f"Cameras [fallback+probe, "
                      f"{(_time.monotonic()-t0)*1000:.0f} ms]: "
                      f"{[d.label for d in devices]}")
            return devices

    # ── last resort: blind probe across all indices ────────────────────────────
    log.debug("No ordered names — blind probe")
    fb = _wmic_names()
    used = 0
    for i in range(max_index):
        backend = _probe(i)
        if backend is not None:
            name = fb[used] if used < len(fb) else f"Camera {i}"
            devices.append(CameraDevice(index=i, name=name, backend=backend))
            used += 1
    log.debug(f"Cameras [blind probe, {(_time.monotonic()-t0)*1000:.0f} ms]: "
              f"{[d.label for d in devices]}")
    return devices


def get_camera_labels(max_index: int = 8) -> list[str]:
    devs = enumerate_cameras(max_index)
    return [d.short_label for d in devs] if devs else [str(i) for i in range(4)]


def label_to_index(label: str) -> int:
    label = label.strip()
    if "(" in label and label.endswith(")"):
        try:
            return int(label[label.rfind("(") + 1:-1].strip())
        except ValueError:
            pass
    try:
        return int(label)
    except ValueError:
        return 0


def label_to_device(label: str) -> Optional[CameraDevice]:
    idx = label_to_index(label)
    for dev in enumerate_cameras():
        if dev.index == idx:
            return dev
    return None
