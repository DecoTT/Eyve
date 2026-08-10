"""
License manager — 30-day trial + HMAC-signed device-bound keys with expiry.

State stored in ~/.eyve/license.json
Status values: "trial" | "expired" | "activated" | "license_expired"

Key format:  EYVE-XXXX-XXXX-XXXX-XXXX
             \\___hmac_sig_12___/ \\exp/
  - Groups 1-3 (12 hex chars): first 6 bytes of HMAC-SHA256(SECRET, fp|expiry)
  - Group 4   (4  hex chars):  days since 2025-01-01 encoded as 16-bit hex

Generation:  tools/gen_license.py <device_id> <YYYY-MM-DD>
Validation:  fully offline — no network calls needed.

AGPL note: This is part of the official build experience.
Users who build from source may modify or remove this — that is permitted
under AGPL-3.0. Do not add anti-circumvention language here.
"""
from __future__ import annotations
import hashlib
import hmac as _hmac_mod
import json
from datetime import date, datetime, timedelta
from pathlib import Path

_LICENSE_FILE = Path.home() / ".eyve" / "license.json"
_TRIAL_DAYS   = 30
_EPOCH        = date(2025, 1, 1)   # day-counter base

# HMAC secret — change before release; keep identical in gen_license.py
_SECRET = b"Eyve-SBC-K9mN-pQrS-tUvW-2025"


# ── key generation (also used by gen_license.py) ──────────────────────────────

def make_key(fingerprint: str, expiry_date: str) -> str:
    """
    Generate a license key.

    fingerprint  — device ID, e.g. "A1B2-C3D4-E5F6"
    expiry_date  — "YYYYMMDD"

    Returns "EYVE-XXXX-XXXX-XXXX-XXXX"
    """
    fp_clean = fingerprint.replace("-", "").upper()
    message  = f"{fp_clean}|{expiry_date}".encode()
    sig      = _hmac_mod.new(_SECRET, message, hashlib.sha256).hexdigest().upper()

    exp      = date(int(expiry_date[:4]), int(expiry_date[4:6]), int(expiry_date[6:8]))
    exp_hex  = f"{(exp - _EPOCH).days:04X}"

    h = sig[:12]
    return f"EYVE-{h[:4]}-{h[4:8]}-{h[8:12]}-{exp_hex}"


def _parse_key(key: str) -> tuple[bool, str, str]:
    """
    Parse and validate key structure.
    Returns (ok, hmac_part_12, expiry_YYYYMMDD).
    """
    key   = key.strip().upper().replace(" ", "")
    parts = key.split("-")
    if len(parts) != 5 or parts[0] != "EYVE":
        return False, "", ""
    if not all(len(p) == 4 for p in parts[1:]):
        return False, "", ""
    try:
        int(parts[1] + parts[2] + parts[3], 16)   # must be valid hex
        days     = int(parts[4], 16)
        exp_date = _EPOCH + timedelta(days=days)
        expiry   = exp_date.strftime("%Y%m%d")
    except (ValueError, OverflowError):
        return False, "", ""
    return True, parts[1] + parts[2] + parts[3], expiry


def validate_key(fingerprint: str, key: str) -> tuple[bool, str]:
    """
    Verify key against device fingerprint.
    Returns (valid: bool, message: str).
    message on success = expiry "YYYYMMDD"; on failure = error description.
    """
    ok, hmac_part, expiry = _parse_key(key)
    if not ok:
        return False, "Invalid key format"

    # Check expiry
    exp_date = date(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:8]))
    if exp_date < date.today():
        return False, f"Key expired on {exp_date.strftime('%Y-%m-%d')}"

    # Verify HMAC
    fp_clean = fingerprint.replace("-", "").upper()
    message  = f"{fp_clean}|{expiry}".encode()
    expected = _hmac_mod.new(_SECRET, message, hashlib.sha256).hexdigest().upper()[:12]

    if not _hmac_mod.compare_digest(hmac_part, expected):
        return False, "Key does not match this device"

    return True, expiry


# ── persistence helpers ────────────────────────────────────────────────────────

def _load() -> dict:
    if _LICENSE_FILE.exists():
        try:
            with open(_LICENSE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    _LICENSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_LICENSE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── manager ────────────────────────────────────────────────────────────────────

class LicenseManager:
    def __init__(self) -> None:
        self._data = _load()
        if "first_run" not in self._data:
            self._data["first_run"] = date.today().isoformat()
            _save(self._data)

    def status(self) -> str:
        """Return 'trial' | 'expired' | 'activated' | 'license_expired'."""
        if self._data.get("activated"):
            # Check if the stored license key has expired
            expiry_str = self._data.get("expiry")
            if expiry_str:
                try:
                    exp = date(int(expiry_str[:4]),
                               int(expiry_str[4:6]),
                               int(expiry_str[6:8]))
                    if exp < date.today():
                        return "license_expired"
                except (ValueError, TypeError):
                    pass
            return "activated"
        days = self.days_remaining()
        return "trial" if days > 0 else "expired"

    def days_remaining(self) -> int:
        first   = date.fromisoformat(self._data.get("first_run", date.today().isoformat()))
        elapsed = (date.today() - first).days
        return max(0, _TRIAL_DAYS - elapsed)

    def expiry_date(self) -> str:
        """Return license expiry as 'YYYY-MM-DD', or '' if not activated."""
        raw = self._data.get("expiry", "")
        if len(raw) == 8:
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        return ""

    def activate(self, key: str) -> tuple[bool, str]:
        """
        Validate and activate a license key.
        Returns (success: bool, message: str).
        """
        fp = device_fingerprint()
        ok, result = validate_key(fp, key)
        if ok:
            self._data["activated"]    = True
            self._data["key"]          = key.strip().upper()
            self._data["expiry"]       = result          # "YYYYMMDD"
            self._data["activated_at"] = datetime.now().isoformat()
            _save(self._data)
            return True, result
        return False, result

    def deactivate(self) -> None:
        for k in ("activated", "key", "expiry", "activated_at"):
            self._data.pop(k, None)
        _save(self._data)


# ── device fingerprint ─────────────────────────────────────────────────────────

def device_fingerprint() -> str:
    """
    Return a stable hardware-bound ID formatted as XXXX-XXXX-XXXX.
    Based on MAC address + hostname — stable unless hardware changes.
    """
    import uuid
    import platform
    raw    = f"{uuid.getnode()}-{platform.node()}-eyve"
    digest = hashlib.sha256(raw.encode()).hexdigest().upper()
    return f"{digest[:4]}-{digest[4:8]}-{digest[8:12]}"
