"""
Eyve License Key Generator  —  run this on YOUR machine, never ship it.

Usage:
    python tools/gen_license.py <device_id> <expiry>

Examples:
    python tools/gen_license.py A1B2-C3D4-E5F6 2026-12-31
    python tools/gen_license.py A1B2-C3D4-E5F6 2027-06-30

The device_id is shown in Settings → Device ID.
The expiry is the last day the license is valid (inclusive).

Output:
    License Key: EYVE-3F8A-C21B-9D47-07D0
    Device    : A1B2-C3D4-E5F6
    Expires   : 2026-12-31
"""
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyve.license.license_manager import make_key


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    fingerprint  = sys.argv[1].strip()
    expiry_input = sys.argv[2].strip().replace("-", "")   # accept YYYY-MM-DD or YYYYMMDD

    if len(expiry_input) != 8 or not expiry_input.isdigit():
        print(f"Error: expiry must be YYYY-MM-DD or YYYYMMDD, got: {sys.argv[2]}")
        sys.exit(1)

    key = make_key(fingerprint, expiry_input)
    exp = f"{expiry_input[:4]}-{expiry_input[4:6]}-{expiry_input[6:8]}"

    print()
    print(f"  License Key : {key}")
    print(f"  Device ID   : {fingerprint}")
    print(f"  Expires     : {exp}")
    print()


if __name__ == "__main__":
    main()
