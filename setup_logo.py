"""
Run once to install the Eyve logo files into eyve/assets/.

Usage:
    python setup_logo.py <logo_black.png> <logo_white.png>

Both files must already have a transparent background:
    logo_black.png  — black logo, transparent bg  (shown in LIGHT mode)
    logo_white.png  — white logo, transparent bg  (shown in DARK  mode)

You can omit the second argument if you only have one variant.
After running this once you can delete this file.
"""
import sys
import shutil
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python setup_logo.py <logo_black.png> [logo_white.png]")
        sys.exit(1)

    assets = Path(__file__).parent / "eyve" / "assets"
    assets.mkdir(exist_ok=True)

    black_src = Path(sys.argv[1])
    if not black_src.exists():
        print(f"File not found: {black_src}")
        sys.exit(1)
    shutil.copy2(black_src, assets / "logo_black.png")
    print(f"✓  {assets / 'logo_black.png'}")

    if len(sys.argv) >= 3:
        white_src = Path(sys.argv[2])
        if not white_src.exists():
            print(f"File not found: {white_src}")
            sys.exit(1)
        shutil.copy2(white_src, assets / "logo_white.png")
        print(f"✓  {assets / 'logo_white.png'}")
    else:
        print("(only one variant provided — same logo used for both modes)")

    print("\nDone — you can now delete setup_logo.py")


if __name__ == "__main__":
    main()
