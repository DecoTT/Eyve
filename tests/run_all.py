"""
Corre todas las pruebas de Eyve y resume.

    python tests/run_all.py            todas
    python tests/run_all.py --rapido   salta las que cargan YOLO

Cada prueba corre en su PROPIO proceso a propósito: las que abren Tk no
pueden compartir intérprete (crear un segundo root después de destruir el
primero deja las PhotoImage apuntando al intérprete muerto), y así una que
reviente no se lleva a las demás.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: (archivo, argumentos, necesita el modelo entrenado de la demo)
SUITES: list[tuple[str, list[str], bool]] = [
    ("test_counting.py",        [],      False),
    ("test_train_paths.py",     [],      False),
    ("test_merge_fragments.py", [],      False),
    ("test_pattern_module.py",  [],      False),
    ("test_anomalias_contadas.py", [],  False),
    ("test_ui_counting.py",     [],      False),
    ("test_textile.py",         [],      False),
    ("test_dataset.py",         [],      False),
    ("test_demo_screen.py",     [],      False),
    ("test_updater.py",         [],      False),
    ("test_update_ui.py",       [],      False),
    ("test_app_integration.py", ["es"],  False),
    ("test_app_integration.py", ["en"],  False),
    ("test_demo_e2e.py",        [],      True),
]


def main() -> int:
    rapido = "--rapido" in sys.argv
    demo_model = (HERE.parent / "projects" / "Demo_Textil" / "runs" / "train"
                  / "weights" / "best.pt")

    resultados = []
    for name, args, needs_model in SUITES:
        etiqueta = f"{name} {' '.join(args)}".strip()
        if rapido and needs_model:
            resultados.append((etiqueta, "saltada", 0.0))
            continue
        if needs_model and not demo_model.exists():
            print(f"[saltada] {etiqueta}: falta el modelo de la demo.\n"
                  f"          python -m eyve.demo.train --out "
                  f'"projects/Demo_Textil"')
            resultados.append((etiqueta, "saltada", 0.0))
            continue

        print(f"\n{'=' * 64}\n  {etiqueta}\n{'=' * 64}")
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(HERE / name), *args])
        dt = time.perf_counter() - t0
        resultados.append((etiqueta, "ok" if proc.returncode == 0 else "FALLA", dt))

    print(f"\n{'=' * 64}\n  RESUMEN\n{'=' * 64}")
    fallas = 0
    for etiqueta, estado, dt in resultados:
        marca = {"ok": "  OK   ", "FALLA": "  FALLA", "saltada": "  --   "}[estado]
        print(f"{marca}  {etiqueta:<34} {dt:6.1f}s")
        fallas += estado == "FALLA"
    print()
    if fallas:
        print(f"{fallas} suite(s) con fallas")
        return 1
    print("todas las suites pasaron")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
