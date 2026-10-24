# -*- coding: utf-8 -*-
"""
Licencia vencida o revocada: sigue funcionando (por honor) pero se nota.

Una Pro vencida sigue siendo Pro —nada bloquea—, así que lo único que deja
constancia ante una auditoría es la marca de agua. sin_amparo() es lo que la
enciende. No toca ~/.eyve: la llave y el estado se ponen en memoria.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eyve.license.license_manager import LicenseManager  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


lm = LicenseManager()
lm._save_state = lambda: None          # nunca escribir el estado real

lm.licencia, lm._state = None, {}
check("sin llave (Free de fábrica) no hay marca", lm.sin_amparo() == "")

lm.licencia = SimpleNamespace(nivel="pro", vencida=False)
check("PREVIO: Pro vigente, sin marca", lm.sin_amparo() == "")

lm.licencia = SimpleNamespace(nivel="pro", vencida=True)
check("Pro vencida → marca 'vencida'", lm.sin_amparo() == "vencida", lm.sin_amparo())
check("y sigue siendo Pro (por honor, no se bloquea)", lm.es_pro() and lm.permite_checks_avanzados())

lm.licencia = SimpleNamespace(nivel="pro", vencida=False)
lm._state = {"servidor": {"estado": "revocada"}}
check("revocada en el último check-in → marca 'revocada'", lm.sin_amparo() == "revocada",
      lm.sin_amparo())
lm._state = {"servidor": {"estado": "vencida"}}
check("vencida según el servidor → marca 'vencida'", lm.sin_amparo() == "vencida", lm.sin_amparo())

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("LICENCIA SIN AMPARO: MARCA DE AGUA, SIN BLOQUEO")
