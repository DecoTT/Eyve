# -*- coding: utf-8 -*-
"""
Rompe el codigo a proposito y exige que tests/test_mascara.py lo note.

Una prueba que pasa con el codigo roto no prueba nada.  Cada mutacion se aplica
a una COPIA en una carpeta temporal; el repo no se toca.

    python mutar.py resultados/mutaciones.json [carpeta_temporal]
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
M = "eyve/modules/mascara.py"

MUTACIONES = {
    "marca sin aprobar": (
        M, "if not self.lista or frame is None", "if frame is None"),
    "la huella no cubre la celda": (
        M, "for a in (self._celda, self._peso, self._tinta, self._luz, self._base):",
        "for a in (self._peso, self._tinta, self._luz, self._base):"),
    "piso = maximo (un fallo entre los buenos lo sube)": (
        M, "max(min(med_c, cota_c), _PISO_MIN)", "max(max(pc), _PISO_MIN)"),
    "sin signo (valor absoluto, como Patron)": (
        M, "        rw = r * w\n", "        rw = np.abs(r) * w\n"),
    "umbral a la mitad del piso": (
        M, "return 6.5 - 0.04 * s", "return 0.5"),
    "cajas sin regresar a coordenadas del cuadro": (
        M, "self._regiones.append((int(x * escala), int(y * escala),\n"
           "                                   int((x + w_) * escala),\n"
           "                                   int((y + h_) * escala)))",
        "self._regiones.append((int(x), int(y), int(x + w_), int(y + h_)))"),
    "sin enganche: juzga aunque no calce": (
        M, "if self._enganche_ref and ncc < _ENGANCHE_REL * self._enganche_ref:",
        "if False:"),
    "tipo invertido": (
        M, "self._tipos.append(KIND_FALTA if float(st[sel].max()) > alto\n"
           "                               else KIND_EXTRA)",
        "self._tipos.append(KIND_EXTRA if float(st[sel].max()) > alto\n"
        "                               else KIND_FALTA)"),
    "aprende material que no se repite": (
        M, "if float(vals.max()) < 0.25:", "if False:"),
    "pose solo traslacion": (
        M, "            p -= np.clip(d, -2.0, 2.0)\n",
        "            d[2:] = 0.0\n            p -= np.clip(d, -2.0, 2.0)\n"),
    "la rampa de luz no se sigue": (
        M, "self.seguir_luz: bool = True", "self.seguir_luz: bool = False"),
}


def correr(item):
    nombre, (archivo, viejo, nuevo) = item
    d = Path(tempfile.mkdtemp(prefix="mut_", dir=BASE))
    try:
        for sub in ("eyve", "mascaras"):
            shutil.copytree(REPO / sub, d / sub, ignore=shutil.ignore_patterns(
                "__pycache__", "resultados"))
        (d / "tests").mkdir()
        shutil.copy(REPO / "tests" / "test_mascara.py", d / "tests")
        f = d / archivo
        s = f.read_text(encoding="utf-8")
        s2 = s.replace("\r\n", "\n")
        assert viejo in s2, nombre
        f.write_text(s2.replace(viejo, nuevo), encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, str(d / "tests" / "test_mascara.py")],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=3000)
        fallas = [x.strip()[:110] for x in p.stdout.splitlines()
                  if x.strip().startswith("FALLA ")]
        r = dict(mutacion=nombre, codigo=p.returncode, fallas=fallas,
                 error=p.stderr.strip()[-200:] if p.returncode and not fallas else "")
        print(nombre, "->", "DETECTADA" if p.returncode else "NO DETECTADA",
              len(fallas), flush=True)
        return r
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    BASE = sys.argv[2] if len(sys.argv) > 2 else None
    with ThreadPoolExecutor(5) as ex:
        res = list(ex.map(correr, MUTACIONES.items()))
    json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    malas = [r["mutacion"] for r in res if r["codigo"] == 0]
    print("no detectadas:", malas or "ninguna")
