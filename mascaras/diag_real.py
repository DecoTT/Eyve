"""Diagnóstico visual: cuadro | lo esperado | residuo | mapa de la máscara, para una foto real."""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from eyve.modules.mascara import MascaraAprendida, _gris
from medir_real import Linea

def panel(ruta, salida):
    ln = Linea(cv2.imread(str(ruta)), seed=7)
    m = MascaraAprendida(); inf = m.aprender([ln.cuadro(x) for x in ln.xs_aprender()])
    print(Path(ruta).stem, "base", np.round(m._base, 2).tolist() if m._base is not None else None, "celda", None if m._celda is None else m._celda.shape, "piso", np.round(m._piso, 6), "enganche", round(inf.enganche, 3))
    if m._celda is None: return
    f = ln.cuadro(ln.xs_prueba()[20]); g = _gris(f); gl = g / m._luz
    t = m._fase(gl, m._celda); e = m._render(m._celda, t)
    r = gl - e
    zt, zc, ze, ncc, res = m._mapa(g)
    def u8(a, lo, hi): return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    fila = [u8(gl, 0, 1.2), u8(e, 0, 1.2), u8(np.abs(r), 0, 0.3),
            u8(np.maximum(np.abs(zt) / m._piso[0], np.abs(zc) / m._piso[1]), 0, 6)]
    celda = cv2.resize(u8(m._celda, 0, 1.2), (fila[0].shape[0], fila[0].shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(salida), np.hstack(fila + [celda]))
    print("  pose", np.round(t, 3), "ncc", round(ncc, 3), "residuo med", round(float(np.median(np.abs(r))), 4))

for i in sys.argv[2:]:
    ruta = next(Path(sys.argv[1]).glob(i + ".*"))
    panel(ruta, Path(sys.argv[1]).parent / "diag" / f"diag_{i}.png")
