"""
Máscara contra tela REAL: fotos planas de tela estampada o tejida.

Las fotos no van en el repositorio (son de Envato, con licencia).  Se pasa la
carpeta:

    python mascaras/medir_real.py <carpeta_fotos> [--grupo dev|reserva|todas] [--json salida.json]

Cómo se arma una «línea» con una foto:
  - una ventana de 16:9 recorre la foto a lo ancho, como si la tela pasara bajo
    una cámara fija; la ventana se lleva a 960x540;
  - con un temblor de ±0.6 px en vertical y ±0.3 % de escala por cuadro, que una
    tela real nunca pasa idéntica;
  - aprende con 24 cuadros de la mitad izquierda y prueba en la derecha, que la
    máscara no vio;
  - los fallos se agregan encima de la foto, en 3 lugares de la mitad derecha:
    fantasma (copia corrida y oscura del estampado), falta de tinta (lavado
    hacia el color de la tela) y mancha (oscurece parejo).

Lo que se cuenta por foto: enganche y residuo del informe, cuadros limpios con
marca (falsas), y por fallo, cuadros donde el fallo se ve completo y la
máscara marca una región que lo toca.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eyve.modules.mascara import MascaraAprendida  # noqa: E402

# Separación fija: la reserva no se mira mientras se afina.
DEV = ["2d98bb94", "680c3aec", "707ebbca", "0c8661bc", "92e62421", "e7c3ae5e"]
CONTROL = ["sint800_diamantes", "sint800_rayas", "sint800_flores"]
RESERVA = ["1ba26995", "1337f6ed", "2fa543a1", "59e071fe", "679f7c95", "6d71e62a"]

FALLOS = ("fantasma", "falta_tinta", "mancha")


class Linea:
    def __init__(self, img: np.ndarray, ancho_ventana: int = 300, seed: int = 0):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.ww = min(ancho_ventana, self.W // 2 - 20)
        self.wh = int(self.ww * 9 / 16)
        if self.wh > self.H - 8:
            self.wh = self.H - 8
            self.ww = int(self.wh * 16 / 9)
        self.y0 = (self.H - self.wh) // 2
        self.mitad = (self.W - self.ww) // 2
        self.rng = np.random.default_rng(seed)

    def cuadro(self, x: float, src=None, temblor=True) -> np.ndarray:
        src = self.img if src is None else src
        dy = self.rng.uniform(-0.6, 0.6) if temblor else 0.0
        s = 1.0 + (self.rng.uniform(-0.003, 0.003) if temblor else 0.0)
        sx = 960.0 / self.ww * s
        m = np.float32([[sx, 0, -x * sx], [0, sx, -(self.y0 + dy) * sx]])
        return cv2.warpAffine(src, m, (960, 540), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT)

    def xs_aprender(self, n=24):
        return np.linspace(0, self.mitad - 20, n)

    def xs_prueba(self, n=40):
        return np.linspace(self.mitad + 10, self.W - self.ww - 1, n)

    def con_fallo(self, tipo: str, cx: int, cy: int):
        s = self.img.astype(np.float32)
        r = max(6, int(self.ww * 0.10))
        mk = np.zeros((self.H, self.W), np.float32)
        cv2.circle(mk, (cx, cy), r, 1.0, -1)
        mk = cv2.GaussianBlur(mk, (0, 0), r * 0.25)[..., None]
        if tipo == "fantasma":
            m = np.float32([[1, 0, self.ww * 0.05], [0, 1, self.ww * 0.03]])
            sh = cv2.warpAffine(s, m, (self.W, self.H), borderMode=cv2.BORDER_REFLECT)
            s = s * (1 - 0.45 * mk) + np.minimum(s, sh) * (0.45 * mk)
        elif tipo == "falta_tinta":
            claro = np.percentile(s.reshape(-1, 3), 95, axis=0)
            s = s * (1 - 0.6 * mk) + claro * (0.6 * mk)
        else:
            s = s * (1 - 0.35 * mk)
        caja = (cx - r, cy - r, cx + r, cy + r)
        return np.clip(s, 0, 255).astype(np.uint8), caja


def _toca(region, caja_cuadro) -> bool:
    x0, y0, x1, y1 = region
    a0, b0, a1, b1 = caja_cuadro
    return not (x1 < a0 or a1 < x0 or y1 < b0 or b1 < y0)


def medir_foto(ruta: Path, fabrica=MascaraAprendida, sens: float = 50.0,
               calibrar: bool = False) -> dict:
    img = cv2.imread(str(ruta))
    ln = Linea(img, seed=7)
    t0 = time.time()
    m = fabrica()
    inf = m.aprender([ln.cuadro(x) for x in ln.xs_aprender()])
    out = {"foto": ruta.stem, "enganche": round(inf.enganche, 3),
           "residuo_pct": round(inf.residuo_pct, 2), "avisos": inf.avisos,
           "estado": m.estado}
    if m.estado == "sin_aprender":
        return out
    m.sensibilidad = sens
    if calibrar:
        # tela buena de la mitad de aprender, en posiciones que no se usaron
        xs_a = ln.xs_aprender()
        paso = (xs_a[1] - xs_a[0]) / 2
        out["umbral"] = m.calibrar([ln.cuadro(x + paso) for x in xs_a[:-1]] * 2)
    m.aprobar("medicion")
    xs = ln.xs_prueba()
    falsas = 0
    sueltas = 0
    for x in xs:
        reg = m.analizar(ln.cuadro(x))
        falsas += bool(reg)
        sueltas += 0 if m.enganchada else 1
    out["falsas"] = f"{falsas}/{len(xs)}"
    out["sin_enganche"] = sueltas
    escala = 960.0 / ln.ww
    for tipo in FALLOS:
        vistos = marcados = 0
        for k, fx in enumerate((0.25, 0.5, 0.75)):
            cx = int(ln.mitad + ln.ww * 0.5 + (ln.W - ln.mitad - ln.ww) * fx)
            cy = ln.H // 2 + int((k - 1) * ln.wh * 0.2)
            src, (a0, b0, a1, b1) = ln.con_fallo(tipo, cx, cy)
            for x in xs:
                if not (x + 4 <= a0 and a1 <= x + ln.ww - 4):
                    continue
                vistos += 1
                reg = m.analizar(ln.cuadro(x, src))
                cj = ((a0 - x) * escala, (b0 - ln.y0) * escala,
                      (a1 - x) * escala, (b1 - ln.y0) * escala)
                marcados += any(_toca(r, cj) for r in reg)
        out[tipo] = f"{marcados}/{vistos}"
    out["s"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--grupo", default="dev", choices=["dev", "reserva", "control", "todas"])
    ap.add_argument("--json")
    ap.add_argument("--sens", type=float, default=50.0)
    ap.add_argument("--calibrar", action="store_true")
    a = ap.parse_args()
    carpeta = Path(a.carpeta)
    ids = {"dev": DEV, "reserva": RESERVA, "control": CONTROL}.get(a.grupo, DEV + RESERVA)
    filas = []
    for i in ids:
        ruta = next(iter(sorted(carpeta.glob(i + ".*"))), None)
        if ruta is None:
            print("falta", i)
            continue
        r = medir_foto(ruta, sens=a.sens, calibrar=a.calibrar)
        filas.append(r)
        print(" | ".join(f"{k}={v}" for k, v in r.items() if k != "avisos"),
              ("AVISOS: " + "; ".join(r["avisos"])) if r.get("avisos") else "",
              flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(filas, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
