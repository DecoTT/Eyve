"""
Ensamble — verificación de ensamble por pieza, en una estación con operador.

El detector reconoce PARTES (categorías normales del proyecto: la base y lo
que se le monta, p. ej. base / antena / etiqueta). Ensamble decide si la
pieza está bien armada:

  1. Referencia: con una pieza OK bajo la cámara, "Aprender referencia"
     guarda dónde va cada parte RESPECTO A LA BASE (caja normalizada a la
     caja de la base). Por eso no importa dónde caiga la pieza en el
     encuadre: todo se mide contra su propia base.
  2. Por cada pieza, para cada parte de la referencia:
        falta                       → defecto "falta <parte>"   → RECHAZO
        centro o tamaño fuera de    → "<parte> fuera de posición"
        la tolerancia                 → corregir y reinspeccionar
     Sin defectos → OK.

Y como en una estación manual el frame suelto no significa nada (la mano
tapa, la pieza se mueve), el veredicto es por PIEZA:

    VACIO ──base──▶ MANO/MOVIMIENTO ──quieta, sin mano──▶ ASENTANDO
          ──settle──▶ INSPECCIONANDO ──ventana──▶ RESULTADO (fijo)

  - Con mano en zona o la escena en movimiento, la inspección se pausa.
  - Quieta y sin mano: junta votos una ventana corta y fija el veredicto por
    mayoría; no parpadea aunque un frame falle.
  - Si la mano vuelve (a corregir) y se va, reinspecciona la MISMA pieza.
  - Al irse la pieza se cierra: OK / Corregida (hubo NOT OK y terminó OK) /
    Rechazo (se fue con NOT OK).

La mano se reconoce por el color del guante (HSV) y, además, cualquier
movimiento pausa: con un guante de otro color sigue funcionando apoyado en el
movimiento. Todo va por TIEMPO, no por frames: en CPU el detector se salta
frames y el módulo no debe depender de eso.
"""
from __future__ import annotations

import time
from collections import Counter
from types import SimpleNamespace
from types import SimpleNamespace
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from eyve.modules.base import InspectionModule, ModuleVerdict

#: presets de guante: rango HSV de OpenCV (H 0-180)
GLOVES = {
    "azul":   ((95, 90, 60), (128, 255, 255)),
    "negro":  ((0, 0, 0), (180, 80, 45)),
    "blanco": ((0, 0, 200), (180, 35, 255)),
    "ninguno": None,
}

_VERDE, _ROJO, _GRIS, _AMBAR = (60, 190, 60), (40, 40, 230), (134, 126, 120), (20, 160, 240)


def _font(n: int):
    try:
        from PIL import ImageFont
        return ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", n)
    except Exception:
        return None


def _rel(box, base) -> list[float]:
    """Caja (x1,y1,x2,y2) en unidades de la base: 0..1 dentro de ella."""
    bx1, by1, bx2, by2 = base
    bw, bh = max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)
    return [(box[0] - bx1) / bw, (box[1] - by1) / bh, (box[2] - bx1) / bw, (box[3] - by1) / bh]


def _abs(rel, base) -> tuple[int, int, int, int]:
    bx1, by1, bx2, by2 = base
    bw, bh = bx2 - bx1, by2 - by1
    return (int(bx1 + rel[0] * bw), int(by1 + rel[1] * bh),
            int(bx1 + rel[2] * bw), int(by1 + rel[3] * bh))


def from_tracks(tracks, max_lost: int = 3) -> list:
    """
    Instancias del tracker → detecciones para el módulo.

    El ensamble se juzga sobre las instancias PERSISTENTES (las mismas que
    dibuja Producción), no sobre la detección cruda de cada frame: si el
    detector pierde la antena un frame, la instancia sigue ahí y la pieza no
    "pierde" la antena. Solo confirmadas, y como mucho *max_lost* frames sin
    verse: una parte que de verdad se quitó no debe seguir contando como
    puesta.
    """
    out = []
    for tr in tracks:
        if not getattr(tr, "confirmed", True) or tr.frames_lost > max_lost:
            continue
        x1, y1, x2, y2 = tr.bbox
        out.append(SimpleNamespace(class_name=tr.label, confidence=tr.confidence,
                                   x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))
    return out


def from_tracks(tracks, max_lost: int = 3) -> list:
    """
    Instancias del tracker → detecciones para el módulo.

    El ensamble se juzga sobre las instancias PERSISTENTES (las mismas que
    dibuja Producción), no sobre la detección cruda de cada frame: si el
    detector pierde la antena un frame, la instancia sigue ahí y la pieza no
    "pierde" la antena. Solo confirmadas, y como mucho *max_lost* frames sin
    verse: una parte que de verdad se quitó no debe seguir contando como
    puesta.
    """
    out = []
    for tr in tracks:
        if not getattr(tr, "confirmed", True) or tr.frames_lost > max_lost:
            continue
        x1, y1, x2, y2 = tr.bbox
        out.append(SimpleNamespace(class_name=tr.label, confidence=tr.confidence,
                                   x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))
    return out


def from_tracks(tracks, max_lost: int = 3) -> list:
    """
    Instancias del tracker → detecciones para el módulo.

    El ensamble se juzga sobre las instancias PERSISTENTES (las mismas que
    dibuja Producción), no sobre la detección cruda de cada frame: si el
    detector pierde la antena un frame, la instancia sigue ahí y la pieza no
    "pierde" la antena. Solo confirmadas, y como mucho *max_lost* frames sin
    verse: una parte que de verdad se quitó no debe seguir contando como
    puesta.
    """
    out = []
    for tr in tracks:
        if not getattr(tr, "confirmed", True) or tr.frames_lost > max_lost:
            continue
        x1, y1, x2, y2 = tr.bbox
        out.append(SimpleNamespace(class_name=tr.label, confidence=tr.confidence,
                                   x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))
    return out


class AssemblyModule(InspectionModule):
    name = "Ensamble"
    key = "assembly"
    pro = True
    pro = True

    def __init__(self) -> None:
        super().__init__()
        self.target_class = "base"          # la clase que es la pieza
        self.reference: dict[str, list[float]] = {}   # parte → caja relativa a la base
        self.tolerance: float = 0.07        # centro ±7 % de la base; tamaño ±1.7×
        self.glove: str = "azul"
        self.hand_min: float = 0.004        # fracción del frame con color de guante
        self.hand_inside_min: float = 0.03  # ídem, encima de la pieza (sus partes no cuentan)
        self.motion_min: float = 0.004      # fracción del frame que cambió
        self.settle_s: float = 0.35
        self.window_s: float = 0.6
        self.gone_s: float = 0.5
        self.min_conf: float = 0.5
        #: una parte es de la pieza si su centro cae en la base ampliada este
        #: tanto por lado: antenas, cables o conectores que sobresalen
        self.part_margin: float = 0.4
        self._fonts = {}
        self._learn: dict[str, list] = {}
        self.reset()

    # ── estado ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.state = "vacio"
        self.counts = {"ok": 0, "corregida": 0, "rechazo": 0}
        self.piece_n = 0
        self._prev: Optional[np.ndarray] = None
        self._still_since: Optional[float] = None
        self._absent_since: Optional[float] = None
        self._changed_since: Optional[float] = None   # resultado que ya no se ve
        self._votes: list[tuple[bool, frozenset]] = []
        self._window_t0 = 0.0
        self._history: list[bool] = []          # veredictos de la pieza actual
        self.verdict: Optional[str] = None      # "OK" / "NOT_OK" fijo
        self.defects: frozenset = frozenset()
        self.last_closed: Optional[str] = None
        self.hand = False
        self.moving = False
        self._base = None                       # caja de la base, último frame
        self._parts: dict[str, tuple] = {}      # parte → caja, último frame

    @property
    def calibrated(self) -> bool:
        return bool(self.reference)

    @property
    def paused(self) -> bool:
        return self.state in ("mano", "asentando", "vacio")

    # ── referencia ────────────────────────────────────────────────────────
    def learn(self, detections: list) -> int:
        """Acumula una muestra de la pieza OK; devuelve cuántas lleva."""
        base, parts = self._split(detections)
        if base is None:
            return len(self._learn.get("_n", []))
        self._learn.setdefault("_n", []).append(1)
        for name, box in parts.items():
            self._learn.setdefault(name, []).append(_rel(box, base))
        return len(self._learn["_n"])

    def finish_learning(self) -> bool:
        n = len(self._learn.get("_n", []))
        if n == 0:
            return False
        # una parte entra a la referencia si estuvo en la mayoría de las muestras
        self.reference = {k: [float(v) for v in np.median(np.array(cajas), axis=0)]
                          for k, cajas in self._learn.items()
                          if k != "_n" and len(cajas) >= n / 2}
        self._learn = {}
        self.reset()
        return bool(self.reference)

    # ── geometría ─────────────────────────────────────────────────────────
    def _split(self, detections):
        """La base más confiable y, por parte, la detección dentro de ella."""
        ok = [d for d in detections if d.confidence >= self.min_conf]
        bases = [d for d in ok if d.class_name == self.target_class]
        if not bases:
            return None, {}
        b = max(bases, key=lambda d: d.confidence)
        base = (b.x1, b.y1, b.x2, b.y2)
        mx = self.part_margin * (base[2] - base[0])
        my = self.part_margin * (base[3] - base[1])
        parts: dict[str, tuple] = {}
        best: dict[str, float] = {}
        for d in ok:
            if d.class_name == self.target_class:
                continue
            cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
            if not (base[0] - mx <= cx <= base[2] + mx and base[1] - my <= cy <= base[3] + my):
                continue
            if d.confidence > best.get(d.class_name, 0):
                best[d.class_name] = d.confidence
                parts[d.class_name] = (d.x1, d.y1, d.x2, d.y2)
        return base, parts

    def check(self, base, parts) -> frozenset:
        """Defectos de una pieza contra la referencia ("falta:x" / "pos:x")."""
        out = set()
        t = self.tolerance
        for name, ref in self.reference.items():
            box = parts.get(name)
            if box is None:
                out.add(f"falta:{name}")
                continue
            r = _rel(box, base)
            dc = max(abs((r[0] + r[2]) - (ref[0] + ref[2])) / 2,
                     abs((r[1] + r[3]) - (ref[1] + ref[3])) / 2)
            ds = max(abs((r[2] - r[0]) - (ref[2] - ref[0])),
                     abs((r[3] - r[1]) - (ref[3] - ref[1])))
            if dc > t or ds > 1.7 * t:
                out.add(f"pos:{name}")
        return frozenset(out)

    # ── señales de la escena ──────────────────────────────────────────────
    def _signals(self, frame: np.ndarray, base=None) -> tuple[bool, bool]:
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        hand = False
        rng = GLOVES.get(self.glove)
        if rng:
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(rng[0]), np.array(rng[1]))
            dentro = 0.0
            if base is not None:
                # poco color de guante dentro de la pieza es de la pieza (un
                # relevador azul); la mano entra con el brazo desde afuera, o es
                # mucho (cámara cerrada, la mano tapa la pieza)
                sx, sy = 160 / frame.shape[1], 120 / frame.shape[0]
                roi = mask[max(int(base[1] * sy), 0):int(base[3] * sy) + 1,
                           max(int(base[0] * sx), 0):int(base[2] * sx) + 1]
                dentro = roi.sum() / 255.0 / mask.size
                roi[:] = 0
            hand = mask.mean() / 255.0 >= self.hand_min or dentro >= self.hand_inside_min
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        moving = False
        if self._prev is not None:
            moving = (cv2.absdiff(gray, self._prev) > 12).mean() >= self.motion_min
        self._prev = gray
        return hand, moving

    # ── máquina de estados ────────────────────────────────────────────────
    def update(self, frame: np.ndarray, detections: list, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self.last_closed = None
        self._base, self._parts = self._split(detections)
        self.hand, self.moving = self._signals(frame, self._base)
        present = self._base is not None

        if not present and not self.hand:
            if self._absent_since is None:
                self._absent_since = now
            if now - self._absent_since >= self.gone_s:
                self._close_piece()
                self.state = "vacio"
                self._still_since = None
            return
        self._absent_since = None

        if self.hand or self.moving or not present:
            self.state = "mano"
            self._still_since = None
            self._changed_since = None
            self._votes = []
            return

        if self.state == "resultado" and self.calibrated:
            # la pieza cambió sin que se notara movimiento (una parte delgada que
            # gira despacio): si lo que se ve ya no es el veredicto, reinspecciona
            if self.check(self._base, self._parts) != self.defects:
                if self._changed_since is None:
                    self._changed_since = now
                elif now - self._changed_since >= self.settle_s:
                    self.state, self._votes, self._window_t0 = "inspeccionando", [], now
                    self._changed_since = None
            else:
                self._changed_since = None

        if self.state in ("vacio", "mano"):
            self.state, self._still_since = "asentando", now
        if self.state == "asentando" and now - self._still_since >= self.settle_s:
            self.state, self._votes, self._window_t0 = "inspeccionando", [], now
        if self.state == "inspeccionando" and self.calibrated:
            d = self.check(self._base, self._parts)
            self._votes.append((not d, d))
            if now - self._window_t0 >= self.window_s and len(self._votes) >= 3:
                ok = Counter(v for v, _ in self._votes).most_common(1)[0][0]
                self.verdict = "OK" if ok else "NOT_OK"
                self.defects = frozenset() if ok else Counter(
                    d for v, d in self._votes if not v).most_common(1)[0][0]
                if not self._history:
                    self.piece_n += 1
                self._history.append(ok)
                self.state = "resultado"

    def _close_piece(self) -> None:
        if not self._history:
            return
        if not self._history[-1]:
            out = "rechazo"
        elif len(self._history) > 1 and not all(self._history):
            out = "corregida"
        else:
            out = "ok"
        self.counts[out] += 1
        self.last_closed = out
        self._history = []
        self.verdict = None
        self.defects = frozenset()

    # ── texto ─────────────────────────────────────────────────────────────
    def defects_text(self) -> str:
        return ", ".join(("falta " if d.startswith("falta:") else "") + d.split(":", 1)[1]
                         + (" fuera de posición" if d.startswith("pos:") else "")
                         for d in sorted(self.defects))

    def action_text(self) -> str:
        if self.verdict is None:
            return ""
        if self.verdict == "OK":
            return ("Corregida → sigue a la línea" if not all(self._history)
                    else "OK → sigue a la línea")
        txt = self.defects_text()
        txt = txt[0].upper() + txt[1:] if txt else "NOT OK"
        # lo que falta no se arregla en la estación; lo que está movido, sí
        if any(d.startswith("falta:") for d in self.defects):
            return f"{txt} → a RECHAZO"
        return f"{txt} → corregir y reinspeccionar"

    def summary(self) -> str:
        c = self.counts
        return f"OK {c['ok']} · Corregidas {c['corregida']} · Rechazo {c['rechazo']}"

    def state_text(self) -> str:
        if not self.calibrated:
            return "Sin referencia · coloca una pieza OK y pulsa Aprender"
        return {"vacio": "Esperando pieza",
                "mano": ("Mano en zona · inspección en pausa" if self.hand
                         else "Pieza en movimiento · en pausa"),
                "asentando": "Inspeccionando…",
                "inspeccionando": "Inspeccionando…",
                "resultado": ""}[self.state]

    # ── dibujo ────────────────────────────────────────────────────────────
    def _f(self, n):
        if n not in self._fonts:
            self._fonts[n] = _font(n)
        return self._fonts[n]

    def draw(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        s = w / 1280.0
        # con la pieza quieta: dónde DEBE ir cada parte (fantasma punteado) y
        # en rojo la parte que no cumple
        if self.state in ("inspeccionando", "resultado") and self._base is not None:
            bad = {d.split(":", 1)[1] for d in self.defects}
            for name, ref in self.reference.items():
                x1, y1, x2, y2 = _abs(ref, self._base)
                col = _ROJO if (self.state == "resultado" and name in bad) else (200, 200, 200)
                self._dashed(img, (x1, y1, x2, y2), col, max(2, int(2 * s)))
            for name in bad if self.state == "resultado" else ():
                box = self._parts.get(name)
                if box is not None:
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                  _ROJO, max(3, int(4 * s)))
        pills = []
        y = int(22 * s)
        if self.verdict and self.state != "inspeccionando":
            col = _VERDE if self.verdict == "OK" else _ROJO
            big = self.state == "resultado"
            pills.append((self.verdict.replace("_", " "), int((58 if big else 30) * s), col, (int(24 * s), y)))
            y += int((92 if big else 62) * s)
        st = self.state_text()
        if st:
            pills.append((st, int(24 * s), _GRIS, (int(24 * s), y)))
        txt = self.action_text()
        if txt and self.state != "inspeccionando":
            col = _VERDE if self.verdict == "OK" else (_ROJO if "RECHAZO" in txt else _AMBAR)
            pills.append((txt, int(30 * s), col, None))
        pills.append((self.summary(), int(24 * s), None, "tr"))
        self._pills(img, pills)

    @staticmethod
    def _dashed(img, box, col, th, dash=14):
        x1, y1, x2, y2 = box
        for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
            n = max(1, int(np.hypot(bx - ax, by - ay) // dash))
            for i in range(0, n, 2):
                p = (int(ax + (bx - ax) * i / n), int(ay + (by - ay) * i / n))
                q = (int(ax + (bx - ax) * min(i + 1, n) / n), int(ay + (by - ay) * min(i + 1, n) / n))
                cv2.line(img, p, q, col, th)

    def _pills(self, img, pills) -> None:
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return
        h, w = img.shape[:2]
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(im, "RGBA")
        for text, size, col, pos in pills:
            f = self._f(size)
            if f is None:
                continue
            l, t, r, b = d.textbbox((0, 0), text, font=f)
            tw, th = r - l, b - t
            px, py = int(size * .6), int(size * .3)
            if pos == "tr":
                x, y = w - tw - 2 * px - int(24 * w / 1280), int(24 * w / 1280)
                fondo, color = (20, 22, 26, 200), (255, 255, 255)
            elif pos is None:   # franja inferior centrada, texto en color
                x, y = (w - tw) // 2 - px, h - th - 2 * py - int(34 * w / 1280)
                fondo, color = (20, 22, 26, 205), (col[2], col[1], col[0])
            else:
                x, y = pos
                fondo, color = (col[2], col[1], col[0], 228), (255, 255, 255)
            d.rounded_rectangle([x, y, x + tw + 2 * px, y + th + 2 * py], radius=int(size * .4), fill=fondo)
            d.text((x + px - l, y + py - t), text, font=f, fill=color)
        img[:] = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    # ── contrato InspectionModule ─────────────────────────────────────────
    def process(self, frame, detections, annotated) -> ModuleVerdict:
        try:
            self.update(frame, detections)
            self.draw(annotated)
            return ModuleVerdict(ok=self.verdict != "NOT_OK", label=self.summary(),
                                 triggered_by=self.defects_text(),
                                 analyzed=1 if self._base is not None else 0,
                                 details={"state": self.state, "counts": dict(self.counts)})
        except Exception as e:   # nunca lanzar desde process()
            return ModuleVerdict(ok=True, label=f"ensamble: {e}")

    def config(self) -> dict:
        c = super().config()
        c.update(reference={k: list(v) for k, v in self.reference.items()},
                 tolerance=self.tolerance, glove=self.glove, hand_min=self.hand_min,
                 motion_min=self.motion_min, settle_s=self.settle_s,
                 window_s=self.window_s, gone_s=self.gone_s)
        return c

    def apply_config(self, cfg: dict) -> None:
        super().apply_config(cfg)
        ref = cfg.get("reference")
        if isinstance(ref, dict):
            try:
                self.reference = {str(k): [float(x) for x in v][:4] for k, v in ref.items()}
            except (TypeError, ValueError):
                pass
        if cfg.get("glove") in GLOVES:
            self.glove = cfg["glove"]
        for k in ("tolerance", "hand_min", "motion_min", "settle_s", "window_s", "gone_s"):
            if k in cfg:
                try:
                    setattr(self, k, float(cfg[k]))
                except (TypeError, ValueError):
                    pass
