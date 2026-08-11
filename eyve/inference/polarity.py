"""
polarity.py — Análisis de polaridad de ICs / Capacitores SMD
══════════════════════════════════════════════════════════════════════════════
Determina en qué lado (Top/Bottom/Left/Right → Q1-Q4) está el marcador
de polaridad de un componente.

Métodos disponibles:

  ● "stripe"  ★ RECOMENDADO para capacitores SMD con banda de cátodo
              Muestrea la oscuridad a lo largo del ARCO CIRCULAR del
              capacitor en 4 sectores de 90°. El sector más oscuro =
              donde está el stripe/crimp de cátodo.
              - Detecta automáticamente el círculo del cap (HoughCircles)
              - NO requiere cue
              - Inmune al fondo oscuro del pocket (solo muestrea el borde
                del disco del capacitor, no el fondo)
              - Inmune a texto/serigrafía (que está en el centro)
              Parámetro: arc_thickness (% del radio analizado, default=15%)

  ● "half"    Template matching del cue en cada mitad del IC.
              Para marcadores complejos. Requiere cue visual.

  ● "vector"  Template matching global → ángulo del vector.
              Para puntos compactos bien definidos.

  ● "dot"     HoughCircles por cuadrante. Para punto circular.

  ● "notch"   Densidad de bordes por cuadrante. Para muesca/bisel.

  ● "auto"    stripe + dot + notch con votación ponderada.

Cuadrantes (vista superior del IC):
    ┌───────────────────┐
    │   Q1   │   Q2    │  ← Top  (stripe en la mitad superior)
    │top-left│top-right│
    ├────────┼─────────┤
    │   Q4   │   Q3    │  ← Bottom (stripe en la mitad inferior)
    │bot-left│bot-right│
    └───────────────────┘

Uso rápido (capacitor SMD con stripe):
    from polarity import PolarityAnalyzer
    analyzer = PolarityAnalyzer(method="stripe")
    result   = analyzer.analyze(ic_roi)
    print(result.side, result.quadrant, result.confidence)
    # "top"  1  0.87
"""

from __future__ import annotations
import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────────────────────
LEARN_SAMPLES          = 6
LEARN_MIN_CONF         = 0.45
ARC_THICKNESS_DEF      = 0.15   # fracción del radio a muestrear en el borde
ARC_N_SAMPLES          = 40     # muestras angulares por sector
ARC_DEPTH_SAMPLES      = 6      # muestras radiales en cada punto del arco
STRIPE_MIN_DOMINANCE   = 1.12   # ratio mínimo oscuro_ganador / media_otros (modo quad)
STRIPE_AXIS_MIN_RATIO  = 1.04   # ratio mínimo top/bottom en modo "vertical"
# Modo de comparación del stripe:
#   "vertical" → compara top vs bottom solamente (recomendado para caps SMD en cinta)
#                Tolerante a rotación del cap en el pocket (hasta ±45°)
#   "quad"     → max de 4 sectores (más estricto, caps que pueden llegar a 90°)
STRIPE_AXIS_DEFAULT    = "vertical"


# ─────────────────────────────────────────────────────────────────────────────
#  Resultado
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PolarityResult:
    quadrant:    Optional[int]   = None
    confidence:  float           = 0.0
    method:      str             = ""
    position:    str             = ""    # "top-left" | ...
    side:        str             = ""    # "top" | "bottom" | "left" | "right"
    angle_deg:   Optional[float] = None
    concordance: float           = 0.0
    side_scores: Optional[Dict[str, float]] = field(default=None, repr=False)
    debug_img:   Optional[np.ndarray]       = field(default=None, repr=False)
    expected_quadrant: Optional[int]        = None

    @property
    def is_correct(self) -> Optional[bool]:
        if self.expected_quadrant is None or self.quadrant is None:
            return None
        # Para método stripe: comparar por LADO, no por cuadrante exacto.
        # Un stripe en "top" es correcto si ref es Q1 o Q2 (ambos son "top").
        # Esto elimina el ruido Q1/Q2 para bandas horizontales.
        if self.method == "stripe" and self.side:
            return _side_matches(self.side, self.expected_quadrant)
        return self.quadrant == self.expected_quadrant

    @property
    def quadrant_label(self) -> str:
        return {1:"Q1 top-left", 2:"Q2 top-right",
                3:"Q3 bottom-right", 4:"Q4 bottom-left"}.get(self.quadrant or 0, "?")

    @property
    def quadrant_short(self) -> str:
        return {1:"↖ top-left", 2:"↗ top-right",
                3:"↘ bot-right", 4:"↙ bot-left"}.get(self.quadrant or 0, "?")


# ─────────────────────────────────────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _gray(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None or img.size == 0:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()


def _clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _find_cap_circle(gray: np.ndarray) -> Tuple[int, int, int]:
    """
    Detecta el círculo del capacitor en el ROI.
    Retorna (cx, cy, r). Fallback: centroide de la imagen.
    """
    h, w  = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.0, minDist=min(h, w) * 0.5,
        param1=80, param2=22,
        minRadius=int(min(h, w) * 0.28),
        maxRadius=int(min(h, w) * 0.58))
    if circles is not None:
        c = circles[0, 0]
        return int(c[0]), int(c[1]), int(c[2])
    return w // 2, h // 2, int(min(w, h) * 0.42)


def _side_to_quadrant(side: str, cx: float, cy: float,
                       iw: float, ih: float) -> int:
    """
    Convierte (side + posición del círculo) a cuadrante.
    Para método stripe/arc: cx,cy son el CENTRO DEL CÍRCULO detectado,
    no el centroide del match. Esto da subdivisión estable left/right.
    """
    if side == "top":
        return 1 if cx <= iw / 2.0 else 2
    if side == "bottom":
        return 4 if cx <= iw / 2.0 else 3
    if side == "left":
        return 1 if cy <= ih / 2.0 else 4
    if side == "right":
        return 2 if cy <= ih / 2.0 else 3
    return 1


def _side_from_quadrant(q: int) -> str:
    return {1: "top", 2: "top", 3: "bottom", 4: "bottom"}.get(q, "top")


def _side_matches(detected_side: str, expected_quadrant: int) -> bool:
    """
    Para método stripe: comparar solo por LADO (top/bottom/left/right),
    ignorando la subdivisión Q1/Q2 o Q3/Q4 que es ruidosa.
    """
    expected_side = _side_from_quadrant(expected_quadrant)
    return detected_side == expected_side


# ─────────────────────────────────────────────────────────────────────────────
#  ── Método STRIPE (ARC-BASED) ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class _StripeArcDetector:
    """
    Muestrea la oscuridad en 4 arcos de 90° a lo largo del borde circular
    del capacitor. El arco más oscuro = donde está la banda de cátodo.

    Convención angular (cv2/numpy): 0°=derecha, 90°=abajo, 180°=izquierda,
    270°=arriba (sentido antihorario en imagen).

    Sectores:
      top:    270° a 360° + 0° a 90°  → arco en la parte SUPERIOR
      right:   0° a 90°  +  270°-360° NO, corrección:
      
    Corrección a convención de imagen (0°=3 en punto, 90°=6 en punto):
      top     → ángulos de 225° a 315°  (centrado en 270° = 12 en punto)
      right   → ángulos de 315° a 45°   (centrado en   0° =  3 en punto)
      bottom  → ángulos de  45° a 135°  (centrado en  90° =  6 en punto)
      left    → ángulos de 135° a 225°  (centrado en 180° =  9 en punto)
    """

    def __init__(self, arc_thickness: float = ARC_THICKNESS_DEF,
                 stripe_axis: str = STRIPE_AXIS_DEFAULT):
        self.arc_thickness = arc_thickness
        self.stripe_axis   = stripe_axis   # "vertical" o "quad"

    def _arc_score(self, dark_e: np.ndarray, cx: int, cy: int, r: int,
                   a_start: float, a_end: float) -> float:
        """Oscuridad promedio en el arco a_start..a_end (en grados cv2)."""
        h, w = dark_e.shape
        th   = max(0.05, min(0.40, self.arc_thickness))

        # Generar ángulos (manejando cruce de 360°)
        if a_start > a_end:
            # Cruza 360° (ej: 315→45)
            n1 = ARC_N_SAMPLES * 2 // 3
            n2 = ARC_N_SAMPLES - n1
            angles = list(np.linspace(a_start, 360, n1, endpoint=False)) + \
                     list(np.linspace(0, a_end, n2))
        else:
            angles = np.linspace(a_start, a_end, ARC_N_SAMPLES)

        vals = []
        for deg in angles:
            rad = math.radians(deg)
            for frac in np.linspace(1.0 - th, 1.0, ARC_DEPTH_SAMPLES):
                px = int(cx + r * frac * math.cos(rad))
                py = int(cy + r * frac * math.sin(rad))
                if 0 <= px < w and 0 <= py < h:
                    vals.append(float(dark_e[py, px]))
        return float(np.mean(vals)) if vals else 0.0

    def detect(self, gray: np.ndarray) -> Tuple[Optional[str], Optional[int],
                                                  float, Dict[str, float],
                                                  Tuple[int, int, int]]:
        """
        Retorna (side, quadrant, confidence, scores_dict, circle).
        """
        if gray is None or gray.size == 0:
            return None, None, 0.0, {}, (0, 0, 0)

        h, w   = gray.shape
        dark   = cv2.bitwise_not(gray)
        dark_e = _clahe(dark)

        cx, cy, r = _find_cap_circle(gray)

        scores = {
            "top":    self._arc_score(dark_e, cx, cy, r, 225, 315),
            "right":  self._arc_score(dark_e, cx, cy, r, 315, 405),  # cruza 360
            "bottom": self._arc_score(dark_e, cx, cy, r,  45, 135),
            "left":   self._arc_score(dark_e, cx, cy, r, 135, 225),
        }
        # Fix para "right" (315→360 + 0→45)
        scores["right"] = self._arc_score(dark_e, cx, cy, r, 315, 360) * 0.5 + \
                          self._arc_score(dark_e, cx, cy, r,   0,  45) * 0.5

        if all(v < 0.5 for v in scores.values()):
            return None, None, 0.0, scores, (cx, cy, r)

        if self.stripe_axis == "vertical":
            # ── Modo eje vertical: solo compara top vs bottom ────────────
            # Correcto para caps SMD en cinta (solo pueden llegar arriba/abajo,
            # nunca a 90°). Tolerante a rotacion del cap en el pocket hasta ±45°.
            t_sc = scores["top"]; b_sc = scores["bottom"]
            winner = "top" if t_sc >= b_sc else "bottom"
            loser  = "bottom" if winner == "top" else "top"
            dom    = scores[winner] / (scores[loser] + 1e-6)
            if dom < STRIPE_AXIS_MIN_RATIO:
                return None, None, 0.0, scores, (cx, cy, r)
            # Confianza: 0% en dom=1.04, 100% en dom≥1.8
            raw_conf = min(1.0, (dom - 1.0) / 0.8)
        else:
            # ── Modo quad: max de 4 sectores (comportamiento original) ────
            winner = max(scores, key=scores.get)
            others = [v for k, v in scores.items() if k != winner]
            avg_o  = sum(others) / 3 if others else 1e-6
            dom    = scores[winner] / (avg_o + 1e-6)
            if dom < STRIPE_MIN_DOMINANCE:
                return None, None, 0.0, scores, (cx, cy, r)
            raw_conf = min(1.0, (dom - 1.0) / 1.5)

        quad = _side_to_quadrant(winner, cx, cy, w, h)
        return winner, quad, round(raw_conf, 3), scores, (cx, cy, r)


# ─────────────────────────────────────────────────────────────────────────────
#  ── Método HALF ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class _HalfDetector:
    def __init__(self):
        self._templates: List[np.ndarray] = []
        self._cue_valid = False

    def set_cue(self, cue_crop: np.ndarray, ic_roi: np.ndarray):
        gray_ic  = _gray(ic_roi)
        gray_cue = _gray(cue_crop)
        if gray_ic is None or gray_cue is None:
            return
        ih, iw = gray_ic.shape
        ch, cw = gray_cue.shape
        if ch >= ih * 0.80 or cw >= iw * 0.80:
            print(f"[polarity] Cue demasiado grande ({cw}×{ch} vs IC {iw}×{ih}). Usa método stripe.")
            self._cue_valid = False; return
        ec = _clahe(gray_cue)
        self._templates.clear()
        for scale in [0.60, 0.75, 1.00, 1.25, 1.50]:
            th2 = max(4, int(ch * scale)); tw2 = max(4, int(cw * scale))
            if th2 < ih * 0.80 and tw2 < iw * 0.80:
                self._templates.append(
                    cv2.resize(ec, (tw2, th2), interpolation=cv2.INTER_AREA))
        self._cue_valid = bool(self._templates)

    def has_cue(self) -> bool:
        return self._cue_valid

    def detect(self, gray: np.ndarray) -> Tuple[Optional[int], float, float]:
        if not self._cue_valid or not self._templates or gray is None:
            return None, 0.0, 0.0
        enhanced = _clahe(gray); ih, iw = enhanced.shape
        best_score = -1.0; best_cx = iw/2.0; best_cy = ih/2.0
        for tpl in self._templates:
            th, tw = tpl.shape
            if th >= ih or tw >= iw: continue
            result = cv2.matchTemplate(enhanced, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(result)
            if mx > best_score:
                best_score = mx; best_cx = loc[0]+tw/2.0; best_cy = loc[1]+th/2.0
        if best_score < 0.20: return None, 0.0, max(0.0, best_score)
        quad = _side_to_quadrant(
            "top" if best_cy < ih/2 else "bottom",
            best_cx, best_cy, iw, ih)
        dx = abs(best_cx-iw/2)/(iw/2); dy = abs(best_cy-ih/2)/(ih/2)
        conf = 0.70*min(1.0,best_score) + 0.30*min(1.0,(dx+dy)/2*1.5)
        return quad, round(conf, 3), round(float(best_score), 3)


# ─────────────────────────────────────────────────────────────────────────────
#  ── Método VECTOR ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class _VectorDetector:
    def __init__(self):
        self._templates: List[np.ndarray] = []
        self.concordance_threshold = 0.25
        self._cue_valid = False

    def set_cue(self, cue_crop: np.ndarray, ic_roi: Optional[np.ndarray] = None):
        gray = _gray(cue_crop)
        if gray is None: return
        ic_gray = _gray(ic_roi) if ic_roi is not None else gray
        ih, iw  = ic_gray.shape; ch, cw = gray.shape
        if ch >= ih*0.85 or cw >= iw*0.85: self._cue_valid=False; return
        ec = _clahe(gray); self._templates.clear()
        for scale in [0.6, 0.75, 1.0, 1.25, 1.5]:
            th2=max(4,int(ch*scale)); tw2=max(4,int(cw*scale))
            if th2<ih*0.85 and tw2<iw*0.85:
                self._templates.append(cv2.resize(ec,(tw2,th2),interpolation=cv2.INTER_AREA))
        self._cue_valid = bool(self._templates)

    def has_cue(self) -> bool: return self._cue_valid

    def detect(self, gray: np.ndarray) -> Tuple[Optional[int], float, float, Optional[float]]:
        if not self._cue_valid or not self._templates or gray is None:
            return None, 0.0, 0.0, None
        enhanced=_clahe(gray); ih,iw=enhanced.shape
        best=-1.0; bcx=iw/2.0; bcy=ih/2.0
        for tpl in self._templates:
            th,tw=tpl.shape
            if th>=ih or tw>=iw: continue
            result=cv2.matchTemplate(enhanced,tpl,cv2.TM_CCOEFF_NORMED)
            _,mx,_,loc=cv2.minMaxLoc(result)
            if mx>best: best=mx; bcx=loc[0]+tw/2.0; bcy=loc[1]+th/2.0
        if best<self.concordance_threshold: return None,0.0,max(0.0,best),None
        vx=bcx-iw/2.0; vy=bcy-ih/2.0
        angle=math.degrees(math.atan2(vx,-vy))%360
        if angle<45 or angle>=315:   side="top"
        elif 45<=angle<135:          side="right"
        elif 135<=angle<225:         side="bottom"
        else:                        side="left"
        quad=_side_to_quadrant(side,bcx,bcy,iw,ih)
        sc={1:0,2:90,3:180,4:270}[quad]
        diff=min(abs(angle-sc),360-abs(angle-sc))
        conf=0.60*min(1.0,best)+0.40*max(0.0,1.0-diff/45.0)
        return quad,round(conf,3),round(best,3),round(angle,1)


# ─────────────────────────────────────────────────────────────────────────────
#  ── Método DOT ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class _DotDetector:
    def detect(self, gray: np.ndarray) -> Tuple[Optional[int], float]:
        enhanced=_clahe(gray); h,w=enhanced.shape; mh,mw=h//2,w//2
        rois={1:enhanced[:mh,:mw],2:enhanced[:mh,mw:],3:enhanced[mh:,mw:],4:enhanced[mh:,:mw]}
        scores: Dict[int,float]={}
        for q,roi in rois.items():
            if roi.size==0: scores[q]=0.0; continue
            rh,rw=roi.shape; min_r=max(2,min(rh,rw)//12); max_r=max(min_r+1,min(rh,rw)//4)
            best=0.0
            for inv in [roi,cv2.bitwise_not(roi)]:
                circles=cv2.HoughCircles(inv,cv2.HOUGH_GRADIENT,dp=1.2,
                                          minDist=min_r*2,param1=60,param2=15,
                                          minRadius=min_r,maxRadius=max_r)
                if circles is not None: best=max(best,min(1.0,float(circles[0,0,2])/max_r))
            scores[q]=best
        if all(v==0 for v in scores.values()): return None,0.0
        bq=max(scores,key=scores.get); bv=scores[bq]
        av=sum(v for k,v in scores.items() if k!=bq)/3
        if bv<0.05 or bv<av*1.3: return None,0.0
        return bq,round(bv,3)


# ─────────────────────────────────────────────────────────────────────────────
#  ── Método NOTCH ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class _NotchDetector:
    def detect(self, gray: np.ndarray) -> Tuple[Optional[int], float]:
        enhanced=_clahe(gray); edges=cv2.Canny(enhanced,35,110)
        h,w=edges.shape; mh,mw=h//2,w//2
        rois={1:edges[:mh,:mw],2:edges[:mh,mw:],3:edges[mh:,mw:],4:edges[mh:,:mw]}
        scores={q:float(np.mean(r)) if r.size>0 else 0.0 for q,r in rois.items()}
        if all(v<5 for v in scores.values()): return None,0.0
        bq=max(scores,key=scores.get); bv=scores[bq]
        av=sum(v for k,v in scores.items() if k!=bq)/3
        if bv<av*1.4: return None,0.0
        return bq,round(min(1.0,(bv-av)/(av+1e-6)),3)


# ─────────────────────────────────────────────────────────────────────────────
#  Clase principal
# ─────────────────────────────────────────────────────────────────────────────
class PolarityAnalyzer:
    """
    Analiza la polaridad de un IC en un ROI BGR o GRAY.

    Para capacitores SMD con banda de cátodo (sin cue):
        analyzer = PolarityAnalyzer(method="stripe")
        result   = analyzer.analyze(ic_roi)

    Para componentes con marcador visible (con cue):
        analyzer = PolarityAnalyzer(method="half")
        analyzer.set_marker_cue(cue_crop, ic_roi)
        result   = analyzer.analyze(ic_roi)
    """

    METHODS = ("stripe", "half", "vector", "dot", "notch", "auto")

    def __init__(self, method: str = "stripe",
                 arc_thickness: float = ARC_THICKNESS_DEF):
        if method not in self.METHODS: method = "stripe"
        self.method        = method
        self.arc_thickness = arc_thickness

        self._stripe = _StripeArcDetector(arc_thickness)
        self._half   = _HalfDetector()
        self._vector = _VectorDetector()
        self._dot    = _DotDetector()
        self._notch  = _NotchDetector()

        self._ref_samples: List[int] = []
        self._reference:  Optional[int] = None
        self._learning    = True

        self.concordance_threshold = 0.25

    # ── API pública ───────────────────────────────────────────────────────

    def set_method(self, method: str):
        if method in self.METHODS: self.method = method

    def set_arc_thickness(self, value: float):
        """Fracción del radio del cap a muestrear en el arco (0.08 – 0.35)."""
        value = max(0.08, min(0.35, value))
        self.arc_thickness = value
        self._stripe.arc_thickness = value

    def set_stripe_axis(self, axis: str):
        """Modo de comparación del stripe: 'vertical' (top vs bottom) o 'quad' (4 sectores).
        'vertical' es el recomendado para caps SMD en cinta — tolerante a rotación en pocket.
        """
        if axis in ("vertical", "quad"):
            self._stripe.stripe_axis = axis

    def set_concordance_threshold(self, value: float):
        value = max(0.05, min(0.95, value))
        self.concordance_threshold = value
        self._vector.concordance_threshold = value

    def set_marker_cue(self, cue_crop: np.ndarray,
                        ic_roi: Optional[np.ndarray] = None):
        ref = ic_roi if ic_roi is not None else cue_crop
        self._half.set_cue(cue_crop, ref)
        self._vector.set_cue(cue_crop, ref)
        self._ref_samples.clear()

    def set_reference(self, quadrant: int):
        if quadrant in (1, 2, 3, 4):
            self._reference = quadrant
            self._learning  = False

    def reset_reference(self):
        self._ref_samples.clear()
        self._reference  = None
        self._learning   = True

    def analyze(self, ic_roi: np.ndarray) -> PolarityResult:
        if ic_roi is None or ic_roi.size == 0:
            return PolarityResult(expected_quadrant=self._reference)
        gray = _gray(ic_roi)
        if gray is None:
            return PolarityResult(expected_quadrant=self._reference)

        ih, iw  = gray.shape
        weights = self._weights()
        vote:   Dict[int, float] = {1:0.0, 2:0.0, 3:0.0, 4:0.0}

        stripe_side   = None; stripe_scores: Dict[str,float] = {}
        stripe_circle = (iw//2, ih//2, int(min(iw,ih)*0.42))
        all_results   = []   # (method, quad, conf, conc)
        vec_angle     = None

        # ── STRIPE ────────────────────────────────────────────────────────
        if weights.get("stripe", 0) > 0:
            side, q, conf, scores, circle = self._stripe.detect(gray)
            stripe_side   = side
            stripe_scores = scores
            stripe_circle = circle
            if q:
                vote[q] += conf * weights["stripe"]
                all_results.append(("stripe", q, conf, conf))

        # ── HALF ──────────────────────────────────────────────────────────
        if weights.get("half", 0) > 0:
            q, conf, conc = self._half.detect(gray)
            if q:
                vote[q] += conf * weights["half"]
                all_results.append(("half", q, conf, conc))

        # ── VECTOR ────────────────────────────────────────────────────────
        if weights.get("vector", 0) > 0:
            q, conf, conc, angle = self._vector.detect(gray)
            if q:
                vec_angle = angle
                vote[q] += conf * weights["vector"]
                all_results.append(("vector", q, conf, conc))

        # ── DOT ───────────────────────────────────────────────────────────
        if weights.get("dot", 0) > 0:
            q, conf = self._dot.detect(gray)
            if q:
                vote[q] += conf * weights["dot"]
                all_results.append(("dot", q, conf, 0.0))

        # ── NOTCH ─────────────────────────────────────────────────────────
        if weights.get("notch", 0) > 0:
            q, conf = self._notch.detect(gray)
            if q:
                vote[q] += conf * weights["notch"]
                all_results.append(("notch", q, conf, 0.0))

        if not all_results:
            return PolarityResult(
                expected_quadrant=self._reference,
                side_scores=stripe_scores or None)

        # ── Fusión ────────────────────────────────────────────────────────
        best_q     = max(vote, key=vote.get)
        total_v    = sum(vote.values())
        confidence = vote[best_q] / total_v if total_v > 0 else 0.0

        winner_results = [(m,q,c,conc) for m,q,c,conc in all_results if q==best_q]
        dom = max(winner_results, key=lambda x: x[2], default=("",best_q,0,0))
        dominant_meth = dom[0]; dom_conc = dom[3]

        # ── Auto-aprendizaje ──────────────────────────────────────────────
        # Para stripe: guardar el LADO (top/bottom/left/right) como referencia
        # en lugar del cuadrante exacto, para evitar ruido Q1/Q2.
        if self._learning and confidence >= LEARN_MIN_CONF:
            if self.method == "stripe" and stripe_side:
                # Convertir side a quadrant canónico: top→Q1, bottom→Q3, left→Q4, right→Q2
                canonical = {"top": 1, "bottom": 3, "left": 4, "right": 2}
                self._ref_samples.append(canonical.get(stripe_side, best_q))
            else:
                self._ref_samples.append(best_q)
            if len(self._ref_samples) >= LEARN_SAMPLES:
                self._reference = Counter(self._ref_samples).most_common(1)[0][0]
                self._learning  = False

        pos_map = {1:"top-left",2:"top-right",3:"bottom-right",4:"bottom-left"}
        side_out = stripe_side or {1:"top",2:"top",3:"bottom",4:"bottom"}.get(best_q,"")

        result = PolarityResult(
            quadrant=best_q,
            confidence=round(min(1.0,confidence),3),
            method=dominant_meth,
            position=pos_map.get(best_q,""),
            side=side_out,
            angle_deg=vec_angle,
            concordance=round(min(1.0,dom_conc),3),
            side_scores=stripe_scores or None,
            expected_quadrant=self._reference,
        )
        result.debug_img = self._draw_debug(
            ic_roi, result, vote, vec_angle,
            stripe_scores, stripe_circle, iw, ih)
        return result

    # ── Pesos ─────────────────────────────────────────────────────────────

    def _weights(self) -> Dict[str, float]:
        hc = self._half.has_cue(); vc = self._vector.has_cue()
        if self.method == "stripe":
            return {"stripe":2.0,"half":0.0,"vector":0.0,"dot":0.0,"notch":0.0}
        if self.method == "half":
            if hc:  return {"stripe":0.4,"half":2.0,"vector":0.0,"dot":0.0,"notch":0.0}
            return {"stripe":1.5,"half":0.0,"vector":0.0,"dot":0.5,"notch":0.5}
        if self.method == "vector":
            if vc:  return {"stripe":0.3,"half":0.0,"vector":2.0,"dot":0.0,"notch":0.0}
            return {"stripe":1.5,"half":0.0,"vector":0.0,"dot":0.5,"notch":0.5}
        if self.method == "dot":
            return {"stripe":0.3,"half":0.0,"vector":0.0,"dot":2.0,"notch":0.5}
        if self.method == "notch":
            return {"stripe":0.3,"half":0.0,"vector":0.0,"dot":0.5,"notch":2.0}
        # auto
        return {"stripe":1.5,"half":0.8 if hc else 0.0,
                "vector":0.5 if vc else 0.0,"dot":0.5,"notch":0.5}

    # ── Debug image ───────────────────────────────────────────────────────

    def _draw_debug(self, roi: np.ndarray, r: PolarityResult,
                    vote: Dict, angle: Optional[float],
                    stripe_scores: Dict, circle: Tuple,
                    iw: int, ih: int) -> np.ndarray:
        h, w  = roi.shape[:2]
        debug = roi.copy()
        mh, mw = h // 2, w // 2
        cx, cy, rc = circle

        qcol  = {1:(0,220,80),2:(0,190,255),3:(255,140,0),4:(180,80,255)}
        qrect = {1:(0,0,mw,mh),2:(mw,0,w,mh),3:(mw,mh,w,h),4:(0,mh,mw,h)}

        # Colorear cuadrantes
        overlay = debug.copy()
        for q,(x1,y1,x2,y2) in qrect.items():
            alpha = 0.35 if q==r.quadrant else 0.05
            cv2.rectangle(overlay,(x1,y1),(x2-1,y2-1),qcol[q],-1)
            cv2.addWeighted(overlay,alpha,debug,1-alpha,0,debug)
            overlay = debug.copy()

        cv2.line(debug,(mw,0),(mw,h),(50,50,50),1)
        cv2.line(debug,(0,mh),(w,mh),(50,50,50),1)

        # ── Dibujar arcos muestreados (puntos de color) ────────────────
        arc_sectors = {
            "top":    (225,315),
            "bottom": (45, 135),
            "left":   (135,225),
        }
        col_side = {"top":(0,220,80),"bottom":(50,50,220),"left":(200,80,255),"right":(0,190,255)}
        th = max(0.05, min(0.40, self.arc_thickness))

        def draw_arc_dots(a_start, a_end, side):
            col  = col_side[side]
            bright = (side == r.side)
            if a_start > a_end:
                n1 = ARC_N_SAMPLES*2//3; n2 = ARC_N_SAMPLES - n1
                angles = list(np.linspace(a_start,360,n1,endpoint=False)) + \
                         list(np.linspace(0,a_end,n2))
            else:
                angles = np.linspace(a_start,a_end,ARC_N_SAMPLES)
            for deg in angles:
                rad_a = math.radians(deg)
                for frac in np.linspace(1.0-th, 1.0, ARC_DEPTH_SAMPLES):
                    px = int(cx + rc*frac*math.cos(rad_a))
                    py = int(cy + rc*frac*math.sin(rad_a))
                    if 0<=px<w and 0<=py<h:
                        cv2.circle(debug,(px,py),1 if bright else 1,col,-1)

        for side,(as_,ae_) in arc_sectors.items():
            draw_arc_dots(as_,ae_,side)
        # right (cross 0°)
        draw_arc_dots(315,360,"right")
        draw_arc_dots(0,45,"right")

        # Círculo detectado del cap
        cv2.circle(debug,(cx,cy),rc,(80,80,80),1)
        cv2.circle(debug,(cx,cy),2,(140,140,140),-1)

        # Flecha vector si aplica
        if angle is not None:
            ln  = min(mw,mh)*0.55
            rad = math.radians(angle)
            ex  = int(cx + ln*math.sin(rad)); ey = int(cy - ln*math.cos(rad))
            col = qcol.get(r.quadrant,(0,220,80))
            cv2.arrowedLine(debug,(cx,cy),(max(2,min(w-2,ex)),max(2,min(h-2,ey))),
                            col,2,tipLength=0.28,line_type=cv2.LINE_AA)

        # ── Panel inferior ────────────────────────────────────────────────
        ph    = max(52, h // 3)
        panel = np.zeros((ph, w, 3), dtype=np.uint8)

        # Barras de score de arcos
        if stripe_scores and max(stripe_scores.values(), default=0) > 0:
            mx_s = max(stripe_scores.values())
            sides_order = ["top","right","bottom","left"]
            bw = w // 4
            for i, sl in enumerate(sides_order):
                sc   = stripe_scores.get(sl, 0) / (mx_s + 1e-6)
                blen = int(sc * (ph - 20))
                bx   = i * bw + 2
                col  = col_side[sl]
                bright = (sl == r.side)
                cv2.rectangle(panel,(bx,ph-20-blen),(bx+bw-4,ph-20),
                              col if bright else tuple(c//3 for c in col),-1)
                cv2.putText(panel,sl[0].upper(),(bx+2,ph-22),
                            cv2.FONT_HERSHEY_SIMPLEX,0.26,col,1)
        else:
            mx_v = max(vote.values()) if vote else 1
            bw   = w // 4
            for qi,col_q in qcol.items():
                v    = vote.get(qi,0)/(mx_v+1e-6)
                blen = int(v*(ph-18))
                bx   = (qi-1)*bw+2
                cv2.rectangle(panel,(bx,ph-18-blen),(bx+bw-4,ph-18),col_q,-1)
                cv2.putText(panel,f"Q{qi}",(bx,ph-20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.24,col_q,1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        ok_c = ((0,200,80) if r.is_correct else
                (80,80,220) if r.is_correct is False else (130,130,130))
        state = ("CORRECTO"    if r.is_correct else
                 "ERROR"       if r.is_correct is False else
                 f"aprend {len(self._ref_samples)}/{LEARN_SAMPLES}"
                 if self._learning else "Sin ref.")
        angle_s = f"  {angle:.1f}°" if angle else ""

        cv2.putText(panel,f"Q{r.quadrant} {r.position}",(4,11),font,0.37,(200,220,200),1,cv2.LINE_AA)
        cv2.putText(panel,f"{r.method} [{r.confidence:.0%}]{angle_s}",(4,21),font,0.32,(140,160,140),1,cv2.LINE_AA)
        cv2.putText(panel,f"REF Q{r.expected_quadrant} → {state}"
                    if r.expected_quadrant else state,
                    (4,31),font,0.33,ok_c,1,cv2.LINE_AA)

        return np.vstack([debug, panel])

    # ── Overlay en frame principal ─────────────────────────────────────────
    @staticmethod
    def draw_on_frame(frame: np.ndarray,
                      bbox: Tuple[int,int,int,int],
                      result: "PolarityResult") -> np.ndarray:
        x1,y1,x2,y2 = bbox
        cx,cy = (x1+x2)//2,(y1+y2)//2
        if result.quadrant is None:
            cv2.putText(frame,"POL ??",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.36,(100,100,100),1)
            return frame
        col_ok=(0,200,80); col_err=(50,50,210); col_unk=(130,130,0)
        col = col_ok if result.is_correct else col_err if result.is_correct is False else col_unk
        side_lines = {"top":((x1,y1),(x2,y1)),"bottom":((x1,y2),(x2,y2)),
                      "left":((x1,y1),(x1,y2)),"right":((x2,y1),(x2,y2))}
        if result.side in side_lines:
            p1,p2 = side_lines[result.side]; cv2.line(frame,p1,p2,col,3)
        if result.angle_deg is not None:
            ln  = max(10,min((x2-x1),(y2-y1))//2*0.55)
            rad = math.radians(result.angle_deg)
            ex  = int(cx+ln*math.sin(rad)); ey = int(cy-ln*math.cos(rad))
            cv2.arrowedLine(frame,(cx,cy),(ex,ey),col,2,tipLength=0.3,line_type=cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),3,col,-1)
        lbl = f"Q{result.quadrant} {result.side} [{result.confidence:.0%}]"
        if result.is_correct is False:
            cv2.rectangle(frame,(x1,y1),(x2,y2),col_err,3); lbl += " ⚠"
        cv2.putText(frame,lbl,(x1,y2+14),cv2.FONT_HERSHEY_SIMPLEX,0.36,col,1,cv2.LINE_AA)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Demo standalone
# ─────────────────────────────────────────────────────────────────────────────
def _demo():
    import sys, os
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF if os.name=="nt" else cv2.CAP_ANY)
    if not cap.isOpened(): print("❌ Sin cámara"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    analyzer = PolarityAnalyzer(method="stripe")
    methods  = list(PolarityAnalyzer.METHODS); m_idx = 0
    win = "polarity.py  S=cue  R=reset  M=metodo  +/-=thickness  Q=salir"
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret: break
        h,w = frame.shape[:2]; mg=int(min(w,h)*0.15)
        box=(mg,mg,w-mg,h-mg); x1,y1,x2,y2=box
        roi=frame[y1:y2,x1:x2]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(60,60,60),1)
        cv2.line(frame,((x1+x2)//2,y1),((x1+x2)//2,y2),(35,35,35),1)
        cv2.line(frame,(x1,(y1+y2)//2),(x2,(y1+y2)//2),(35,35,35),1)
        result=analyzer.analyze(roi)
        PolarityAnalyzer.draw_on_frame(frame,box,result)
        if result.debug_img is not None:
            dh,dw=result.debug_img.shape[:2]; sc=190/max(dw,dh)
            small=cv2.resize(result.debug_img,(int(dw*sc),int(dh*sc)))
            dsh,dsw=small.shape[:2]; frame[8:8+dsh,w-dsw-8:w-8]=small
        for i,ln in enumerate([f"Método: {analyzer.method}",
                                f"Arc: {analyzer.arc_thickness:.0%} (+/-)",
                                f"Ref: Q{analyzer._reference}" if analyzer._reference
                                else f"Aprend: {len(analyzer._ref_samples)}/{LEARN_SAMPLES}"]):
            cv2.putText(frame,ln,(8,20+i*16),cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,200,80),1)
        cv2.imshow(win,frame)
        k=cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        elif k==ord('s'): analyzer.set_marker_cue(roi,roi); print("Cue set")
        elif k==ord('r'): analyzer.reset_reference(); print("Reset")
        elif k==ord('m'):
            m_idx=(m_idx+1)%len(methods); analyzer.set_method(methods[m_idx])
        elif k==ord('+'):
            analyzer.set_arc_thickness(analyzer.arc_thickness+0.02)
        elif k==ord('-'):
            analyzer.set_arc_thickness(analyzer.arc_thickness-0.02)
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
