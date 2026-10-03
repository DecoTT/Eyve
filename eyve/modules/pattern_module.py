"""
Módulo Patrón — inspección SIN clases, contra lo que el material tiene de bueno.

Por qué existe
──────────────
Detectar con YOLO obliga a enumerar los defectos, y los defectos son
infinitos.  Funciona bien para cosas puntuales y nombrables —un rayón, una
mancha, una pieza que hay que contar— porque ahí hace falta saber QUÉ es.
Para fallos de impresión no: fantasma, offset, falta de tinta, y los diez
que nadie listó, son variantes sin fin de lo mismo — "esto no se parece a
lo bueno".

Este módulo responde esa otra pregunta.  No se entrena, no se etiqueta, y
encuentra cosas que nunca vio.  A cambio dice DÓNDE está mal, no QUÉ es.
Los dos enfoques son complementarios y corren a la vez:

    YOLO      → "rayón", "mancha"           nombra lo que le enseñaste
    Patrón    → "aquí algo no cuadra"       descubre lo que nunca vio

Cómo funciona (método "periodo")
────────────────────────────────
El material se repite —estampado, malla, extrusión, azulejo, tejido— así
que **el material es su propia referencia**.  Para cada píxel se compara su
vecindad contra la misma vecindad desplazada un periodo, en varias
direcciones, y se toma la MENOR diferencia:

    un píxel normal   se parece a su repetición en alguna dirección → bajo
    un defecto        no se parece a ninguna                        → alto

Se usa el mínimo y no el promedio a propósito: cerca de una orilla, o donde
el estampado se interrumpe por diseño, basta con que coincida en una
dirección para no marcarlo.

El periodo se estima solo por autocorrelación, y se puede fijar a mano
cuando el material lo tiene impreso en la ficha técnica.

Limitaciones, dichas de frente:
  - necesita que el material se repita.  Para una pieza única que siempre
    va en la misma posición, el método correcto es una muestra patrón
    (golden sample), que es otro método de este mismo módulo.
  - marca cualquier desviación, incluida una que no sea un defecto (una
    costura, una orilla, una etiqueta cosida).  Por eso la sensibilidad es
    un control de la UI y no una constante.
  - el umbral se apoya en que la MAYOR PARTE del encuadre sea material
    sano.  Medido: con un defecto o dos se comporta bien (cero falsas
    alarmas en 64 frames), pero cuando lo anómalo pasa de un tercio del
    encuadre la referencia robusta se degrada y empieza a marcar de más.
    Para material así de malo la respuesta no es afinar el umbral: es que
    la línea ya está fuera de control y hay que pararla.
  - es ciego al FANTASMA tenue, y por construcción: compara el patrón
    contra copias desplazadas de sí mismo, y un fantasma es justamente una
    copia desplazada del patrón.  Medido, puntúa entre 0.93x y 1.03x del
    ruido del propio material.  Para ese fallo sirven el método de
    referencia o una clase entrenada.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from eyve.modules.base import InspectionModule, ModuleVerdict

_COLOR = (255, 120, 255)      # magenta (BGR) — distinto de las clases YOLO

#: ancho al que se reduce el frame para analizar.  El defecto más chico que
#: interesa mide varios píxeles aquí; bajar de esto pierde los finos.
_WORK_W = 480


class PatternModule(InspectionModule):
    name = "Patrón"

    #: métodos del módulo.  "periodo" no necesita nada; "referencia" aprende
    #: de frames buenos y sirve cuando el material NO se repite.
    METHODS = ("periodo", "referencia")

    def __init__(self) -> None:
        super().__init__()
        self.method: str = "periodo"
        #: 0..100 — cuánto hay que desviarse para marcar
        self.sensitivity: float = 50.0
        #: área mínima de una región, en % del frame
        self.min_area_pct: float = 0.08
        #: periodo en píxeles de trabajo; None = estimarlo solo
        self.period: Optional[Tuple[int, int]] = None
        self.auto_period: bool = True

        self._ref: Optional[np.ndarray] = None     # referencia aprendida
        self._ref_n: int = 0
        #: pico de ruido del material bueno, aprendido con calibrate().
        #: None = sin calibrar, y entonces se usa un umbral conservador.
        self._baseline: Optional[float] = None
        self._baseline_n: int = 0
        #: normalizacion aprendida del material bueno (mediana y escala de
        #: la energia local). Se guarda al calibrar y se REUSA: con varios
        #: defectos a la vez, la mediana del propio frame se contamina, los
        #: scores se comprimen y solo sobrevive el defecto mas fuerte.
        self._norm: Optional[Tuple[float, float]] = None
        self._regions: list[tuple[int, int, int, int]] = []
        self._score_max: float = 0.0
        self._verdict = ModuleVerdict()
        self._last_period: Optional[Tuple[int, int]] = None

    # ── configuración ─────────────────────────────────────────────────────
    def set_method(self, method: str) -> None:
        if method in self.METHODS and method != self.method:
            self.method = method
            self.reset()

    def reset(self) -> None:
        self._ref = None
        self._ref_n = 0
        self._baseline = None
        self._baseline_n = 0
        self._norm = None
        self._regions = []
        self._score_max = 0.0
        self._verdict = ModuleVerdict()
        if self.auto_period:
            self.period = None

    def learn_reference(self, frame: np.ndarray) -> None:
        """
        Suma un frame BUENO a la referencia (método "referencia").

        Se promedia en vez de quedarse con el último: un solo frame trae su
        propio ruido, y ese ruido se volvería el criterio de lo correcto.
        """
        g = self._prep(frame)
        if self._ref is None or self._ref.shape != g.shape:
            self._ref = g.astype(np.float32)
            self._ref_n = 1
        else:
            self._ref_n += 1
            self._ref += (g.astype(np.float32) - self._ref) / self._ref_n

    @property
    def reference_count(self) -> int:
        return self._ref_n

    # ── calibración contra el material ────────────────────────────────────
    def calibrate(self, frame: np.ndarray) -> Optional[float]:
        """
        Suma un frame de material BUENO a la calibración.

        Guarda el pico de anomalía que da el material sano, que es el piso
        sobre el que hay que destacar para ser un defecto.  Se queda con el
        MAYOR de los frames vistos, no con el promedio: basta que un frame
        bueno haya llegado a cierto nivel para que ese nivel no sirva como
        señal de defecto.
        """
        # primero se aprende la normalizacion de este frame bueno, y luego
        # se mide el pico ya con ella puesta
        crudo = self._local_energy(frame)
        if crudo is None:
            return None
        med = float(np.median(crudo))
        mad = float(np.median(np.abs(crudo - med)))
        sigma = max(mad * 1.4826, 0.004)
        if self._norm is None:
            self._norm = (med, sigma)
        else:
            n = self._baseline_n + 1
            self._norm = (self._norm[0] + (med - self._norm[0]) / n,
                          self._norm[1] + (sigma - self._norm[1]) / n)
        pico = float(((crudo - float(np.median(crudo))) / self._norm[1]).max())
        self._baseline_n += 1
        self._baseline = pico if self._baseline is None else max(self._baseline, pico)
        return self._baseline

    @property
    def calibrated(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline

    @property
    def calibration_count(self) -> int:
        return self._baseline_n

    def clear_calibration(self) -> None:
        self._baseline = None
        self._baseline_n = 0
        self._norm = None

    def raw_score(self, frame: np.ndarray) -> Optional[float]:
        """Pico de anomalía del frame, sin aplicar umbral."""
        mapa = self._score_map(frame)
        return None if mapa is None else float(mapa.max())

    @property
    def last_period(self) -> Optional[Tuple[int, int]]:
        """Periodo usado en el último análisis, en píxeles de trabajo."""
        return self._last_period

    # ── preparación del frame ─────────────────────────────────────────────
    @staticmethod
    def _prep(frame: np.ndarray) -> np.ndarray:
        """
        Gris, reducido y con el brillo local normalizado.

        La división por una versión muy borrosa quita el viñeteado y la
        iluminación despareja: sin eso, una esquina oscura se marca como
        defecto en cada frame y el operador aprende a ignorar el módulo.
        """
        h, w = frame.shape[:2]
        if w > _WORK_W:
            s = _WORK_W / float(w)
            frame = cv2.resize(frame, (_WORK_W, max(1, int(h * s))),
                               interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        fondo = cv2.GaussianBlur(g, (0, 0), 21)
        g = g / np.maximum(fondo, 1.0)
        return cv2.GaussianBlur(g, (0, 0), 1.0)

    # ── estimación del periodo ────────────────────────────────────────────
    @staticmethod
    def _period_1d(perfil: np.ndarray, lo: int, hi: int) -> Optional[int]:
        """Primer pico claro de la autocorrelación, entre lo y hi."""
        x = perfil - perfil.mean()
        if np.allclose(x, 0):
            return None
        n = 1 << int(np.ceil(np.log2(len(x) * 2)))
        f = np.fft.rfft(x, n)
        ac = np.fft.irfft(f * np.conj(f), n)[:len(x)]
        if ac[0] <= 0:
            return None
        ac = ac / ac[0]
        hi = min(hi, len(ac) - 1)
        if hi <= lo:
            return None
        seg = ac[lo:hi]
        k = int(np.argmax(seg))
        # un pico flojo significa que el material no se repite de verdad
        if seg[k] < 0.25:
            return None
        return lo + k

    def estimate_period(self, g: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Periodo (px, py) por autocorrelación de los perfiles.

        Proyectar sobre cada eje antes de autocorrelar hace la estimación
        barata y estable; para un estampado en tresbolillo el periodo que
        sale es el de la celda, que es justo el que sirve para comparar.
        """
        h, w = g.shape
        lo = 6
        px = self._period_1d(g.mean(axis=0), lo, max(lo + 2, w // 3))
        py = self._period_1d(g.mean(axis=1), lo, max(lo + 2, h // 3))
        if px is None and py is None:
            return None
        # si solo un eje se repite (rayas), se usa ese en los dos
        px = px or py
        py = py or px
        return int(px), int(py)

    # ── análisis ──────────────────────────────────────────────────────────
    def _local_energy(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Energía local SIN normalizar — es lo que la calibración resume."""
        return self._score_map(frame, raw=True)

    def _score_map(self, frame: np.ndarray,
                   raw: bool = False) -> Optional[np.ndarray]:
        """
        Mapa de anomalía en coordenadas de trabajo, o None si no aplica.

        Normalizado, usa la mediana y la escala APRENDIDAS si el módulo está
        calibrado, y las del propio frame si no.
        """
        if frame is None or frame.size == 0:
            return None
        g = self._prep(frame)
        if min(g.shape) < 16:
            return None

        if self.method == "referencia":
            if self._ref is None or self._ref.shape != g.shape:
                return None
            dif = np.abs(g - self._ref)
            if self._last_period is None:
                self._last_period = self.estimate_period(g) or (9, 9)
        else:
            per = self.period
            if per is None or self.auto_period:
                per = self.estimate_period(g)
            if per is None:
                self._last_period = None
                return None
            self._last_period = per
            dif = self._self_similarity(g, per)

        # ── energia local, no pixel suelto ────────────────────────────
        # El residuo de una tela buena esta repartido por todo el frame:
        # hay un borde de motivo en cada celda y, con la tela inclinada,
        # ningun borde calza al pixel. Un defecto, en cambio, es una
        # concentracion local. Se promedia a la escala de una repeticion y
        # se compara contra la mediana del propio frame: lo uniforme queda
        # en la mediana y no marca; lo concentrado sobresale.
        k = max(5, (int(max(self._last_period or (9, 9)) * 0.8) | 1))
        local = cv2.boxFilter(dif, -1, (k, k))

        if raw:
            return local

        if self._norm is not None:
            # Mediana del PROPIO frame, escala del material CALIBRADO.
            # Cada una por su motivo:
            #  - la mediana sigue al frame porque la normalizacion de
            #    contraste local desplaza el nivel global cuando hay
            #    defectos grandes; fijarla marcaba medio frame bueno.
            #  - la escala viene de la calibracion porque es la que se
            #    infla con varios defectos a la vez, comprimiendo los
            #    scores hasta que solo sobrevive el mas fuerte.
            return (local - float(np.median(local))) / self._norm[1]

        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med)))
        # Piso absoluto en la escala: sin el, un frame sin ninguna anomalia
        # divide entre casi cero y todo parece enorme. Es el error que hacia
        # que 45 de 48 frames limpios dieran falsa alarma.
        # OJO con el nombre: `escala` ya es el factor de coordenadas de
        # trabajo a coordenadas del frame. Reusarlo aqui dividia las cajas
        # entre 69 en vez de multiplicarlas por 2, y el modulo "no detectaba
        # nada" cuando en realidad detectaba y luego encogia el resultado.
        sigma = max(mad * 1.4826, 0.004)
        return (local - med) / sigma

    def analyze(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Regiones anómalas, en coordenadas del FRAME original."""
        score = self._score_map(frame)
        if score is None:
            self._regions = []
            self._score_max = 0.0
            return []
        self._score_max = float(score.max())
        gh, gw = score.shape
        escala = frame.shape[1] / float(gw)

        mask = (score > self.threshold).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        min_area = max(12.0, (self.min_area_pct / 100.0) * gh * gw)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        regiones = []
        for i in range(1, n):
            x, y, w_, h_, area = stats[i]
            if area < min_area:
                continue
            regiones.append((int(x * escala), int(y * escala),
                             int((x + w_) * escala), int((y + h_) * escala)))
        self._regions = regiones
        return regiones

    @staticmethod
    def _self_similarity(g: np.ndarray, per: Tuple[int, int]) -> np.ndarray:
        """
        Para cada píxel, la MENOR diferencia contra su repetición vecina.

        Se prueban las dos direcciones de cada eje más las dos diagonales:
        un píxel normal coincide con alguna de ellas; un defecto con
        ninguna.  Los bordes, donde no hay repetición que comparar, se
        dejan en el mínimo para no marcarlos.
        """
        px, py = max(2, per[0]), max(2, per[1])
        h, w = g.shape
        mejor = None
        for dx, dy in ((px, 0), (-px, 0), (0, py), (0, -py),
                       (px, py), (-px, -py)):
            if abs(dx) >= w or abs(dy) >= h:
                continue
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            # BORDER_REPLICATE y no CONSTANT: un borde negro inventado
            # seria una diferencia enorme y marcaria toda la orilla
            desp = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            d = np.abs(g - desp)
            mejor = d if mejor is None else np.minimum(mejor, d)
        if mejor is None:
            return np.zeros_like(g)
        # las franjas sin repetición al alcance no se juzgan
        m = max(px, py)
        if m * 2 < min(h, w):
            borde = np.zeros_like(mejor)
            borde[m:h - m, m:w - m] = 1.0
            mejor = mejor * borde
        return cv2.GaussianBlur(mejor, (0, 0), 1.5)

    # ── salida ────────────────────────────────────────────────────────────
    @property
    def regions(self) -> list[tuple[int, int, int, int]]:
        return list(self._regions)

    @property
    def score_max(self) -> float:
        """
        Pico de anomalia del ultimo frame, en unidades del umbral.

        Util para afinar: el operador sube la sensibilidad hasta que este
        numero quede por debajo del umbral con material bueno.
        """
        return self._score_max

    @property
    def threshold(self) -> float:
        """
        Umbral en unidades del mapa de anomalía.

        Calibrado: un múltiplo del pico que dio el material bueno.  La
        sensibilidad mueve el múltiplo entre 2.2x (solo lo evidente) y
        1.05x (apenas por encima del ruido del material).

        Sin calibrar: un valor conservador fijo.  Sirve para empezar, pero
        el ruido del material bueno varía de 4 a 16 según el estampado, así
        que sin calibrar el módulo o grita o se queda callado.
        """
        sens = float(np.clip(self.sensitivity, 0, 100))
        if self._baseline is not None:
            factor = 2.2 - 0.0115 * sens          # 2.2x .. 1.05x
            return max(0.5, self._baseline * factor)
        return 16.0 - 0.10 * sens                 # 16 .. 6, conservador

    def summary(self) -> str:
        if not self.enabled:
            return ""
        if self.method == "referencia" and self._ref is None:
            return "sin referencia"
        if self.method == "periodo" and not self.calibrated:
            return "sin calibrar"
        n = len(self._regions)
        return "sin anomalías" if n == 0 else f"{n} anomalía{'s' if n > 1 else ''}"

    def draw(self, annotated: np.ndarray) -> None:
        if not self.enabled:
            return
        for (x1, y1, x2, y2) in self._regions:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _COLOR, 2)
            cv2.putText(annotated, "?", (x1 + 4, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR, 2, cv2.LINE_AA)

    # ── contrato de InspectionModule ──────────────────────────────────────
    def process(self, frame: np.ndarray, detections: list,
                annotated: np.ndarray) -> ModuleVerdict:
        """
        Analiza el frame completo.  No mira `detections`: su trabajo es
        justamente encontrar lo que el detector no sabe nombrar.
        """
        if not self.enabled:
            self._verdict = ModuleVerdict()
            return self._verdict
        try:
            regiones = self.analyze(frame)
        except Exception:
            # un módulo nunca tumba el bucle de producción
            self._verdict = ModuleVerdict()
            return self._verdict
        if regiones:
            self._verdict = ModuleVerdict(
                ok=False,
                label=self.summary(),
                triggered_by="patrón",
                analyzed=len(regiones),
            )
        else:
            self._verdict = ModuleVerdict(ok=True, label=self.summary())
        return self._verdict
