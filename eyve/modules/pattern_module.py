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

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from eyve.modules.base import InspectionModule, ModuleVerdict

_COLOR = (255, 120, 255)      # magenta (BGR) — distinto de las clases YOLO
_COLOR_EXTRA = (0, 160, 255)  # naranja: hay tinta donde no deberia
_COLOR_FALTA = (255, 220, 0)  # cian:    falta tinta donde si deberia

#: ancho al que se reduce el frame para analizar.  El defecto más chico que
#: interesa mide varios píxeles aquí; bajar de esto pierde los finos.
_WORK_W = 480


class PatternModule(InspectionModule):
    name = "Patrón"

    #: métodos del módulo:
    #:   periodo     ¿esto se parece a sus vecinos?  No necesita nada.
    #:   layout      ¿dónde debería haber tinta y dónde no?  Distingue
    #:               "tinta de más" de "tinta de menos", y por eso ve el
    #:               fantasma, que al método periodo se le escapa.
    #:   referencia  aprende de frames buenos; para material que NO se repite.
    METHODS = ("periodo", "layout", "referencia")

    #: qué significa cada signo de la diferencia contra lo esperado
    KIND_EXTRA = "tinta de más"
    KIND_FALTA = "falta tinta"

    #: clase con la que las anomalías entran al tracker.  No es una clase
    #: entrenada: es la etiqueta con la que viajan para heredar confirmación
    #: por frames, ID estable y conteo, igual que cualquier instancia.
    ANOMALY_LABEL = "anomalía"

    def __init__(self) -> None:
        super().__init__()
        self.method: str = "periodo"
        #: 0..100 — cuánto hay que desviarse para marcar
        self.sensitivity: float = 50.0
        #: área mínima de una región, en % del frame
        self.min_area_pct: float = 0.08
        #: margen de borde, en % del ancho.  Una región pegada a la orilla
        #: es un defecto a medio entrar o a medio salir: su forma cambia en
        #: cada frame, se parte y se vuelve a unir, y cada pedazo nuevo se
        #: vuelve una instancia distinta.  Medido, es de donde salian los
        #: conteos absurdos (29 por dos fallos).  0 = no filtrar.
        self.edge_margin_pct: float = 2.0
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
        #: picos CRUDOS (max - mediana) de cada frame de calibracion. Se
        #: guardan sin normalizar y se dividen al final por la escala ya
        #: promediada: dividir sobre la marcha, con la escala a medio
        #: converger, inflaba el piso un 50% con el material en
        #: movimiento — y en una linea real el material siempre se mueve.
        self._cal_peaks: list = []
        self._regions: list[tuple[int, int, int, int]] = []
        #: con el método layout, qué tipo es cada región de _regions
        self._kinds: list[str] = []
        self._score_max: float = 0.0
        self._verdict = ModuleVerdict()
        self._last_period: Optional[Tuple[int, int]] = None
        #: diferencia con signo del último frame (método layout)
        self._signed: Optional[np.ndarray] = None
        #: inclinación de la retícula estimada en el último frame
        self._angle: float = 0.0

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
        self._cal_peaks = []
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
        # Se guarda el pico CRUDO y se normaliza al final, con la escala
        # ya promediada. Normalizar aqui, con la escala a medio converger,
        # inflaba el piso de 1.47 a 2.25 con la tela en movimiento: el
        # umbral subia a 3.14 y los fallos tenues quedaban por debajo.
        self._cal_peaks.append(float((crudo - float(np.median(crudo))).max()))
        self._baseline_n += 1
        self._baseline = max(self._cal_peaks) / self._norm[1]
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
        self._cal_peaks = []

    def raw_score(self, frame: np.ndarray) -> Optional[float]:
        """Pico de anomalía del frame, sin aplicar umbral."""
        mapa = self._score_map(frame)
        return None if mapa is None else float(mapa.max())

    @property
    def last_period(self) -> Optional[Tuple[int, int]]:
        """Periodo usado en el último análisis, en píxeles de trabajo."""
        return self._last_period

    # ── preparación del frame ─────────────────────────────────────────────
    def _prep(self, frame: np.ndarray) -> np.ndarray:
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
        g = cv2.GaussianBlur(g, (0, 0), 1.0)

        # Enderezar la retícula antes de analizar: los desplazamientos de un
        # periodo asumen que la repetición va a lo largo de los ejes, y una
        # retícula girada los desalinea un poco más en cada salto.
        ang = self.estimate_angle(g)
        self._angle = ang
        if abs(ang) > 0.2:
            h2, w2 = g.shape
            M = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), ang, 1.0)
            g = cv2.warpAffine(g, M, (w2, h2), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        return g

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

    @staticmethod
    def estimate_angle(g: np.ndarray) -> float:
        """
        Inclinación de la retícula, en grados.

        Se autocorrelaciona la imagen en 2D: el pico más fuerte fuera del
        centro cae justo donde está la repetición más cercana, así que su
        ángulo ES el de la retícula.  Se busca solo en ±30 grados: más allá
        el eje "horizontal" y el "vertical" se confunden y enderezar saldría
        peor que no hacerlo.
        """
        h, w = g.shape
        if min(h, w) < 48:
            return 0.0
        x = g - g.mean()
        F = np.fft.rfft2(x)
        ac = np.fft.irfft2(F * np.conj(F), x.shape)
        ac = np.fft.fftshift(ac)
        cy, cx = h // 2, w // 2
        r = min(h, w) // 3
        ventana = ac[max(0, cy - r):cy + r, max(0, cx - r):cx + r].copy()
        vy, vx = ventana.shape
        oy, ox = vy // 2, vx // 2
        # tapar el centro: el pico de lag cero no dice nada
        yy, xx = np.ogrid[:vy, :vx]
        d2 = (yy - oy) ** 2 + (xx - ox) ** 2
        ventana[d2 < 36] = -np.inf
        k = int(np.argmax(ventana))
        py_, px_ = divmod(k, vx)
        dy, dx = py_ - oy, px_ - ox
        if dx == 0 and dy == 0:
            return 0.0
        ang = np.degrees(np.arctan2(dy, dx))
        # llevar a la familia de ejes mas cercana (cada 90 grados)
        ang = (ang + 45.0) % 90.0 - 45.0
        return float(ang) if abs(ang) <= 30.0 else 0.0

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
            if self.method == "layout":
                esperado = self._expected(g, per)
                # Diferencia CON SIGNO contra lo esperado.  Se guarda para
                # poder decir después de qué lado está cada región: más
                # oscuro que lo esperado es tinta de más, más claro es
                # tinta que falta.
                self._signed = cv2.GaussianBlur(g - esperado, (0, 0), 1.2)
                dif = np.abs(self._signed)
            else:
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

        # Cierre a la escala de UNA repeticion del estampado.  Sin esto un
        # mismo defecto sale roto en pedazos —medido: hasta 22 regiones para
        # un solo fallo— y cada pedazo se vuelve una instancia distinta, asi
        # que un fallo cruzando el encuadre llegaba a contarse 159 veces.
        # El tamano no es arbitrario: dos defectos separados por mas de una
        # repeticion siguen siendo dos (comprobado en la prueba).
        per = self._last_period or (9, 9)
        kc = max(3, (int(max(per) * 1.0) | 1))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))

        min_area = max(12.0, (self.min_area_pct / 100.0) * gh * gw)
        m_borde = int((self.edge_margin_pct / 100.0) * gw)
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        regiones = []
        tipos = []
        for i in range(1, n):
            x, y, w_, h_, area = stats[i]
            if area < min_area:
                continue
            # Pegada a la orilla: el defecto esta a medio entrar o a medio
            # salir. Se reporta cuando acabe de entrar, no antes.
            if m_borde > 0 and (x <= m_borde or y <= m_borde
                                or x + w_ >= gw - m_borde
                                or y + h_ >= gh - m_borde):
                continue
            caja = self._unrotate((x, y, x + w_, y + h_), gw, gh)
            regiones.append(tuple(int(v * escala) for v in caja))
            tipos.append(self._kind_of(etiquetas == i))
        self._regions = regiones
        self._kinds = tipos
        return regiones

    def _unrotate(self, caja, gw: int, gh: int):
        """
        Caja del espacio enderezado de vuelta al del frame.

        Se giran las cuatro esquinas y se toma su caja envolvente, que es
        lo mismo que hace el etiquetado del dataset: una caja alineada a los
        ejes no puede representar un giro, así que se toma la que lo
        contiene.
        """
        if abs(self._angle) <= 0.2:
            return caja
        x1, y1, x2, y2 = caja
        a = math.radians(-self._angle)
        cx, cy = gw / 2.0, gh / 2.0
        xs, ys = [], []
        for px_, py_ in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
            dx, dy = px_ - cx, py_ - cy
            xs.append(dx * math.cos(a) - dy * math.sin(a) + cx)
            ys.append(dx * math.sin(a) + dy * math.cos(a) + cy)
        return (max(0, min(xs)), max(0, min(ys)),
                min(gw, max(xs)), min(gh, max(ys)))

    def _kind_of(self, mascara: np.ndarray) -> str:
        """De qué lado está la región: tinta de más o tinta que falta."""
        if self._signed is None or self.method != "layout":
            return ""
        v = float(np.mean(self._signed[mascara]))
        # g está normalizado por el fondo local: más oscuro = valor menor
        return self.KIND_EXTRA if v < 0 else self.KIND_FALTA

    @staticmethod
    def _expected(g: np.ndarray, per: Tuple[int, int]) -> np.ndarray:
        """
        El estampado que DEBERÍA verse, reconstruido del propio material.

        Para cada píxel se juntan sus valores en las repeticiones vecinas y
        se toma la MEDIANA.  Es lo que impide que un defecto se cuele en su
        propia referencia: entre ocho repeticiones, una mala no mueve la
        mediana.  Un promedio sí la movería, y el defecto quedaría medio
        perdonado.
        """
        px, py = max(2, per[0]), max(2, per[1])
        h, w = g.shape
        # Repeticiones a UNO, DOS y TRES periodos de distancia, no solo las
        # pegadas. Con los vecinos inmediatos la referencia se contamina a
        # si misma: un fantasma de tres celdas de ancho tambien fantasmea a
        # sus vecinos, la mediana sale fantasmeada y el fallo desaparece.
        # Desde dos o tres periodos ya se sale del defecto.
        # Un periodo de distancia, no tres. Probado: buscar la referencia
        # mas lejos la empeora, porque cada salto acumula el error que deja
        # la reticula enderezada solo aproximadamente, y "lo esperado" sale
        # borroso. El contagio del defecto a sus vecinos se tolera: para eso
        # esta la mediana.
        capas = []
        saltos = [(px, 0), (-px, 0), (0, py), (0, -py),
                  (px, py), (-px, -py), (px, -py), (-px, py)]
        for dx, dy in saltos:
            if abs(dx) >= w or abs(dy) >= h:
                continue
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            capas.append(cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE))
        if not capas:
            return g.copy()
        return np.median(np.stack(capas, axis=0), axis=0).astype(np.float32)

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
        if n == 0:
            return "sin anomalías"
        if self.method == "layout" and self._kinds:
            extra = sum(1 for k in self._kinds if k == self.KIND_EXTRA)
            falta = n - extra
            partes = []
            if extra:
                partes.append(f"{extra} tinta de más")
            if falta:
                partes.append(f"{falta} falta tinta")
            return "  ".join(partes)
        return f"{n} anomalía{'s' if n > 1 else ''}"

    def draw(self, annotated: np.ndarray) -> None:
        if not self.enabled:
            return
        for i, (x1, y1, x2, y2) in enumerate(self._regions):
            tipo = self._kinds[i] if i < len(self._kinds) else ""
            if tipo == self.KIND_EXTRA:
                col, txt = _COLOR_EXTRA, "+"
            elif tipo == self.KIND_FALTA:
                col, txt = _COLOR_FALTA, "-"
            else:
                col, txt = _COLOR, "?"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            cv2.putText(annotated, txt, (x1 + 4, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    @property
    def kinds(self) -> list[str]:
        """Tipo de cada región de `regions` (solo con el método layout)."""
        return list(self._kinds)

    # ── las anomalías como instancias ─────────────────────────────────────
    @staticmethod
    def _solapa(a, b) -> float:
        """Intersección sobre el área de la caja más chica."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        if inter == 0:
            return 0.0
        aa = max(1, (ax2 - ax1) * (ay2 - ay1))
        bb = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / min(aa, bb)

    def as_detections(self, named: Optional[list] = None,
                      solape_max: float = 0.35) -> list:
        """
        Las regiones de este frame, como detecciones para el tracker.

        named: cajas (x1, y1, x2, y2) que el detector YA nombró.  Una región
        que coincide con una de ellas NO se reporta: si YOLO ya dijo que eso
        es un rayón, el Patrón no tiene que volver a levantar la mano — y si
        lo hiciera, el mismo defecto se contaría dos veces.

        Las detecciones salen con la clase ANOMALY_LABEL, así que al entrar
        al tracker heredan la confirmación por frames que mata el parpadeo:
        una región que aparece un solo frame nunca llega a confirmarse y por
        tanto nunca se dibuja ni se cuenta.
        """
        from eyve.inference.tracker import RawDetection
        cajas = named or []
        salida = []
        for (x1, y1, x2, y2) in self._regions:
            if any(self._solapa((x1, y1, x2, y2), c) > solape_max
                   for c in cajas):
                continue
            salida.append(RawDetection(self.ANOMALY_LABEL, 1.0,
                                       int(x1), int(y1), int(x2), int(y2)))
        return salida

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
