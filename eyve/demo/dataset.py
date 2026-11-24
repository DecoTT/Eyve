"""
Generador del dataset de la demo.

Produce un proyecto Eyve normal y corriente — raw/images, tagged/labels,
classes.yaml — que el flujo de entrenamiento de Eyve consume sin saber que
las imágenes no salieron de una cámara.  Eso importa para la expo: lo que
se muestra en el stand no es un camino especial, es el mismo pipeline.

La regla que hace que la demo funcione en vivo:

    el generador dibuja los defectos con LAS MISMAS primitivas
    (streak / blob / missing_print) con las que el visitante los dibuja
    en el canvas.

Si el dataset se generara con, digamos, elipses perfectas, el modelo
aprendería "elipse" y no reconocería el garabato de un visitante.  Al
compartir las primitivas, lo que la gente dibuja cae dentro de la
distribución de entrenamiento por construcción, no por suerte.

Uso:
    python -m eyve.demo.dataset --out "projects/Demo Textil" --frames 600
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import cv2

from eyve.core.project_manager import ClassDef, Project, create_project, load_project
from eyve.demo.tela_real import FALLAS_REALES, PALETAS, TelaRealista
from eyve.demo.textile import (MOTIFS, PRINT_FAULTS, TextilePattern,
                               WEAVES, YOLO_CLASSES)

#: fallas de linea que SI son clase de YOLO (lo puntual y nombrable); las
#: demas (doble impresion, franja sin tinta, arruga) van sin etiqueta: son de
#: Patron y de la mascara
_REALES_YOLO = {"raya_racla": "rayon", "hilacha": "rayon",
                "hilo_jalado": "rayon", "salpicadura": "mancha",
                "aceite": "mancha", "agujero": "mancha"}
_REALES_FONDO = tuple(f for f in FALLAS_REALES if f not in _REALES_YOLO)

#: cómo se ve cada clase en la UI de Eyve
_CLASS_COLORS = {
    "rayon":  "#ffd700",
    "mancha": "#00bcd4",
}

#: frames por tanda antes de reconstruir la tela (motivo/velocidad nuevos)
_BATCH = 24


def ensure_project(root: Path) -> Project:
    """
    Crea el proyecto demo con sus 3 clases, o carga el que ya exista.

    El nombre sale de la carpeta pedida: create_project arma
    <folder>/<name>, así que pasarle un nombre fijo dejaba una carpeta
    huérfana al lado de la que de verdad se quería.
    """
    root = Path(root).resolve()
    if (root / "project.yaml").exists():
        proj = load_project(root)
    else:
        proj = create_project(root.name, root.parent,
                              target="Demo textil de expo")
    proj.classes = [ClassDef(name=c, kind="nok", color=_CLASS_COLORS[c])
                    for c in YOLO_CLASSES]
    proj.paths.create_all()
    proj.save()
    return proj


def build_dataset(out: Path, frames: int = 600, width: int = 960,
                  height: int = 540, seed: int = 2026,
                  clean_ratio: float = 0.12,
                  progress=None, realista: bool = False) -> Project:
    """
    Genera *frames* imágenes etiquetadas en el proyecto demo de *out*.

    clean_ratio: fracción de frames SIN ningún defecto.  Van con etiqueta
    vacía a propósito: sin ejemplos limpios el modelo nunca aprende que la
    tela normal no es un defecto, y en la expo marcaría el patrón mismo.
    """
    rng = random.Random(seed)
    proj = ensure_project(Path(out))
    p = proj.paths
    img_dir = p.raw_images / "demo"
    img_dir.mkdir(parents=True, exist_ok=True)
    p.tagged_labels.mkdir(parents=True, exist_ok=True)

    cls_index = {c: i for i, c in enumerate(YOLO_CLASSES)}
    pattern: Optional[TextilePattern] = None
    written = 0

    for i in range(frames):
        # cada tanda estrena tela: motivo, velocidad e inclinación distintos,
        # para que el modelo no se ate a un fondo concreto
        if i % _BATCH == 0 and realista:
            pattern = TelaRealista(
                width=width, height=height,
                axis=rng.choice(("x", "y")),
                motif=rng.choice(MOTIFS),
                weave=rng.choice(("sarga", "tafetan", "sarga_fina", "canasta")),
                weave_amp=rng.uniform(0.06, 0.12),
                speed=rng.uniform(60, 190),
                tilt_deg=rng.uniform(-3.0, 3.0),
                seed=rng.randrange(1 << 30),
                paleta=rng.choice(list(PALETAS)),
                rollo=rng.randrange(1000),
                realismo=rng.uniform(0.6, 1.0),
                jpeg=rng.random() < 0.3,
            )
            pattern.offset = rng.uniform(0, pattern.fabric_len)
        elif i % _BATCH == 0:
            pattern = TextilePattern(
                width=width, height=height,
                axis=rng.choice(("x", "y")),
                motif=rng.choice(MOTIFS),
                # Variedad de tejido a proposito: si el modelo solo viera
                # una tela, la textura seria una dificultad; viendolas
                # todas, se vuelve invariancia. Se mide despues el mAP por
                # ligamento para saber si de verdad le costo.
                weave=rng.choice(WEAVES),
                weave_amp=rng.uniform(0.04, 0.11),
                speed=rng.uniform(60, 190),
                tilt_deg=rng.uniform(-3.5, 3.5),
                seed=rng.randrange(1 << 30),
            )
            pattern.offset = rng.uniform(0, pattern.fabric_len)

        assert pattern is not None
        pattern.clear_defects()
        pattern.advance(rng.uniform(0.05, 0.4))

        if realista:
            if rng.random() >= clean_ratio:
                for _ in range(rng.randint(1, 3)):
                    _paint_real(pattern, rng, rng.choice(list(_REALES_YOLO)))
            if rng.random() < 0.45:
                _paint_real(pattern, rng, rng.choice(_REALES_FONDO))
        else:
            if rng.random() >= clean_ratio:
                for _ in range(rng.randint(1, 3)):
                    _paint_random(pattern, rng)
            if rng.random() < 0.33:
                _paint_print_fault(pattern, rng)

        frame = pattern.frame()
        # Con la tela realista las rayas son largas y cruzan el encuadre: un
        # trozo de raya sigue siendo raya, y dejarlo sin caja le ensena a YOLO
        # que es fondo.  Por eso ahi se etiqueta lo visible desde un 25 %.
        labels = (pattern.visible_labels(min_visible=0.25) if realista
                  else pattern.visible_labels())

        stem = f"frame_{i:04d}"
        img_path = img_dir / f"{stem}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # La etiqueta DEBE escribirse con el resolvedor canónico del proyecto:
        # inventar el nombre aquí es exactamente el BUG-01 que rompió el
        # entrenamiento en silencio una vez.
        lines = []
        for cls, x1, y1, x2, y2 in labels:
            if cls not in cls_index:
                continue        # fallo de impresion: va sin etiqueta, a proposito
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            lines.append(f"{cls_index[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path = p.label_file(img_path)
        # Un frame limpio lleva etiqueta VACIA, no "sin etiqueta": YOLO lo
        # entiende como fondo negativo.  Pero collect_pairs descarta los
        # archivos de tamaño 0, asi que el frame limpio se guarda con una
        # linea en blanco para que sobreviva al recolector.
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1

        if progress and (i % 20 == 0 or i == frames - 1):
            progress(i + 1, frames)

    _write_data_yaml(proj)
    return proj


def _paint_random(pattern: TextilePattern, rng: random.Random) -> None:
    """
    Un defecto al azar, con las mismas primitivas del canvas de la demo.

    Los rangos importan tanto como las primitivas.  La primera versión
    generaba rayones de 55 a 190 px, y en la demo un visitante arrastra el
    mouse de lado a lado: un trazo de 300 px era algo que el modelo nunca
    había visto, así que lo partía en tres cajas y lo contaba tres veces.
    Ahora los rayones llegan a media diagonal del encuadre.
    """
    # margen: un defecto pegado al borde se descarta al etiquetar, así que
    # generarlos ahí solo gasta frames
    m = 60
    sx = rng.randint(m, pattern.width - m)
    sy = rng.randint(m, pattern.height - m)
    fx, fy = pattern.screen_to_fabric(sx, sy)
    kind = rng.choices(YOLO_CLASSES, weights=(0.55, 0.45))[0]
    diag = (pattern.width ** 2 + pattern.height ** 2) ** 0.5
    if kind == "rayon":
        # sesgado a los cortos (son los más comunes) pero con cola larga
        largo = int(rng.triangular(45, diag * 0.55, 140))
        pattern.streak(fx, fy, length=largo, thickness=rng.randint(3, 11))
    else:
        pattern.blob(fx, fy, size=int(rng.triangular(24, 170, 60)))


def _paint_print_fault(pattern: TextilePattern, rng: random.Random) -> None:
    """
    Un fallo de impresión, que va SIN etiqueta.

    Aparece en los frames de entrenamiento para que YOLO aprenda que es
    fondo —no un rayón ni una mancha— y lo deje en paz.  Quien lo encuentra
    es el módulo Patrón, que no necesita que nadie se lo enseñe.
    """
    m = 80
    sx = rng.randint(m, pattern.width - m)
    sy = rng.randint(m, pattern.height - m)
    fx, fy = pattern.screen_to_fabric(sx, sy)
    kind = rng.choice(PRINT_FAULTS)
    if kind == "fantasma":
        pattern.ghost(fx, fy, size=rng.randint(90, 220))
    elif kind == "offset":
        pattern.misregister(fx, fy, size=rng.randint(100, 240))
    else:
        pattern.ink_starved(fx, fy, size=rng.randint(80, 220),
                            severity=rng.uniform(0.45, 1.0))


def _paint_real(pattern: TelaRealista, rng: random.Random, tipo: str) -> None:
    """Una falla de linea de la tela realista, con su clase de YOLO o sin ella."""
    m = 70
    sx = rng.randint(m, pattern.width - m)
    sy = rng.randint(m, pattern.height - m)
    fx, fy = pattern.screen_to_fabric(sx, sy)
    kw = {}
    if tipo == "raya_racla":
        kw = dict(largo=rng.randint(150, 520), ancho=rng.randint(2, 5))
    elif tipo == "franja_sin_tinta":
        kw = dict(largo=rng.randint(300, 900))
    elif tipo == "doble_impresion":
        kw = dict(largo=rng.randint(250, 700))
    elif tipo == "salpicadura":
        kw = dict(size=rng.randint(20, 60))
    elif tipo == "aceite":
        kw = dict(size=rng.randint(50, 140))
    elif tipo == "agujero":
        kw = dict(size=rng.randint(6, 16))
    elif tipo == "hilacha":
        kw = dict(largo=rng.randint(40, 120))
    d = pattern.falla(tipo, fx, fy, **kw)
    # la clase de YOLO; las de fondo se quedan con la suya, que no esta en
    # YOLO_CLASSES y por eso no se etiqueta
    if tipo in _REALES_YOLO:
        d.cls = _REALES_YOLO[tipo]
    else:
        d.cls = "fondo_" + tipo


def _write_data_yaml(proj: Project) -> Path:
    """data.yaml del dataset, armado por el constructor normal de Eyve."""
    from eyve.training.dataset_builder import build_dataset as build_yolo
    return build_yolo(proj, val_split=0.15, seed=42)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Genera el dataset de la demo de expo")
    ap.add_argument("--out", default="projects/Demo_Textil",
                    help="carpeta del proyecto demo")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--realista", action="store_true",
                    help="tela realista y fallas de linea (eyve.demo.tela_real)")
    args = ap.parse_args(argv)

    def show(done, total):
        print(f"\r  {done}/{total} frames", end="", flush=True)

    proj = build_dataset(Path(args.out), frames=args.frames,
                         width=args.width, height=args.height,
                         seed=args.seed, progress=show, realista=args.realista)
    print(f"\nProyecto demo listo: {proj.root}")
    print(f"  clases: {', '.join(proj.class_names)}")
    print(f"  imagenes: {proj.raw_image_count()}")
    print(f"  dataset:  {proj.paths.dataset_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
