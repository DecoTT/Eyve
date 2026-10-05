# -*- coding: utf-8 -*-
"""
Maraton: el stand corre 8 horas, y eso no se ha probado ni una vez.

Simula la demo textil funcionando sola durante horas —con el modo
automatico poniendo defectos— y mide lo que se degrada con el tiempo:

    memoria              una fuga mata el stand a media tarde
    tiempo por frame     si sube, la demo se arrastra
    tracks vivos         si crecen sin parar, el tracker se come la CPU
    defectos en la tela  si se acumulan, la tela se satura
    IDs asignados        crecen siempre, pero el ritmo dice si hay churn

No usa YOLO a proposito: lo que se busca son fugas y crecimientos del lado
de Eyve, y meter la inferencia haria la prueba 20 veces mas lenta sin
cambiar lo que se mide. El modulo Patron SI corre, porque es el que mas
trabajo hace por frame.
"""
import gc
import sys
import time
from pathlib import Path

REPO = Path(r"D:\Desarrollo\Claude Code\Eyve\2.1-dev")
sys.path.insert(0, str(REPO))

import numpy as np

from eyve.demo.textile import PRINT_FAULTS, TextilePattern, YOLO_CLASSES
from eyve.inference.tracker import InstanceTracker, RawDetection, TrackerConfig
from eyve.modules import CountingModule, PatternModule


def memoria_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        import resource  # no en Windows
        return 0.0


def main(minutos_simulados=480, fps=30):
    import random
    rng = random.Random(4)

    p = TextilePattern(width=960, height=540, axis="x", motif="diamantes",
                       weave="sarga", speed=110.0, tilt_deg=1.8)
    pm = PatternModule()
    pm.enabled = True
    pm.sensitivity = 70
    for _ in range(10):
        pm.calibrate(p.frame())
        p.advance(1 / fps)

    tk = InstanceTracker(config=TrackerConfig(
        iou_match=0.20, max_lost_frames=18, min_confirm_frames=2,
        process_conf_min=0.30))
    cm = CountingModule()
    cm.enabled = True
    cm.set_method("screen")

    total_frames = int(minutos_simulados * 60 * fps)
    # se simula a la velocidad que de la maquina, no en tiempo real
    paso_informe = total_frames // 12
    prox_defecto = 0.0
    t_sim = 0.0
    auto_puestos = 0

    print(f"simulando {minutos_simulados} min de stand "
          f"({total_frames:,} frames)\n")
    print(f"{'min sim':>8} {'RAM MB':>8} {'ms/frame':>9} {'tracks':>7} "
          f"{'defectos':>9} {'IDs':>8} {'conteo':>7}")

    m0 = memoria_mb()
    t_ventana = time.perf_counter()
    frames_ventana = 0
    hist = []

    for i in range(total_frames):
        dt = 1 / fps
        t_sim += dt
        p.advance(dt)

        # modo automatico, igual que la pantalla
        if t_sim >= prox_defecto:
            prox_defecto = t_sim + rng.uniform(11.0, 18.0)
            if auto_puestos >= 4:
                p.clear_defects()
                auto_puestos = 0
            else:
                auto_puestos += 1
                sx = rng.randint(150, 810)
                sy = rng.randint(110, 430)
                fx, fy = p.screen_to_fabric(sx, sy)
                if rng.random() < 0.5:
                    if rng.choice(YOLO_CLASSES) == "rayon":
                        p.streak(fx, fy, length=rng.randint(90, 280),
                                 thickness=rng.randint(4, 9))
                    else:
                        p.blob(fx, fy, size=rng.randint(50, 130))
                else:
                    k = rng.choice(PRINT_FAULTS)
                    if k == "fantasma":
                        p.ghost(fx, fy, size=rng.randint(130, 200))
                    elif k == "offset":
                        p.misregister(fx, fy, size=rng.randint(140, 220))
                    else:
                        p.ink_starved(fx, fy, size=rng.randint(120, 200),
                                      severity=rng.uniform(0.5, 1.0))

        frame = p.frame()
        pm.analyze(frame)
        dets = pm.as_detections()
        tr = tk.update(dets, frame)
        cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
        frames_ventana += 1

        if (i + 1) % paso_informe == 0:
            ahora = time.perf_counter()
            ms = (ahora - t_ventana) / max(1, frames_ventana) * 1000
            t_ventana, frames_ventana = ahora, 0
            from eyve.inference.tracker import TrackedInstance
            fila = (t_sim / 60, memoria_mb(), ms, len(tk.get_all()),
                    len(p.defects), TrackedInstance._id_counter, cm.total)
            hist.append(fila)
            print(f"{fila[0]:>8.0f} {fila[1]:>8.1f} {fila[2]:>9.2f} "
                  f"{fila[3]:>7} {fila[4]:>9} {fila[5]:>8} {fila[6]:>7}")

    print()
    mem_ini, mem_fin = hist[0][1], hist[-1][1]
    ms_ini, ms_fin = hist[0][2], hist[-1][2]
    print(f"memoria   : {mem_ini:.1f} -> {mem_fin:.1f} MB "
          f"({mem_fin - mem_ini:+.1f})")
    print(f"ms/frame  : {ms_ini:.2f} -> {ms_fin:.2f} "
          f"({(ms_fin / max(ms_ini, 1e-6) - 1) * 100:+.0f} %)")
    print(f"tracks    : {hist[0][3]} -> {hist[-1][3]}")
    print(f"defectos  : {hist[0][4]} -> {hist[-1][4]}")
    print(f"IDs       : {hist[-1][5]:,} asignados en total")
    print()
    problemas = []
    if mem_fin - mem_ini > 150:
        problemas.append(f"la memoria crecio {mem_fin - mem_ini:.0f} MB")
    if ms_fin > ms_ini * 1.5 and ms_fin > 5:
        problemas.append(f"el frame se volvio {ms_fin/ms_ini:.1f}x mas lento")
    if hist[-1][3] > 40:
        problemas.append(f"{hist[-1][3]} tracks vivos al final")
    if hist[-1][4] > 12:
        problemas.append(f"{hist[-1][4]} defectos acumulados en la tela")
    if problemas:
        print("PROBLEMAS:")
        for x in problemas:
            print("  -", x)
        return 1
    print("sin degradacion apreciable en toda la maraton")
    return 0


if __name__ == "__main__":
    mins = int(sys.argv[1]) if len(sys.argv) > 1 else 480
    raise SystemExit(main(mins))
