# Mide Patron como en la demo (sin YOLO): por estampado x tejido x fallo.
# Por defecto: atrapado = confirmado por el tracker en al menos un cuadro
# mientras esta visible; y % de cuadros visibles con el fallo confirmado.
# Falsas alarmas: tracks confirmados que no tocan ningun defecto.
import sys, random, json, time
REPO = r"D:\Desarrollo\Claude Code\Eyve\2.1-mascaras"
sys.path.insert(0, REPO)
from eyve.demo.textile import TextilePattern, MOTIFS, WEAVES, PRINT_FAULTS
from eyve.modules import PatternModule
from eyve.inference.tracker import InstanceTracker, TrackerConfig

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DT = 1 / 16.0            # la demo corre a ~16 fps con YOLO
W, H = 960, 540

def toca(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

def nuevo(motif, weave, seed):
    p = TextilePattern(width=W, height=H, axis="x", motif=motif,
                       weave=weave, speed=110.0, seed=seed)
    m = PatternModule(); m.enabled = True; m.sensitivity = 70.0
    for _ in range(10):
        m.calibrate(p.frame()); p.advance(0.07)
    return p, m

def tracker():
    return InstanceTracker(config=TrackerConfig(iou_match=0.20, max_lost_frames=18,
                                                min_confirm_frames=2, process_conf_min=0.30))

rng = random.Random(2027)
res = {}
t0 = time.time()
for motif in MOTIFS:
    for weave in WEAVES:
        # falsas alarmas en tela limpia
        p, m = nuevo(motif, weave, rng.randrange(1 << 30)); tr = tracker()
        fa = 0
        for _ in range(64):
            p.advance(DT); f = p.frame(); m.analyze(f)
            tks = tr.update(m.as_detections(), f)
            fa += sum(1 for t in tks if t.confirmed) > 0
        for fault in PRINT_FAULTS:
            atr = vis = conf = 0; fa_def = 0
            for k in range(K):
                p, m = nuevo(motif, weave, rng.randrange(1 << 30)); tr = tracker()
                sx = rng.randint(150, W - 150); sy = rng.randint(110, H - 110)
                fx, fy = p.screen_to_fabric(sx, sy)
                if fault == "fantasma": p.ghost(fx, fy, size=rng.randint(130, 200))
                elif fault == "offset": p.misregister(fx, fy, size=rng.randint(140, 220))
                else: p.ink_starved(fx, fy, size=rng.randint(120, 200), severity=rng.uniform(0.5, 1.0))
                visto = False; v = c = 0
                for _ in range(60):
                    f = p.frame(); m.analyze(f)
                    tks = [t for t in tr.update(m.as_detections(), f) if t.confirmed]
                    gt = [g[1:] for g in p.visible_labels()]
                    if gt:
                        v += 1
                        if any(toca(t.bbox, g) for t in tks for g in gt):
                            c += 1; visto = True
                    p.advance(DT)
                    if v and not gt: break
                atr += visto; vis += v; conf += c
            res[f"{motif}|{weave}|{fault}"] = dict(atrapados=atr, de=K,
                                                   pct_cuadros=round(100 * conf / max(vis, 1), 1))
        res[f"{motif}|{weave}|limpia"] = dict(cuadros_con_falsa=fa, de=64)
        print(motif, weave, {k.split('|')[2]: v for k, v in res.items()
                             if k.startswith(f"{motif}|{weave}|")}, f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "medicion.json", "w"), indent=1)
