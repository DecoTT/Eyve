# -*- coding: utf-8 -*-
"""Tabla por material de un resultado de medir.py.

    python tabla.py resultados/video.json [resultados/linea_base.json]
"""
import json
import sys

r = json.load(open(sys.argv[1], encoding="utf-8"))
base = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
CL = ("antes", "mascara", "despues")


def cuenta(fs):
    """atrapados y % de cuadros, por seguimiento"""
    o = {}
    for c in CL:
        at = sum(1 for f in fs if f["conf"][c] > 0)
        vis = sum(f["visibles"] for f in fs)
        o[c] = (at, 100.0 * sum(f["conf"][c] for f in fs) / max(vis, 1))
    return o


tot = {"fantasma": [], "otros": []}
fa = {c: 0 for c in CL}
lejos = {c: 0 for c in CL}
lejos_n = 0
n_cuad = 0
sin_eng = 0
ms = []
print("| Material | Fantasma antes | después | solo máscara | Movido + falta antes | después | "
      "Falsas limpia antes | después | Lejos del fallo antes | después |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in r["materiales"]:
    fan = [f for f in m["fallos"] if f["fault"] == "fantasma"]
    otr = [f for f in m["fallos"] if f["fault"] != "fantasma"]
    tot["fantasma"] += fan
    tot["otros"] += otr
    cf, co = cuenta(fan), cuenta(otr)
    l = m["limpia"]
    for c in CL:
        fa[c] += l["falsas"][c]
        lejos[c] += sum(f["lejos"][c] for f in m["fallos"])
    lejos_n += sum(f["visibles"] for f in m["fallos"])
    n_cuad += l["cuadros"]
    sin_eng += l["sin_enganche"]
    ms.append(l["ms"])
    la = sum(f["lejos"]["antes"] for f in m["fallos"])
    ld = sum(f["lejos"]["despues"] for f in m["fallos"])
    print(f"| {m['motif']} · {m['weave']} "
          f"| {cf['antes'][0]}/{len(fan)} · {cf['antes'][1]:.0f} % "
          f"| {cf['despues'][0]}/{len(fan)} · {cf['despues'][1]:.0f} % "
          f"| {cf['mascara'][0]}/{len(fan)} · {cf['mascara'][1]:.0f} % "
          f"| {co['antes'][0]}/{len(otr)} · {co['antes'][1]:.0f} % "
          f"| {co['despues'][0]}/{len(otr)} · {co['despues'][1]:.0f} % "
          f"| {l['falsas']['antes']}/{l['cuadros']} | {l['falsas']['despues']}/{l['cuadros']} "
          f"| {la} | {ld} |")
cf, co = cuenta(tot["fantasma"]), cuenta(tot["otros"])
print()
print(f"TOTAL fantasma: antes {cf['antes'][0]}/{len(tot['fantasma'])} ({cf['antes'][1]:.1f} % cuadros)  "
      f"despues {cf['despues'][0]}/{len(tot['fantasma'])} ({cf['despues'][1]:.1f} %)  "
      f"mascara sola {cf['mascara'][0]} ({cf['mascara'][1]:.1f} %)")
print(f"TOTAL otros:    antes {co['antes'][0]}/{len(tot['otros'])} ({co['antes'][1]:.1f} %)  "
      f"despues {co['despues'][0]}/{len(tot['otros'])} ({co['despues'][1]:.1f} %)  "
      f"mascara sola {co['mascara'][0]} ({co['mascara'][1]:.1f} %)")
print(f"Falsas en limpia (cuadros con algo confirmado): antes {fa['antes']}  mascara {fa['mascara']}  "
      f"despues {fa['despues']}  de {n_cuad}; sin enganche {sin_eng}")
print(f"Lejos del fallo: antes {lejos['antes']}  mascara {lejos['mascara']}  despues {lejos['despues']}  "
      f"de {lejos_n} cuadros con fallo")
mp = [m["limpia"].get("ms_patron", 0) for m in r["materiales"]]
print(f"Costo: mascara {sum(ms)/len(ms):.1f} ms/cuadro, Patron {sum(mp)/len(mp):.1f} ms/cuadro; avisos: "
      f"{[ (m['motif'],m['weave'],m['avisos']) for m in r['materiales'] if m['avisos']]}")
malos = [(m["motif"], m["weave"], f["fault"], f["k"], f["sx"], f["sy"], f["size"])
         for m in r["materiales"] for f in m["fallos"] if f["conf"]["despues"] == 0]
print("Escapes despues:", malos)
malos = [(m["motif"], m["weave"], f["fault"], f["k"], f["sx"], f["sy"], f["size"])
         for m in r["materiales"] for f in m["fallos"] if f["conf"]["antes"] == 0]
print("Escapes antes:", malos)
if base:
    dif = []
    for m in r["materiales"]:
        for fault in ("fantasma", "offset", "falta_impresion"):
            fs = [f for f in m["fallos"] if f["fault"] == fault]
            at = sum(1 for f in fs if f["conf"]["antes"] > 0)
            pc = round(100 * sum(f["conf"]["antes"] for f in fs) / max(sum(f["visibles"] for f in fs), 1), 1)
            b = base[f"{m['motif']}|{m['weave']}|{fault}"]
            if (at, pc) != (b["atrapados"], b["pct_cuadros"]):
                dif.append((m["motif"], m["weave"], fault, at, pc, b))
    print("Diferencias de 'antes' contra la linea base:", dif or "ninguna")
