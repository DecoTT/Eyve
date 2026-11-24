"""
Corre la pantalla Demo de verdad (bucle, modelo YOLO de la demo y Patrón) con la
tela realista y la máscara, y guarda capturas del recorrido.  Para revisar a ojo
y para planear las tomas del video; no es prueba automática.

    python mascaras/captura_demo.py <carpeta_salida> [motivo] [tejido]
"""
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import customtkinter as ctk
from PIL import ImageGrab
from eyve.i18n import set_language
from eyve.ui import theme as T

T.apply("dark")
set_language("es")
from eyve.ui.screens import demo_screen as DS  # noqa: E402

salida = Path(sys.argv[1])
salida.mkdir(parents=True, exist_ok=True)
motivo = sys.argv[2] if len(sys.argv) > 2 else "flores"
tejido = sys.argv[3] if len(sys.argv) > 3 else "sarga"


class FakeApp(ctk.CTk):
    license = types.SimpleNamespace(permite_checks_avanzados=lambda: True)

    def get_project(self):
        return None


app = FakeApp()
app.geometry("1600x950+0+0")
scr = DS.DemoScreen(app, app)
scr.pack(fill="both", expand=True)
app.update()
scr.on_show()


def foto(nombre, ventana=None):
    w = ventana or app
    w.update()
    x, y = w.winfo_rootx(), w.winfo_rooty()
    ImageGrab.grab((x, y, x + w.winfo_width(), y + w.winfo_height())).save(
        salida / f"{nombre}.png")
    print("captura", nombre, flush=True)


pasos = []


def en(seg, fn):
    pasos.append((seg, fn))


def realista():
    scr._real_sw.select()
    scr._on_realista()
    scr._rebuild_fabric(motivo, tejido)
    scr._auto = False
    scr._refresh_auto_label()


def falla(tipo):
    def f():
        p = scr._pattern
        fx = int(p.offset + p.width - 160)
        p.falla(tipo, fx, 270 if tipo != "hilo_jalado" else 200)
    return f


en(1.0, realista)
en(4.0, lambda: foto("01_tela_realista"))
en(5.0, scr._mask_learn_click)
en(7.0, lambda: foto("02_aprendiendo"))


def esperar_revision():
    if scr._masc_fase != "revisar" or scr._mask_top is None:
        app.after(300, esperar_revision)
        return
    app.after(800, lambda: foto("03_revision", scr._mask_top))
    app.after(1600, lambda: scr._mask_decidir(True))
    app.after(2500, lambda: setattr(scr, "_auto", False))
    t0 = 3000
    fallas = ("doble_impresion", "aceite", "franja_sin_tinta", "salpicadura",
              "raya_racla", "agujero", "arruga")
    for i, tipo in enumerate(fallas):
        app.after(t0 + i * 9000, falla(tipo))
        app.after(t0 + i * 9000 + 6500, lambda n=i, tp=tipo: foto(f"{4 + n:02d}_{tp}"))
    app.after(t0 + len(fallas) * 9000 + 500, app.destroy)


en(9.0, esperar_revision)
for seg, fn in pasos:
    app.after(int(seg * 1000), fn)
app.mainloop()
print("fin")
