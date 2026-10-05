# -*- coding: utf-8 -*-
"""
Smoke test de la pantalla que explica el modulo de conteo.

Lo que importa de esta pantalla no es que no reviente: es que lo que
ensena sea cierto. Un visitante que ve "cruce de meta" tiene que estar
viendo una meta de verdad y un numero que cuenta cruces de verdad, no una
animacion bonita con un contador suelto al lado.

Asi que se comprueba, para cada metodo: que la escena cambie, que la
geometria que corresponde quede puesta, que los textos sean los suyos, y
que el bucle avance solo.
"""
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

import customtkinter as ctk
from eyve.i18n import t, set_language
from eyve.ui import theme as T
T.apply("dark")
set_language("es")

from eyve.demo.conveyor import SCENES, SCENE_INFO

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FALLA'}  {name}: got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


def check_true(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


class FakeApp(ctk.CTk):
    def get_project(self):
        return None


app = FakeApp()
app.geometry("1500x900")

from eyve.ui.screens import counting_demo_screen as CDS
scr = CDS.CountingDemoScreen(app, app)
scr.pack(fill="both", expand=True)
app.update()

print("\n[1] Arranca en el primer metodo")
check("metodo inicial", scr._counting.method, SCENES[0])
check("escena en el mismo metodo", scr._scene.method, SCENES[0])
check("el bucle arranca encendido", scr._auto, True)

print("\n[2] Cada metodo pone SU geometria y SUS textos")
for i, metodo in enumerate(SCENES):
    scr._idx = i
    scr._aplicar_metodo(metodo)
    app.update()
    info = SCENE_INFO[metodo]
    check(f"{metodo}: el modulo de conteo lo toma", scr._counting.method, metodo)
    check(f"{metodo}: la escena lo toma", scr._scene.method, metodo)

    # la geometria tiene que coincidir con lo que la escena dibuja
    if info.draws_line:
        check(f"{metodo}: hay meta y donde la escena la pinta",
              scr._counting.line,
              (scr._scene.line_x, 0, scr._scene.line_x, scr._scene.height))
    else:
        check(f"{metodo}: sin meta", scr._counting.line, None)
    if info.draws_zone:
        check(f"{metodo}: hay zona y donde la escena la pinta",
              scr._counting.zone, scr._scene.zone)
    else:
        check(f"{metodo}: sin zona", scr._counting.zone, None)
    check(f"{metodo}: rango esperado",
          (scr._counting.expect_min, scr._counting.expect_max), info.expect)

    # textos: los del metodo, no los de otro
    check(f"{metodo}: titulo", scr._metodo_lbl.cget("text"),
          t("count_m_" + metodo))
    check(f"{metodo}: la pregunta", scr._pregunta_lbl.cget("text"),
          t("cdemo_q_" + metodo))
    # texto propio de esta pantalla: el de Produccion termina en "dibuja
    # la linea sobre el video", instruccion que aqui no aplica
    check(f"{metodo}: como funciona", scr._como_lbl.cget("text"),
          t("cdemo_como_" + metodo))
    check_true(f"{metodo}: sin instrucciones que aqui no aplican",
               "ibuja" not in scr._como_lbl.cget("text"),
               scr._como_lbl.cget("text")[:60])
    check(f"{metodo}: donde sirve", scr._uso_lbl.cget("text"),
          t("cdemo_uso_" + metodo))
    check(f"{metodo}: el contador vuelve a cero", scr._num_lbl.cget("text"), "0")

print("\n[3] El bucle avanza solo y da la vuelta")
scr._idx = 0
scr._aplicar_metodo(SCENES[0])
for esperado in list(SCENES[1:]) + [SCENES[0]]:
    scr._saltar(1)
    check(f"avanza a {esperado}", scr._counting.method, esperado)
# y hacia atras
scr._saltar(-1)
check("tambien va hacia atras", scr._counting.method, SCENES[-1])

print("\n[4] Se puede detener en uno")
scr._auto = True
scr._toggle_auto()
check("detenido", scr._auto, False)
check("el boton lo dice", scr._auto_btn.cget("text"), t("cdemo_auto_off"))
scr._toggle_auto()
check("vuelve a avanzar", scr._auto, True)
check("y el boton tambien", scr._auto_btn.cget("text"), t("cdemo_auto_on"))

print("\n[5] El bucle corre y el contador se mueve")
scr._idx = SCENES.index("line")
scr._aplicar_metodo("line")
scr._auto = False               # que no salte de metodo mientras medimos
scr._running = True
scr._ultimo = 0.0
import time as _time
vistos = set()
for _ in range(240):
    scr._ultimo = _time.perf_counter() - 1 / 30
    frame, dets = scr._scene.step(1 / 30)
    tr = scr._tracker.update(dets, frame)
    scr._counting.update_tracks(tr, expired=scr._tracker.last_expired,
                                frame_wh=(960, 540))
    vistos.add(scr._counting.total)
check_true("el contador avanza con las piezas", max(vistos) >= 2,
           f"valores vistos: {sorted(vistos)}")

print("\n[6] Las cadenas existen en los dos idiomas")
from eyve.i18n.es import STRINGS as ES
from eyve.i18n.en import STRINGS as EN
claves = (["nav_cdemo", "cdemo_title", "cdemo_sub", "cdemo_paso",
           "cdemo_count", "cdemo_donde", "cdemo_auto_on", "cdemo_auto_off",
           "cdemo_dir", "cdemo_zona", "cdemo_esperado"]
          + [f"cdemo_q_{m}" for m in SCENES]
          + [f"cdemo_uso_{m}" for m in SCENES]
          + [f"cdemo_como_{m}" for m in SCENES])
faltan = [k for k in claves if k not in ES or k not in EN]
check("ninguna clave falta", faltan, [])

print("\n[7] Cierre limpio")
scr.on_hide()
check("el bucle se detuvo", scr._running, False)
scr.on_close()
app.destroy()
print("  OK    on_hide/on_close sin excepciones")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("PANTALLA DE CONTEO: TODO PASO")
