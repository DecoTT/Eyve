# -*- coding: utf-8 -*-
"""
Vuelta sola a modo automatico tras un rato sin que nadie toque.

La comprobacion que de verdad importa no es que vuelva —eso es facil— sino
QUE REINICIA EL RELOJ.  Si el modo automatico colocando defectos contara
como tocar, el stand vacio no se limpiaria nunca y la prueba de "vuelve
sola" pasaria igual, porque en una prueba siempre hay alguien llamando a
los metodos.  Por eso cada caso bueno va con el malo:

    toca una persona            -> el reloj se mueve
    el automatico pone defectos -> el reloj NO se mueve
    el boton "Limpiar tela"     -> el reloj se mueve
    reset_demo() por dentro     -> el reloj NO se mueve

El reloj se manipula poniendo `_last_touch` en el pasado en vez de
esperar tres minutos: asi la prueba es determinista y no depende de
dormir.
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("YOLO_OFFLINE", "1")

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


# ── neutralizar camara, modelo, red y repintado ───────────────────────────
from eyve.inference import detector as _det
_det.VideoSource.start = lambda self: False
_det.VideoSource.stop = lambda self: None
_det.VideoSource.read = lambda self: None

from eyve.ui.screens import production_screen as _ps
_ps.ProductionScreen._try_autoload_model = lambda self: None
from eyve.ui.screens import demo_screen as _ds
_ds.DemoScreen._load_model = lambda self: None
_ds.DemoScreen._paint = lambda self, canvas, frame, attr: (960, 540, 960, 540, 0, 0)
from eyve.ui.screens import counting_demo_screen as _cds
_cds.CountingDemoScreen._pintar = lambda self, frame: None

from eyve.core import updater as _U
_U.check_async = lambda cb: None

import customtkinter as ctk

from eyve.demo.textile import MOTIFS, WEAVES
from eyve.i18n import t
from eyve.modules import CountingModule
from eyve.ui.app import EyveApp


def settle(app, veces=4):
    for _ in range(veces):
        app.update()
        app.update_idletasks()


app = EyveApp()
app.geometry("1200x800+80+40")
app.navigate("nav_demo")
settle(app)
demo = app._screens["nav_demo"]
# la calibracion se come los primeros frames; aqui estorba
demo._cal_left = 0

print("\n[1] Solo una persona reinicia el reloj")
demo._auto = True
antes = demo._last_touch = time.time() - 50.0
# el automatico trabajando: 12 defectos puestos por su cuenta
for _ in range(12):
    demo._auto_next = 0.0
    demo._auto_tick()
check("el automatico puso defectos",
      demo._auto_placed > 0 or len(demo._pattern.defects) > 0,
      f"{len(demo._pattern.defects)} en la tela")
check("el automatico NO reinicia el reloj", demo._last_touch == antes,
      f"{demo._last_touch - antes:+.3f} s")
# y el paso del tiempo tampoco
settle(app)
check("el paso del tiempo tampoco lo reinicia", demo._last_touch == antes)
# ahora una persona
demo._place_fault("fantasma")
check("una persona SI lo reinicia", demo._last_touch > antes,
      f"{demo._last_touch - antes:+.1f} s")

print("\n[2] Todos los controles del visitante cuentan como tocar")
controles = [
    ("dibujar (soltar el raton)", lambda: demo._on_release(None)),
    ("elegir herramienta",        lambda: demo._set_tool("mancha")),
    ("boton de fallo",            lambda: demo._place_fault("offset")),
    ("boton Limpiar tela",        demo._on_clear_click),
    ("boton Pausar tela",         demo._toggle_pause),
    ("boton Modo auto",           demo._toggle_auto),
    ("deslizador de velocidad",   lambda: demo._on_speed(120.0)),
    ("cambiar estampado",         lambda: demo._on_motif(t("motif_" + MOTIFS[1]))),
    ("cambiar tejido",            lambda: demo._on_weave(t("weave_" + WEAVES[1]))),
    ("cambiar metodo de conteo",
     lambda: demo._on_count_method(t("count_m_" + CountingModule.METHODS[1]))),
    ("boton Reiniciar conteo",    demo._reset_counting),
]
for nombre, fn in controles:
    demo._last_touch = 0.0
    fn()
    settle(app, 1)
    check(f"{nombre} reinicia el reloj", demo._last_touch > 0.0)

# ── el caso malo: la vuelta automatica usa reset_demo y NO debe contar ───
demo._last_touch = 0.0
demo.reset_demo()
check("reset_demo() por dentro NO reinicia el reloj", demo._last_touch == 0.0,
      "si contara, la tela no se limpiaria nunca sola")

print("\n[2b] Y los botones DE VERDAD, no solo sus metodos")
# Lo de arriba llama a los manejadores. Un boton cableado al callback
# equivocado pasaria desapercibido —y el visitante pulsa botones, no
# metodos—, asi que aqui se pulsan los botones reales.


def boton(raiz, texto):
    """Primer CTkButton del arbol cuyo texto sea *texto*."""
    pendientes = [raiz]
    while pendientes:
        w = pendientes.pop(0)
        if isinstance(w, ctk.CTkButton):
            try:
                if w.cget("text") == texto:
                    return w
            except Exception:
                pass
        pendientes.extend(w.winfo_children())
    return None


botones_demo = [
    ("Limpiar tela", t("demo_clear")),
    ("herramienta Rayon",  t("demo_tool_rayon")),
    ("herramienta Mancha", t("demo_tool_mancha")),
    ("fallo Fantasma",     t("demo_fault_fantasma")),
    ("fallo Movido",       t("demo_fault_offset")),
    ("Reiniciar conteo",   t("demo_count_reset")),
]
for nombre, etiqueta in botones_demo:
    b = boton(demo, etiqueta)
    if b is None:
        check(f"existe el boton {nombre}", False, f"no encontrado: {etiqueta!r}")
        continue
    demo._last_touch = 0.0
    b.invoke()
    settle(app, 1)
    check(f"pulsar {nombre} reinicia el reloj", demo._last_touch > 0.0)

# estos dos se guardan como atributo. El de pausa no se busca por
# texto a proposito: en [2] ya se pulso y ahora pone "Reanudar".
demo._last_touch = 0.0
demo._pause_btn.invoke()
settle(app, 1)
check("pulsar Pausar/Reanudar reinicia el reloj", demo._last_touch > 0.0)
demo._last_touch = 0.0
demo._auto_btn.invoke()
settle(app, 1)
check("pulsar Modo auto reinicia el reloj", demo._last_touch > 0.0)

print("\n[3] Antes del plazo no pasa nada (el caso malo de la vuelta)")
demo._cal_left = 0
demo._auto = False
demo._refresh_auto_label()
demo._pattern.streak(100, 100, length=120, thickness=6)
demo._pattern.blob(300, 200, size=80)
sucios = len(demo._pattern.defects)
check("la tela esta sucia y el automatico apagado",
      sucios > 0 and not demo._auto, f"{sucios} defectos")
vueltas = demo._vueltas_solas
demo._last_touch = time.time()          # acaban de tocar
demo._idle_tick()
check("no vuelve antes de tiempo",
      demo._vueltas_solas == vueltas and not demo._auto
      and len(demo._pattern.defects) == sucios)

print("\n[4] Pasado el plazo vuelve sola, y deja todo como recien abierto")
demo._last_touch = time.time() - demo._idle_seconds - 1.0
demo._idle_tick()
settle(app)
check("cuenta una vuelta", demo._vueltas_solas == vueltas + 1)
check("el modo automatico esta encendido", demo._auto)
check("la tela quedo limpia", len(demo._pattern.defects) == 0,
      f"{len(demo._pattern.defects)} defectos")
check("el contador quedo a cero", demo._counting.total == 0,
      str(demo._counting.total))
check("el aviso esta en pantalla",
      demo._hint.cget("text") == t("demo_idle_back"), demo._hint.cget("text"))
check("el aviso dice algo distinto del texto normal",
      t("demo_idle_back") != t("demo_hint"))
demo._quitar_aviso()
check("el aviso se quita y vuelve el texto de siempre",
      demo._hint.cget("text") == t("demo_hint"), demo._hint.cget("text"))

print("\n[5] Si ya esta como debe, no hace nada")
vueltas = demo._vueltas_solas
demo._last_touch = time.time() - demo._idle_seconds - 1.0
demo._idle_tick()
check("tela limpia y automatico puesto: no cuenta vuelta",
      demo._vueltas_solas == vueltas)

print("\n[6] El reloj corre dentro del bucle, no solo si lo llamo yo")
llamadas = {"n": 0}
real_idle = demo._idle_tick
demo._idle_tick = lambda: (llamadas.__setitem__("n", llamadas["n"] + 1),
                           real_idle())[1]


def una_vuelta_del_bucle():
    """
    Ejecuta el cuerpo del bucle EXACTAMENTE una vez.

    `_loop` se reprograma con after(33), asi que dejarlo vivo mete vueltas
    de mas y el contador deja de medir lo que se cree.  Apagar `_running`
    justo despues hace que las pendientes salgan por la primera linea.
    """
    demo._running = True
    demo._loop()
    demo._running = False
    settle(app, 2)


llamadas["n"] = 0
demo._cal_left = 0
una_vuelta_del_bucle()
check("el bucle llama al reloj de inactividad", llamadas["n"] == 1,
      str(llamadas["n"]))
# caso malo: mientras calibra, el reloj no corre (la tela aun no es util)
llamadas["n"] = 0
demo._cal_left = 5
una_vuelta_del_bucle()
check("mientras calibra NO corre", llamadas["n"] == 0, str(llamadas["n"]))
demo._idle_tick = real_idle
demo._cal_left = 0

print("\n[7] La demo de conteo tambien sigue sola si la dejan parada")
app.navigate("nav_cdemo")
settle(app)
cd = app._screens["nav_cdemo"]
cd._running = False                     # el bucle no hace falta aqui
cd._on_auto_click()                     # alguien la detiene
check("queda detenida en un metodo", not cd._auto)
vueltas = cd._vueltas_solas
cd._last_touch = time.time()
cd._idle_tick()
check("no sigue antes de tiempo",
      not cd._auto and cd._vueltas_solas == vueltas)
cd._last_touch = time.time() - cd._idle_seconds - 1.0
cd._idle_tick()
settle(app)
check("pasado el plazo vuelve a avanzar sola", cd._auto)
check("cuenta una vuelta", cd._vueltas_solas == vueltas + 1)
check("y lo avisa", cd._sub_lbl.cget("text") == t("cdemo_idle_back"),
      cd._sub_lbl.cget("text"))
check("el aviso dice algo distinto del subtitulo",
      t("cdemo_idle_back") != t("cdemo_sub"))

print("\n[7b] Y los botones de la de conteo, de verdad")
for nombre, etiqueta in [("avanzar", "›"), ("retroceder", "‹")]:
    b = boton(cd, etiqueta)
    if b is None:
        check(f"existe el boton {nombre}", False, repr(etiqueta))
        continue
    cd._last_touch = 0.0
    b.invoke()
    settle(app, 1)
    check(f"pulsar {nombre} reinicia el reloj", cd._last_touch > 0.0)
cd._last_touch = 0.0
cd._auto_btn.invoke()
settle(app, 1)
check("pulsar Auto en la de conteo reinicia el reloj", cd._last_touch > 0.0)

print("\n[8] En la de conteo, avanzar solo tampoco cuenta como tocar")
cd._last_touch = 0.0
cd._saltar(1)                           # lo que hace el bucle al avanzar
check("el bucle avanzando NO reinicia el reloj", cd._last_touch == 0.0)
cd._on_saltar(1)                        # lo que hace el boton
check("el boton de avanzar SI lo reinicia", cd._last_touch > 0.0)
cd._last_touch = 0.0
cd._toggle_auto()                       # lo que hace la vuelta sola
check("_toggle_auto() por dentro NO reinicia el reloj", cd._last_touch == 0.0)
cd._on_auto_click()                     # lo que hace el boton
check("el boton de Auto SI lo reinicia", cd._last_touch > 0.0)

# ── cierre ────────────────────────────────────────────────────────────────
try:
    app._teardown_all_screens()
    app.update()
    app.destroy()
    print("\n  OK    cierre limpio")
except Exception as e:
    fails.append(f"cierre: {e}")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("INACTIVIDAD: VUELVE SOLA, Y SOLO UNA PERSONA REINICIA EL RELOJ")
