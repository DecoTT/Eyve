# -*- coding: utf-8 -*-
"""
Los dialogos se VEN aunque Produccion este corriendo.

Con Produccion a `prod_fps_cap: 0` (bucle a after(1)), cualquier
CTkToplevel —los cuatro de components/dialogs.py, Ajustes— se quedaba
invisible y ask_string / ask_confirm bloqueaban para siempre.  La causa:
el constructor de CTkToplevel en Windows hace withdraw() + update(), y
update() no vuelve mientras haya algo listo.  Con un bucle a 1 ms y trabajo
de repintado entre vuelta y vuelta (en Tk el repintado va como callback
"idle"), cuando termina el repintado la siguiente vuelta ya esta vencida:
el constructor no termina nunca.  Ver eyve/ui/ventanas.py.

La prueba monta eso mismo: un bucle a after(1) que deja 5 ms de trabajo en
reposo por vuelta.  Y el CONTROL: sin ese trabajo el dialogo aparecia
siempre, incluso con el fallo — por eso los primeros arneses no lo
reproducian y no hay que fiarse de uno que no lo tenga.

Cada caso corre en su propio proceso: sin el arreglo el constructor se
queda colgado hasta que el bucle se detiene, y no debe arrastrar al
siguiente caso.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def caso(que: str, idle_ms: float) -> None:
    """Corre un caso y escribe en stdout: visible titulo barra_oscura."""
    import ctypes
    import time
    import customtkinter as ctk
    from eyve.i18n import set_language
    from eyve.ui import theme as T
    T.apply("dark")
    set_language("es")
    from eyve.ui.components import dialogs

    root = ctk.CTk()
    root.geometry("700x450+40+40")
    st = {"t0": None, "visto": None, "titulo": None, "oscura": None,
          "fin": False}

    def trabajo_en_reposo():
        time.sleep(idle_ms / 1000.0)

    def toplevels():
        out = []
        pend = list(root.winfo_children())
        while pend:
            w = pend.pop()
            if isinstance(w, ctk.CTkToplevel):
                out.append(w)
            pend.extend(w.winfo_children())
        return out

    def oscura(w):
        hwnd = ctypes.windll.user32.GetParent(w.winfo_id())
        v = ctypes.c_int(-1)
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd, 20, ctypes.byref(v), ctypes.sizeof(v))
        return v.value == 1

    def tick():
        if idle_ms > 0:
            root.after_idle(trabajo_en_reposo)
        if st["t0"] is not None:
            dt = time.perf_counter() - st["t0"]
            for w in toplevels():
                if st["visto"] is None and w.winfo_viewable():
                    st["visto"] = dt
                    st["titulo"] = w.title()
                if st["visto"] is not None and dt - st["visto"] > 0.3:
                    # medio segundo despues: ya debio pintarse la barra
                    if st["oscura"] is None:
                        st["oscura"] = oscura(w)
                    w.destroy()        # desbloquea ask_string/ask_confirm
            if dt > 2.0:
                st["fin"] = True
        if not st["fin"]:
            root.after(1, tick)        # como Produccion con prod_fps_cap: 0
        else:
            root.after(300, root.destroy)

    def abrir():
        st["t0"] = time.perf_counter()
        if que == "show_error":
            dialogs.show_error(root, "prueba")
        elif que == "show_info":
            dialogs.show_info(root, "prueba")
        elif que == "ask_confirm":
            dialogs.ask_confirm(root, "prueba")
        elif que == "ask_string":
            dialogs.ask_string(root, "prueba", title="Titulo puesto")
        elif que == "settings":
            from eyve.ui.screens.settings_popup import SettingsPopup
            SettingsPopup(root, app=None)

    root.after(100, tick)
    root.after(400, abrir)
    root.after(8000, root.destroy)          # red de seguridad
    root.mainloop()
    print(f"RESULTADO {st['visto'] is not None and st['visto'] < 1.0} "
          f"{st['titulo']!r} {st['oscura']}")


if len(sys.argv) > 1 and sys.argv[1] == "--caso":
    caso(sys.argv[2], float(sys.argv[3]))
    sys.exit(0)

fails = []


def check_true(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


def correr(que, idle_ms):
    r = subprocess.run([sys.executable, __file__, "--caso", que, str(idle_ms)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=60)
    linea = next((l for l in r.stdout.splitlines()
                  if l.startswith("RESULTADO")), "RESULTADO False None None")
    _, visible, *resto = linea.split(" ", 2)
    titulo, oscura = (resto[0].rsplit(" ", 1) + ["None"])[:2]
    return visible == "True", titulo, oscura == "True", r.stderr


print("\n[1] Control: sin trabajo entre vueltas, el dialogo aparece")
# Esto pasaba tambien con el fallo. Si fallara, el arnes esta roto.
vis, titulo, _, err = correr("show_error", 0)
check_true("show_error con el bucle a 1 ms y sin repintado", vis, titulo)

print("\n[2] Con Produccion corriendo (after(1) + 5 ms de repintado)")
for que, titulo_esperado in (("show_error", "'Error'"),
                             ("show_info", None),
                             ("ask_confirm", None),
                             ("ask_string", "'Titulo puesto'"),
                             ("settings", None)):
    vis, titulo, oscura, err = correr(que, 5)
    check_true(f"{que}: se ve antes de 1 s", vis, f"titulo={titulo}")
    # El titulo es la huella del fallo: se quedaba en "CTkToplevel"
    # porque el constructor no terminaba y title() nunca corria.
    check_true(f"{que}: con su titulo, no 'CTkToplevel'",
               "CTkToplevel" not in titulo and titulo != "None", titulo)
    if titulo_esperado:
        check_true(f"{que}: el titulo que se le paso", titulo == titulo_esperado,
                   titulo)
    # Sin el withdraw()+update() la barra oscura se pone de otro modo; que
    # se siga poniendo.
    check_true(f"{que}: barra de titulo oscura", oscura)
    if not vis and err.strip():
        print("       stderr:", err.strip().splitlines()[-1][:160])

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("DIALOGOS: TODO PASO")
