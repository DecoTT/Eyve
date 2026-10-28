# Eyve 2.1 — checklist de release y distribución

> Decisión del dueño (18-sep-2026): **GitHub Releases es el origen único del
> instalador**. La tienda (correo de licencia, thank-you y cuenta del cliente)
> apunta a una URL fija de `releases/latest/download/…`. Ni Mega ni el bucket
> de Supabase para el público: sin caducidad, sin egress del plan micro, con
> versión y checksum a la vista. El bucket `eyve-builds` queda como respaldo
> privado (sigue funcionando si algún día se sube algo ahí).

## 0. Nombres fijos (no cambiar sin actualizar la tienda)

| Qué | Valor |
|---|---|
| Repo público | `https://github.com/DecoTT/Eyve` *(existente; el prototipo 2025 vive en la rama `prototype-2025`)* |
| Asset del release | `Eyve-2.1-setup.zip` (el ZIP de `Release\build_release.ps1`, renombrado) |
| URL fija de descarga | `https://github.com/DecoTT/Eyve/releases/latest/download/Eyve-2.1-setup.zip` |
| Checksum | `SHA256SUMS.txt` junto al asset |
| Tag | `v<version>` (= `eyve/__init__.py: __version__`); hoy `v2.1.1` |

`releases/latest/download/<asset>` siempre resuelve a la última versión publicada
(no pre-release), así que la tienda no se toca al sacar 2.1.1.

## 1. Antes de empaquetar

- [x] Los 4 bugs pendientes cerrados y commiteados en `2.1-dev` (rama con git; `2.1` es el respaldo congelado, no se toca).
- [x] `eyve/__init__.py` → `__version__ = "2.1.0"` (es lo que Eyve manda en `version_eyve` al activar; en el admin se ve por equipo).
- [x] `requirements.txt` incluye `PyJWT[crypto]` (licencias) — ya está.
- [x] Quitar el sufijo `[dev]` del título (`eyve/ui/app.py`, `self.title(...)`).
- [x] `Release\build_release.ps1`: `$Version = "2.1.0"`, produce directamente `Eyve-2.1-setup.zip` + `SHA256SUMS.txt`, valida los `.bat` (CRLF/ASCII/operadores sin escapar) y aborta si el asset pasa de 2 GB. Solo copia `eyve/` (sin `__pycache__`/`.pyc`/`.log`) + archivos raíz; `projects/` va vacío; `yolov8n.pt` y `HANDOFF_FABLE.md` ya no están rastreados.
- [x] Prueba fría (18-sep, carpeta limpia desde el ZIP final): `setup.bat` completo en 129 s (4/4, `run.bat` generado con `%~dp0`); `run.bat` abre **"Eyve 2.1"** y escribe `%USERPROFILE%\.eyve\eyve_run.log`.
- [ ] **Pendiente (necesita una llave real):** Settings → Licencia… → pegar llave Free → "Comprobar ahora" = *Equipo registrado 1 de 3*. Sin red también debe abrir.
- [x] `Release\README.md`: sin "Beta"; sección **Licencia** con niveles, activación, 3 equipos y política por honor (Free sin llave; pegar la llave del correo; por honor, nada bloquea; enlace a `https://sbcsuite.com.mx/store/cuenta/licencias` para liberar equipos).
- [x] README raíz del repo público (`README.md`): qué es, AGPL-3.0, cómo se compra la licencia (`https://sbcgroup.com.mx/prueba-eyve/`), y que la política de licencia es por honor (PRD §14).

## 2. Empaquetar

```powershell
cd "D:\Desarrollo\Claude Code\Eyve\2.1-dev\Release"
.\build_release.ps1     # deja Eyve-2.1-setup.zip + SHA256SUMS.txt; no hay que renombrar nada
```

- [x] Tamaño del ZIP < 2 GB (0.17 MB; el build lo verifica) (límite por asset en GitHub). Debe ser chico: `setup.bat` baja Python y las dependencias; los pesos `yolov8n.pt` se descargan en la primera ejecución (ya lo hace la app).
- [x] Abrir el ZIP y confirmar estructura plana (el build lo verifica con 8 checks) (`setup.bat`, `run.bat`, `eyve/`, `requirements.txt` en la raíz) — el bug del build corrupto ya pasó una vez (commit `53dd16e`).

## 3. Publicar en GitHub

- [x] Auditoría del historial (18-sep): sin claves privadas. El único secret que aparece es el HMAC de la licencia **vieja**, retirado en `ad9057f` al migrar a JWT EdDSA — la app ya no acepta nada firmado con él. La clave Supabase del historial es `role: anon` (pública por diseño). `yolov8n.pt` (6 MB, peso público de Ultralytics) y `HANDOFF_FABLE.md` (notas internas, sin secretos) siguen en commits viejos; no vale la pena reescribir el historial por ellos.
- [x] Repo público: se reutilizó **`DecoTT/Eyve`** (ya existía con el prototipo de 2025). El prototipo quedó intacto en la rama `prototype-2025`; `main` se reemplazó con Eyve 2.1 (force push, 18-sep-2026). GitHub no distingue mayúsculas: `DecoTT/eyve` y `DecoTT/Eyve` son el mismo repo.
- [x] `origin` = `https://github.com/DecoTT/Eyve.git`, `main` subido (`86fdbee`).
- [x] Tag `v2.1.0` subido.
- [x] Release **Eyve 2.1.0** publicado con `gh release create --latest` (no pre-release, no draft), assets `Eyve-2.1-setup.zip` (180 192 B) + `SHA256SUMS.txt`: <https://github.com/DecoTT/Eyve/releases/tag/v2.1.0>
- [x] URL fija verificada (18-sep): `https://github.com/DecoTT/Eyve/releases/latest/download/Eyve-2.1-setup.zip` → 302 → 302 → **200**; el archivo descargado coincide byte a byte con `SHA256SUMS.txt` (`8060b705…7758`).

## 4. Prender la descarga en la tienda (sbcsuite)

Todo está preparado y apagado; se prende con dos valores iguales:

- [x] **Secret** `EYVE_DOWNLOAD_URL` (21-sep) = la URL fija, en Supabase Dashboard → Edge Functions → Secrets. La function `licencias-emitir` lo lee al vuelo (no hay que redesplegar): desde ese momento el correo de licencia lleva el botón **Descargar Eyve 2.1** en vez de "se está publicando".
  **Revisado por el dueño el 5-oct tras publicar la 2.1.1** (no verificado desde el repo: el secret no se puede leer desde fuera de Supabase).
- [x] **Front** (PR #27, 21-sep): `src/modules/store/lib/eyve.ts` → `EYVE_DOWNLOAD_URL = '<la misma URL>'`; PR a `main`, Netlify publica. Con eso aparece el botón en la thank-you (`LicenciasEyve.tsx`) y en `/store/cuenta/licencias`.
  **Verificado el 5-oct en el bundle que sirve sbcsuite.com.mx**, no sólo en el repo: `assets/index-BcMDYLvn.js` contiene la URL fija, y esa URL resuelve a `/v2.1.1/` con los bytes exactos del build.

### Qué pasó al publicar la 2.1.1 (5-oct)

**La tienda cambió sola.** Como la URL apunta a `releases/latest/download/` y
el nombre del asset no cambia entre versiones, no hubo que tocar ni el front
ni el secret: dashboard y thank-you empezaron a servir 2.1.1 en cuanto el
release pasó a ser `latest`. Esa era justo la razón de elegir esta forma de
distribuir (§0).

Dos condiciones de visibilidad, por si alguien reporta que no ve el botón:

- en `/store/cuenta/licencias` sólo aparece si la cuenta tiene **al menos una
  licencia** (`lics.length > 0`);
- en la thank-you aparece siempre que la URL esté puesta; si faltara, sale el
  texto "se está publicando".

**Cuidado con el camino de reserva del correo.** Si `EYVE_DOWNLOAD_URL`
estuviera vacío, `licencias-emitir` genera una URL firmada de 72 h al bucket
`eyve-builds` apuntando a `eyve-2.1/Eyve-2.1-setup.exe` — un **`.exe`**, y lo
que se publica es un **`.zip`**. O sea: si el secret se borra, el correo sale
con un enlace a un archivo que no existe, no con un error visible.

- [ ] **Lo único sin comprobar de punta a punta**: pedido Free desde
      `https://sbcgroup.com.mx/prueba-eyve/` con un correo propio → que el
      correo traiga el botón y la thank-you también. Borrar el pedido de
      prueba después. Es la única forma de confirmar el camino del correo,
      porque el secret no se puede leer desde fuera.
- [ ] Opcional: en `/prueba-eyve/` (HostGator, fuera del repo) un enlace "¿Ya tienes llave? Descarga Eyve 2.1" al pie, con la misma URL.
- [ ] Marcar hecho en `qr-lead-connect/docs/licencias-eyve.md` §6 ("Subir el instalador…") y en `ESTADO.md`.

## 5. Después del release

- [ ] Vigilar `/store/admin/licencias`: "Activadas alguna vez" contra "Emitidas" dice si la gente llega a pegar la llave (embudo-eyve.md §08).
- [ ] Versión siguiente (2.1.x): subir `__version__`, tag nuevo, release nuevo con el **mismo nombre de asset**. La URL fija no cambia; la tienda no se toca.
- [ ] Renovaciones y cortesías: desde el admin (Emitir manual / Renovar); Eyve recibe la llave nueva sola en su check-in semanal.

## 6. Versión 2.1.1 — PUBLICADA (5-oct-2026)

`__version__ = "2.1.1"` y `$Version = "2.1.1"` en `build_release.ps1`. **El
nombre del asset no cambia** (`$AssetName` es fijo y no depende de
`$Version`), así que la URL de la tienda no se toca.

Motivo del release: desde la 2.1.0 hay 26 commits y el actualizador **no
tenía a qué actualizar** — con `__version__` en 2.1.0 contestaba "ya tienes
la última versión". Publicar lo vuelve demostrable.

Qué entra, de cara al usuario:

- **Modo kiosco** (`F11` entra y sale, `Esc` sale) en las dos pantallas de
  demo, sin barra lateral ni barra de estado.
- **Vuelta sola a modo automático** en la demo textil tras 3 minutos sin
  que nadie toque, con la tela limpia; y la demo de conteo vuelve a
  avanzar sola si la dejan detenida en un método.
- **Botón de "empezar de cero"** en la demo textil, con recalibrado del
  módulo Patrón.
- **La demo va al doble y medio de fps**: generar la tela pasa de ~31 a
  6.2 ms (componer sólo donde hay defectos, `cv2.multiply` y desplazar por
  rebanadas) y el repintado reutiliza el PhotoImage y deja de ampliar el
  frame más allá de su tamaño nativo. En pantalla completa a 3440×1440 va
  de **~12.5 a ~31 fps**, y en ventana de ~21 a 40. La tela se ve igual:
  la diferencia es de 0.87 niveles sobre 255, siempre hacia arriba, por
  pasar de truncar a redondear.
- **Ajuste nuevo**: Settings → Rendimiento → "Demo: llenar la pantalla"
  (`demo_fill_screen`, por defecto apagado). Encendido la demo se ve más
  grande en monitores grandes, a cambio de la mitad de los fps. En
  1920×1080 no cambia nada.
- **El actualizador vuelve a funcionar.** Hasta 2.1.0 pedía la API de
  GitHub con `Accept: application/octet-stream`, que la API rechaza con
  **415**; como `check()` se traga los errores a propósito, contestaba "no
  hay actualización" para siempre y en silencio. **Consecuencia: quien ya
  tenga 2.1.0 instalado NO verá la 2.1.1 solo** — lleva el fallo dentro.
  Esa primera actualización hay que hacerla a mano; de 2.1.1 en adelante
  ya es automática.
- Antes de esto: módulo de conteo con 5 métodos, módulo Patrón, demo
  textil, demo de conteo y el propio actualizador.

Checklist de esta versión:

- [x] `eyve/__init__.py` → `__version__ = "2.1.1"`.
- [x] `Release\build_release.ps1` → `$Version = "2.1.1"`; `$AssetName` sin
      tocar.
- [x] Suite completa en verde: 22 de 22, incluida `test_demo_e2e.py` (la
      que carga YOLO).
- [x] `cd Release && .\build_release.ps1` → `Eyve-2.1-setup.zip` +
      `SHA256SUMS.txt`, con los 8 checks de estructura y la validación de
      los `.bat`.
- [x] `main` subido (`118de77`), tag `v2.1.1` subido y release publicado
      con `gh release create --latest` (no borrador, no pre-release):
      <https://github.com/DecoTT/Eyve/releases/tag/v2.1.1>
- [x] Verificado contra lo PUBLICADO, no contra lo local:
      - la API dice que `latest` es `v2.1.1`, sin borrador ni pre-release,
        con los dos assets;
      - la URL fija hace `302 → /v2.1.1/ → 302 → CDN → 200`, y los
        259 612 bytes que bajan son **byte a byte** los construidos, con el
        `SHA256SUMS.txt` publicado cuadrando
        (`d2201ef0d2fff801…`);
      - bajado el ZIP a una carpeta limpia: estructura plana, `projects/`
        vacía, el código importa y dice `2.1.1` y lleva el `Accept`
        arreglado.
- [x] El actualizador, contra la API real: desde 2.1.0 diría que **sí** hay
      actualización y bajaría `.../v2.1.1/Eyve-2.1-setup.zip`; desde 2.1.1
      dice que no (no se ofrece a sí misma).
- [ ] **Pendiente y sabido**: una instalación 2.1.0 **de verdad no verá la
      2.1.1 sola**, porque lleva dentro el fallo del 415 (comprobado: su
      consulta sigue dando 415 hoy). Esa primera actualización hay que
      hacerla a mano desde la URL fija. De 2.1.1 en adelante ya es
      automática. Queda por probar el ciclo completo instalando una 2.1.1
      y publicando una 2.1.2 de prueba.
- [ ] Mirar de dónde sale `update_last_installed: "2.2.0"` en el
      `~/.eyve/config.json` de la máquina de desarrollo: no corresponde a
      ninguna versión publicada.
- [ ] Pendiente de antes, sin hacer: Settings → Licencia… con una llave
      real (necesita una llave de verdad).

## 7. Versión 2.1.2 — PUBLICADA (16-oct-2026)

`__version__ = "2.1.2"` y `$Version = "2.1.2"` en `build_release.ps1`. **El
nombre del asset no cambia**, así que la tienda no se toca: dashboard,
thank-you y correo siguen apuntando a `releases/latest/download/`.

Seis commits desde la 2.1.1. Qué entra, de cara al usuario:

- **Jobs por SKU.** Los parámetros de producción —qué módulos van
  encendidos, con qué método, la geometría del conteo, la sensibilidad y
  la calibración del Patrón— se guardan por SKU dentro del proyecto
  (`<proyecto>/jobs/<SKU>.yaml`) y se reabre solo el último usado. Antes
  había que ponerlo todo de cero en cada arranque.
- **El módulo Patrón, arreglado en la demo.** No marcaba el fantasma en
  ningún estampado y en rayas no marcaba nada: el ángulo de la retícula
  salía mal y la calibración se comía un frame de la tela anterior. Ahora
  las 12 combinaciones de estampado y fallo se marcan en el 82-99 % de los
  frames, con 0 falsas alarmas en 2560 frames de tela limpia.
- **Esconder las demos** (Settings → Apariencia). Las quita del menú y las
  descarga de memoria; apagado de fábrica.

Checklist de esta versión:

- [x] `eyve/__init__.py` → `__version__ = "2.1.2"`.
- [x] `Release\build_release.ps1` → `$Version = "2.1.2"`; `$AssetName` sin
      tocar.
- [x] Suite completa en verde: 27 de 27, incluida `test_demo_e2e.py`.
- [x] `main` subido a `origin` (`9b4763b`).
- [x] `main` subido, tag `v2.1.2` subido y release publicado con
      `gh release create --latest`:
      <https://github.com/DecoTT/Eyve/releases/tag/v2.1.2>
- [x] Verificado contra lo PUBLICADO: `latest` es `v2.1.2`, sin borrador ni
      pre-release, con los dos assets; la URL fija hace
      `302 → /v2.1.2/ → CDN → 200` y los 273 759 bytes que bajan son
      **byte a byte** los construidos, con el `SHA256SUMS.txt` publicado
      cuadrando (`4ca4d54d0520f88f…`).
- [x] **El actualizador, probado por fin de verdad** — era lo que arrastraba
      esta lista desde septiembre, imposible hasta ahora porque la 2.1.0
      llevaba dentro el fallo del 415. Se bajó la **2.1.1 tal como está
      publicada**, se extrajo en una carpeta limpia y se corrió **su**
      actualizador: ve la 2.1.2, dice que sí hay actualización y apunta al
      asset de `/v2.1.2/`. La instalación en sí (descarga, respaldo,
      reemplazo y conservar `projects/`) no se ejecutó ahí porque
      reemplazaría la carpeta; la cubre `tests/test_updater.py`.
- [ ] Queda, para cuando toque: hacer la actualización completa a mano en
      una máquina de pruebas y confirmar que `projects/` sobrevive.

### Diálogos invisibles con Producción corriendo — arreglado después de 2.1.2

Los cuatro diálogos de `eyve/ui/components/dialogs.py`, y cualquier
`CTkToplevel` (Ajustes incluido), se quedaban invisibles con Producción a
`prod_fps_cap: 0`. La causa era el `withdraw()` + `update()` que hace el
constructor de `CTkToplevel` de CustomTkinter en Windows. Arreglado en
`eyve/ui/ventanas.py`, con `tests/test_dialogos.py`. Detalle en
`HANDOFF_EXPO.md` §6. Afecta a la 2.1.2 publicada: entra en la siguiente.

Lo que de verdad rompía era **abrir Ajustes con Producción corriendo**. La
nota anterior decía que también bloqueaba crear y borrar una categoría;
no es cierto: esa pantalla no usa los diálogos.

## 8. Versión 2.1.3

`__version__ = "2.1.3"` y `$Version = "2.1.3"` en `build_release.ps1`. **El
nombre del asset no cambia**, así que la tienda no se toca: dashboard,
thank-you y correo siguen apuntando a `releases/latest/download/`.

Una sola cosa, y es la razón de sacarla tan pegada a la 2.1.2: **con
Producción corriendo no se podía abrir Ajustes**. Ni Ajustes ni ningún otro
`CTkToplevel` —los cuatro diálogos de `dialogs.py` incluidos—: la ventana se
creaba, agarraba el foco y se quedaba invisible. La causa no era de Eyve,
sino del constructor de `CTkToplevel` de CustomTkinter 6.0 en Windows, que
hace `withdraw()` + `update()`; con `prod_fps_cap: 0` (que es `after(1)`)
ese `update()` no vuelve nunca y el constructor no termina. Arreglado en
`eyve/ui/ventanas.py`, que pinta la barra oscura sin esconder la ventana ni
llamar a `update()`.

Quien tenga la 2.1.2 instalada tiene el fallo. Por eso esta versión.

Checklist de esta versión:

- [x] `eyve/__init__.py` → `__version__ = "2.1.3"`.
- [x] `Releaseuild_release.ps1` → `$Version = "2.1.3"`; `$AssetName` sin
      tocar.
- [x] Suite completa en verde: 28 de 28, con `test_dialogos.py` dentro.
- [x] Mirado en la app, no sólo en las pruebas: con cámara y modelo reales y
      Producción a `after(1)`, sin el parche el engrane no abre nada; con él
      Ajustes sale con su título y la barra oscura.
- [ ] Tag `v2.1.3`, push del tag y `gh release create v2.1.3 --latest` con
      los dos assets.
- [ ] Sigue pendiente de la 2.1.2: la actualización completa a mano en una
      máquina de pruebas, confirmando que `projects/` sobrevive. Ahora se
      puede hacer de verdad, porque la 2.1.2 instalada ya sabe actualizarse.

## 9. Versión 2.1.4

`__version__ = "2.1.4"` y `$Version = "2.1.4"` en `build_release.ps1`. **El
nombre del asset no cambia**, así que la tienda no se toca.

Qué entra, de cara al usuario:

- **Módulo Ensamble (Pro).** Inspección de ensambles por partes. El modelo
  detecta la base y cada parte; con una pieza buena, Eyve aprende dónde va
  cada parte respecto a la base (la pieza puede quedar en cualquier lugar del
  encuadre). Si falta una parte, es rechazo; si está fuera de posición, se
  pide corregir y se vuelve a inspeccionar. Juzga por pieza, no por cuadro:
  se pausa con la mano del operador (color del guante) o con movimiento, y
  cuenta cada pieza una vez al salir como OK, corregida o rechazo.
- **Instancias persistentes en el Ensamble.** Juzga sobre las cajas del
  tracker: una parte que el detector pierde 1-3 cuadros no parpadea.
- **Fuente por URL.** Producción acepta una cámara IP (http/rtsp) como
  fuente del job y se reconecta si el stream se cae.
- **Proyecto copiado a otra PC.** Si la ruta guardada del modelo no existe,
  se usa el `models/best.pt` del proyecto.
- Categorías: se guardan todas como OK, con aviso.

Checklist de esta versión:

- [x] `eyve/__init__.py` → `__version__ = "2.1.4"`.
- [x] `Releaseuild_release.ps1` → `$Version = "2.1.4"`; `$AssetName` sin tocar.
- [x] Suite completa en verde: 32 de 32, con `test_demo_e2e.py`, `test_assembly_module.py` y `test_proyecto_copiado.py`.
- [x] Build y prueba fría desde el ZIP (23-oct): `setup.bat` en una carpeta limpia en 146 s, `run.bat` abre «Eyve 2.1» y el log dice 2.1.4; con el venv de esa instalación pasan `test_assembly_module.py` y `test_proyecto_copiado.py`. El ZIP no nombra a ningún cliente.
- [x] Push de `main` (`c2af97c`), tag `v2.1.4` y release publicado con `gh release create --latest`: <https://github.com/DecoTT/Eyve/releases/tag/v2.1.4>
- [x] Verificado contra lo PUBLICADO: `latest` es `v2.1.4`, sin borrador ni pre-release, con los dos assets; lo que baja la URL fija es byte a byte lo construido (`cf1139c5…`, cuadra con el `SHA256SUMS.txt` publicado); el actualizador de la 2.1.3 publicada ve la 2.1.4 y apunta a su asset.

## 10. Versión 2.1.5

`__version__ = "2.1.5"` y `$Version = "2.1.5"` en `build_release.ps1`. **El
nombre del asset no cambia**, así que la tienda no se toca.

Qué entra, de cara al usuario:

- **Paquete de job (`.eyvejob`).** En Producción, «Exportar job para otra
  PC…» empaqueta el modelo, las clases y los parámetros del job (módulos,
  referencia, tolerancias, tracker). En Inicio, «Importar Job» lo abre en
  otra PC como proyecto nuevo, en Producción, con la fuente en la cámara 0.
  Nunca escribe encima de un proyecto existente, verifica el modelo por
  sha256 y solo extrae los archivos que espera.
- **Licencia vencida o revocada: marca de agua.** Por honor nada se
  bloquea, pero Producción lo muestra en pantalla con cualquier job, como
  Windows sin activar.

Checklist de esta versión:

- [x] `eyve/__init__.py` → `__version__ = "2.1.5"`.
- [x] `Releaseuild_release.ps1` → `$Version = "2.1.5"`; `$AssetName` sin tocar.
- [x] Suite completa: 33 de 34 en verde. `test_dialogos.py` («settings: se ve
      antes de 1 s») es una prueba de tiempo que falla igual con el código de
      la 2.1.4 publicada en esta máquina cargada (Resolve abierto), y la 2.1.5
      la pasa en una de dos corridas: inestable, no regresión.
- [x] Prueba fría desde el ZIP (26-oct): `setup.bat` en carpeta limpia en
      206 s, `run.bat` abre «Eyve 2.1» y el log dice 2.1.5. Vuelta completa
      del paquete: «Estación interactiva» del proyecto del minisplit exportada
      desde la instalación de desarrollo e importada con «Importar Job» en la
      limpia (su Python y su código): abre en Producción con el job cargado,
      las 3 clases, el modelo en su carpeta nueva, la referencia del Ensamble
      y la fuente en la cámara 0; el modelo importado detecta base 1.00,
      antena 0.97 y etiqueta 0.95. Marca de licencia vencida verificada en la
      limpia. Con su venv pasan test_job_package, test_licencia_marca,
      test_assembly_module y test_proyecto_copiado. Ni el ZIP ni el paquete
      nombran a ningún cliente.
- [ ] Push de `main`, tag `v2.1.5` y `gh release create v2.1.5 --latest`.
- [ ] Verificado contra lo publicado, y el actualizador de la 2.1.4 la ve.

## 11. Versión 2.1.6 — PUBLICADA (28-oct-2026)

`__version__ = "2.1.6"` y `$Version = "2.1.6"` en `build_release.ps1`. **El
nombre del asset no cambia**, así que la tienda no se toca.

La 2.1.5 nunca se publicó: la 2.1.6 es la primera versión pública con todo
lo de la 2.1.5 (paquete `.eyvejob`, marca de agua de licencia) más esto.

Qué entra, de cara al usuario:

- **La demo textil trae su modelo.** `build_release.ps1` copia
  `projects/Demo_Textil/models/best.pt` (6 MB) a `eyve/demo/modelo/best.pt`
  y el build FALLA si no está. Va dentro de `eyve/` y no en `projects/`
  porque el actualizador nunca toca `projects/`: quien actualice desde una
  versión anterior también lo recibe. Antes, una instalación descargada
  abría la demo con «Falta el modelo de la demo».
- **Patrón es Pro**, como Polaridad y Ensamble. En Producción la casilla se
  desactiva sin Pro y calibrar no lo enciende; un job que lo trae encendido
  corre con marca de agua. En la demo textil, que abre para todos, YOLO
  marca rayón y mancha; sin Pro, Patrón no corre, su recuadro dice
  «Requiere licencia Pro» y el modo automático solo pone defectos de YOLO.
- **Patrón en tejido canasta.** El periodo se estimaba mal en diamantes con
  canasta y no se marcaba ni un fantasma; ahora sí. Detalle en
  `HANDOFF_EXPO.md` §7.

Checklist de esta versión:

- [x] `eyve/__init__.py` → `__version__ = "2.1.6"`.
- [x] `Release\build_release.ps1` → `$Version = "2.1.6"`; `$AssetName` sin tocar.
- [x] `tests/test_patron_pro.py`: Producción y demo con y sin Pro, el job
      con marca de agua y el orden de búsqueda del modelo. 10 mutaciones,
      las 10 detectadas.
- [x] Suite completa: 35 de 35 en verde (27-oct, y otra vez el 28-oct con
      el arreglo del canasta).
- [x] ZIP final (28-oct): 5.67 MB, 78 archivos, SHA-256
      `1024b1162a73f313420f43cfa7ea423ad09092d466e4ec7dab5d413093b34e10`;
      lleva el arreglo del canasta, Patrón Pro y el modelo idéntico al de
      `projects/`.
- [x] El build falla si falta el modelo: probado con el script en una
      carpeta sin `projects/`, sale con código 1 y no genera ZIP.
- [x] Prueba fría desde el ZIP (27-oct): ZIP de 5.67 MB, el `best.pt` de
      dentro idéntico por sha256 al de `projects/`; `setup.bat` en carpeta
      limpia en 212 s; con su propio Python, sin proyecto abierto, la Demo
      carga `eyve/demo/modelo/best.pt` y YOLO marca rayón y mancha en 2.3 s.
      Esta máquina tiene licencia Pro: el caso Free lo cubre
      `test_patron_pro.py`.
- [x] Push de `main`, tag `v2.1.6` y `gh release create v2.1.6 --latest`
      (28-oct).
- [x] Verificado contra lo publicado: el ZIP bajado de
      `releases/latest/download/Eyve-2.1-setup.zip` da el mismo SHA-256
      que el local y que `SHA256SUMS.txt`; la API marca v2.1.6 como latest,
      sin draft ni prerelease, con los dos assets. Con esa respuesta,
      `is_newer` se la ofrece a una 2.1.4 y no a una 2.1.6.
- [ ] Falta ver el aviso en una 2.1.4 instalada de verdad: aquí el Python
      no valida el certificado de GitHub (`CERTIFICATE_VERIFY_FAILED`,
      mientras `curl` sí entra), así que `updater.check()` no se pudo
      correr de punta a punta en esta máquina.

## Lo que NO hay que hacer

- No poner el ZIP en Mega ni en el bucket público: un enlace suelto no tiene versión ni checksum y hay que reemplazarlo a mano cada vez.
- No usar URLs firmadas de 72 h para el instalador: generan soporte ("ya no me abre el enlace").
- No cambiar el nombre del asset entre versiones sin cambiar también `EYVE_DOWNLOAD_URL` y `lib/eyve.ts`.
