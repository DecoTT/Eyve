# Eyve

**Inspección visual con IA para manufactura — 100 % local, en tu PC.**

Eyve te deja entrenar tu propio modelo de visión artificial con tus propias piezas y
ponerlo a inspeccionar en minutos, sin enviar una sola imagen a internet y sin saber
nada de machine learning.

```
Captura  →  Etiquetado  →  Entrenamiento  →  Producción
```

<p align="center">
  <a href="https://github.com/DecoTT/eyve/releases/latest/download/Eyve-2.1-setup.zip"><b>⬇ Descargar Eyve 2.1 para Windows</b></a>
  &nbsp;·&nbsp; <a href="https://github.com/DecoTT/eyve/releases/latest">Notas de la versión y SHA-256</a>
</p>

---

## Qué hace

| Pantalla | Para qué |
|---|---|
| **Proyectos** | Un proyecto por pieza o línea: clases, imágenes, etiquetas, modelos y sesiones, todo en una carpeta tuya. |
| **Categorías** | Define qué detectar y si cada cosa es `OK`, `NOK` o `IGNORE`. |
| **Captura** | Fotos o video desde cámara USB, a 30 fps. |
| **Etiquetado** | Dibuja cajas sobre fotos, sobre un video pausado (barra de navegación) o congelando la cámara en vivo. |
| **Entrenamiento** | YOLOv8 local, CPU o GPU. 40 imágenes ≈ 5 min en CPU. |
| **Producción** | Tu modelo en tiempo real sobre cámara o video: OK / NOT OK, conteo de piezas, evidencia de defectos, log de sesión. |
| **Demo** | Una tela con patrón pasa frente a una cámara simulada: dibuja un defecto y mira a Eyve encontrarlo, seguirlo y contarlo. Sin cámara ni cableado. |
| **Módulos** | Inspecciones más allá de detectar: **conteo con 5 métodos** y polaridad de componentes. Arquitectura abierta para escribir el tuyo. |

Todo corre en tu máquina. Eyve usa internet solo para instalarse y para bajar los
pesos base del modelo la primera vez.

## Contar, de cinco maneras

"Contar" no significa lo mismo en una banda que en una charola. El módulo de conteo
trabaja sobre instancias seguidas entre frames, así que una pieza que parpadea en la
detección no se cuenta dos veces.

| Método | Cuenta | Ejemplo |
|---|---|---|
| **En pantalla** | lo que hay ahora en el encuadre | ¿van las 12 tortillas en la charola? |
| **Cruce de meta** | cada pieza al cruzar una línea, por sentido | banda transportadora: entran, salen, neto |
| **Zona** | al entrar a un área, al salir, o ambas | celda de trabajo, zona de carga |
| **Al aparecer** | cada pieza nueva, una vez | piezas que llegan por cualquier lado |
| **Al desaparecer** | cada pieza que se va, por el borde que elijas | piezas que alguien retira |

"En pantalla" y la ocupación de "Zona" aceptan un rango esperado: fuera de rango, el
frame se marca NOT OK. Eso convierte el conteo en un criterio de inspección.

## Actualizarse

Eyve avisa en la barra de estado cuando hay versión nueva. Un clic, y se actualiza solo:
descarga, verifica el SHA-256 publicado, respalda la versión anterior y reemplaza el
programa. **Tus proyectos, tus modelos entrenados y tu licencia no se tocan.** Se acabó
descomprimir carpetas y volver a cargar proyectos a mano.

## Instalar

1. Descarga [`Eyve-2.1-setup.zip`](https://github.com/DecoTT/eyve/releases/latest/download/Eyve-2.1-setup.zip) y descomprímelo.
2. Doble clic en **`setup.bat`** — detecta o instala Python, crea un entorno aislado y baja las dependencias (una sola vez, 10-30 min).
3. Doble clic en **`run.bat`**.

Requisitos: Windows 10/11 64-bit, 8 GB RAM (16 para entrenar), 10 GB libres, cámara USB
opcional. La guía completa viene dentro del ZIP (`README.md`).

## Licencia y niveles

Eyve es software libre bajo **[AGPL-3.0](LICENSE)**.

| Nivel | Incluye | Cómo |
|---|---|---|
| **Free** | Detección, conteo y registro de sesiones | Sin llave, solo instala |
| **Estudiante / Normal** | Plataforma completa | Llave por correo: <https://sbcgroup.com.mx/prueba-eyve/> |
| **Pro** | + módulos de inspección SBC (polaridad, flujo, serigrafía, patrones) | Misma tienda |

La política es **por honor**: Eyve nunca se bloquea. Sin llave sigues en Free; una llave
vencida conserva su nivel y solo avisa; la verificación es local y sin internet. Una
llave cubre 3 equipos, administrables en <https://sbcsuite.com.mx/store/cuenta/licencias>.
Quien saca provecho de Eyve en su planta y paga su licencia es quien mantiene el
proyecto abierto.

## Para desarrolladores

```bash
git clone https://github.com/DecoTT/eyve.git
cd eyve
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m eyve.main
```

- `eyve/ui/` — pantallas (CustomTkinter). `eyve/inference/` — YOLO worker, tracker
  persistente, motor de polaridad. `eyve/modules/` — módulos de inspección; para
  escribir uno nuevo implementa `InspectionModule` y regístralo en `MODULE_REGISTRY`.
  `eyve/core/updater.py` — actualización desde GitHub Releases.
- `eyve/demo/` — la tela sintética de la pantalla Demo. Para prepararla en la máquina
  del stand (una vez; el dataset tarda segundos, el entrenamiento ~45 min en CPU):

  ```bash
  python -m eyve.demo.train --out "projects/Demo_Textil" --frames 700 --epochs 40
  ```
- `PRD.md` — alcance, decisiones y notas técnicas del release (léelo antes de tocar
  la cámara: el Anexo A documenta varias trampas de DSHOW y Tkinter que ya nos costaron).
- `Release/build_release.ps1` — empaqueta y valida el instalador.

Los reportes de errores van en [Issues](https://github.com/DecoTT/eyve/issues) con el
log de `%USERPROFILE%\.eyve\eyve_run.log`.

---

Hecho por [SBC Group](https://sbcgroup.com.mx) en México. Dependencias de terceros en
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
