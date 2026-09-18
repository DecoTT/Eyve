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
| Repo público | `https://github.com/DecoTT/eyve` *(crear en el paso 3; si se llama distinto, cambia las dos URLs de abajo)* |
| Asset del release | `Eyve-2.1-setup.zip` (el ZIP de `Release\build_release.ps1`, renombrado) |
| URL fija de descarga | `https://github.com/DecoTT/eyve/releases/latest/download/Eyve-2.1-setup.zip` |
| Checksum | `SHA256SUMS.txt` junto al asset |
| Tag | `v2.1.0` (= `eyve/__init__.py: __version__`) |

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
- [ ] Crear el repo público `DecoTT/eyve` (AGPL-3.0; `LICENSE` y `THIRD_PARTY_NOTICES.md` ya están). Rama principal `main` (ya renombrada localmente).
- [ ] `git remote add origin https://github.com/DecoTT/eyve.git && git push -u origin main`
- [ ] `git tag -a v2.1.0 -m "Eyve 2.1.0" && git push origin v2.1.0`
- [ ] Release en GitHub: título `Eyve 2.1.0`, notas (qué hay: proyectos, etiquetado, entrenamiento local, producción con conteo; licencias por honor; requisitos: Windows 10/11 64-bit, 8 GB RAM, cámara USB), assets: `Eyve-2.1-setup.zip` + `SHA256SUMS.txt`. **No** marcar como pre-release (si no, `latest` no lo toma).
- [ ] Verificar que la URL fija descarga: `curl -sIL https://github.com/DecoTT/eyve/releases/latest/download/Eyve-2.1-setup.zip | grep -i "^HTTP\|location"` → termina en 200.

## 4. Prender la descarga en la tienda (sbcsuite)

Todo está preparado y apagado; se prende con dos valores iguales:

- [ ] **Secret** `EYVE_DOWNLOAD_URL` = la URL fija, en Supabase Dashboard → Edge Functions → Secrets. La function `licencias-emitir` lo lee al vuelo (no hay que redesplegar): desde ese momento el correo de licencia lleva el botón **Descargar Eyve 2.1** en vez de "se está publicando".
- [ ] **Front**: `src/modules/store/lib/eyve.ts` → `EYVE_DOWNLOAD_URL = '<la misma URL>'`; PR a `main`, Netlify publica. Con eso aparece el botón en la thank-you (`LicenciasEyve.tsx`) y en `/store/cuenta/licencias`.
- [ ] Probar: pedido Free desde `https://sbcgroup.com.mx/prueba-eyve/` con un correo tuyo → el correo trae botón y la thank-you también. Borrar el pedido de prueba después.
- [ ] Opcional: en `/prueba-eyve/` (HostGator, fuera del repo) un enlace "¿Ya tienes llave? Descarga Eyve 2.1" al pie, con la misma URL.
- [ ] Marcar hecho en `qr-lead-connect/docs/licencias-eyve.md` §6 ("Subir el instalador…") y en `ESTADO.md`.

## 5. Después del release

- [ ] Vigilar `/store/admin/licencias`: "Activadas alguna vez" contra "Emitidas" dice si la gente llega a pegar la llave (embudo-eyve.md §08).
- [ ] Versión siguiente (2.1.x): subir `__version__`, tag nuevo, release nuevo con el **mismo nombre de asset**. La URL fija no cambia; la tienda no se toca.
- [ ] Renovaciones y cortesías: desde el admin (Emitir manual / Renovar); Eyve recibe la llave nueva sola en su check-in semanal.

## Lo que NO hay que hacer

- No poner el ZIP en Mega ni en el bucket público: un enlace suelto no tiene versión ni checksum y hay que reemplazarlo a mano cada vez.
- No usar URLs firmadas de 72 h para el instalador: generan soporte ("ya no me abre el enlace").
- No cambiar el nombre del asset entre versiones sin cambiar también `EYVE_DOWNLOAD_URL` y `lib/eyve.ts`.
