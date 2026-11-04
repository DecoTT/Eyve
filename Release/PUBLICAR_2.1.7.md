# Publicar Eyve 2.1.7

Todo está construido y probado en local. Falta solo lo que sale de esta PC.
Los comandos van en PowerShell, en este orden. `main` lleva commits sin subir
(el de Patrón, el de las etiquetas y los de esta versión).

## 1. Comprobar que el ZIP es el bueno

```powershell
cd "D:\Desarrollo\Claude Code\Eyve\2.1-dev"
git status --short
git log --oneline -5
(Get-FileHash Release\Eyve-2.1-setup.zip -Algorithm SHA256).Hash.ToLower()
Get-Content Release\SHA256SUMS.txt
```

`git status` no debe listar nada de `eyve/`, y los dos hashes deben ser
`22991d278c870eceac509cb950e631517f387d435769f5ee79e7fc65401498cd`.

## 2. Subir `main` y el tag

```powershell
git push origin main
git tag v2.1.7
git push origin v2.1.7
```

## 3. Crear el release

Guarda las notas de abajo (sin el título de sección) en
`Release\notas_2.1.7.md`, y luego:

```powershell
gh release create v2.1.7 Release\Eyve-2.1-setup.zip Release\SHA256SUMS.txt --title "Eyve 2.1.7" --notes-file Release\notas_2.1.7.md --latest
```

El nombre del asset no cambia, así que la tienda no se toca: la URL fija pasa
sola a la 2.1.7 en cuanto el release es `latest`.

## 4. Verificar contra lo publicado

```powershell
gh release view v2.1.7 --json tagName,isDraft,isPrerelease,assets --jq "{tag:.tagName,draft:.isDraft,pre:.isPrerelease,assets:[.assets[].name]}"
curl.exe -L -o "$env:TEMP\eyve217.zip" https://github.com/DecoTT/Eyve/releases/latest/download/Eyve-2.1-setup.zip
(Get-FileHash "$env:TEMP\eyve217.zip" -Algorithm SHA256).Hash.ToLower()
Remove-Item "$env:TEMP\eyve217.zip"
```

Debe decir `draft:false`, `pre:false`, con los dos assets, y el hash debe ser
el mismo del paso 1. Luego marca los pendientes de la sección 12 de
`RELEASE.md`.

## Notas de release (para pegar)

### Eyve 2.1.7

**Producción**

- **Patrón ahora se ve en Producción.** Sus anomalías pasan por el tracker,
  igual que en la demo: salen con recuadro magenta, ID y conteo. Antes el
  cuadro salía NO OK por Patrón sin ningún recuadro y sin llegar al contador.
- **Etiquetas sin «??».** Los textos con acento o símbolos que se dibujan
  sobre la imagen ya no salen con signos de interrogación («anomal??a»). Se
  dibujan sin acento; los nombres de tus clases no cambian.

**Entrenamiento**

- **«Crear Paquete» funciona en otra PC.** El `data.yaml` del paquete llevaba
  la ruta de la PC de origen y entrenar en otra máquina fallaba con
  `FileNotFoundError`. Ahora el paquete se puede descomprimir en cualquier
  carpeta. Tampoco lleva ya los archivos `.cache` de Ultralytics.

Si ya tienes un paquete hecho con una versión anterior, vuelve a crearlo con
«Crear Paquete», o borra la línea `path:` de su `dataset/data.yaml`.

**Instalación:** descarga `Eyve-2.1-setup.zip`, descomprímelo y corre
`setup.bat`. Si ya tienes Eyve, el actualizador te ofrece esta versión y no
toca tus proyectos.
