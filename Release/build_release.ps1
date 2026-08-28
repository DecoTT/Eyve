<#
    build_release.ps1 - empaqueta Eyve para distribucion.

    Toda la logica vive aqui en vez de en el .bat: los bloques
    `powershell -Command ^` multilinea dentro de un .bat se rompen de formas
    dificiles de diagnosticar (carets + comillas + rutas con espacios), que
    fue exactamente lo que corrompio un build anterior.

    NOTA: sin acentos a proposito. PowerShell 5.1 lee los .ps1 sin BOM como
    ANSI, y los caracteres UTF-8 multibyte rompen el parser.

    Uso:  .\build_release.ps1     (o doble clic en build_release.bat)
#>
$ErrorActionPreference = "Stop"

$Version   = "2.1_Beta"
$DistName  = "Eyve_$Version"
$Here      = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceDir = Split-Path -Parent $Here
$DistDir   = Join-Path $Here $DistName
$ZipOut    = Join-Path $Here "$DistName.zip"

Write-Host ""
Write-Host "  Building Eyve $Version release package..." -ForegroundColor Cyan
Write-Host "  Source : $SourceDir"
Write-Host "  Output : $DistDir"
Write-Host ""

# -- limpiar build anterior ---------------------------------------------------
if (Test-Path $DistDir) {
    Write-Host "  Removing previous build..."
    Remove-Item $DistDir -Recurse -Force
}
New-Item -ItemType Directory -Path $DistDir -Force | Out-Null

# -- copiar el paquete eyve/ (sin __pycache__, .pyc ni logs) ------------------
Write-Host "  Copying eyve package..."
$src = Join-Path $SourceDir "eyve"
$dst = Join-Path $DistDir  "eyve"
$copied = 0
Get-ChildItem $src -Recurse | Where-Object {
    $_.FullName -notmatch '__pycache__' -and $_.Extension -notin '.pyc', '.log'
} | ForEach-Object {
    $rel = $_.FullName.Substring($src.Length).TrimStart('\')
    $target = Join-Path $dst $rel
    if ($_.PSIsContainer) {
        New-Item -ItemType Directory -Force -Path $target | Out-Null
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
        Copy-Item $_.FullName -Destination $target -Force
        $copied++
    }
}
Write-Host "    $copied archivos"

# -- archivos raiz de la distribucion -----------------------------------------
Write-Host "  Copying root files..."
foreach ($f in @("requirements.txt", "LICENSE", "THIRD_PARTY_NOTICES.md")) {
    $p = Join-Path $SourceDir $f
    if (Test-Path $p) { Copy-Item $p $DistDir -Force }
}
# instaladores + README del tester (viven en Release/, son la version buena)
foreach ($f in @("setup.bat", "run.bat", "README.md")) {
    Copy-Item (Join-Path $Here $f) (Join-Path $DistDir $f) -Force
}

# carpeta de proyectos vacia
$projects = Join-Path $DistDir "projects"
New-Item -ItemType Directory -Path $projects -Force | Out-Null
Set-Content -Path (Join-Path $projects ".gitkeep") -Value "" -Encoding ascii

# -- ZIP ----------------------------------------------------------------------
Write-Host ""
Write-Host "  Creating $DistName.zip..."
if (Test-Path $ZipOut) { Remove-Item $ZipOut -Force }
Compress-Archive -Path (Join-Path $DistDir '*') -DestinationPath $ZipOut -Force

$sizeMb = [math]::Round((Get-Item $ZipOut).Length / 1MB, 2)
$hash   = (Get-FileHash $ZipOut -Algorithm SHA256).Hash

# -- verificacion del contenido -----------------------------------------------
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip     = [System.IO.Compression.ZipFile]::OpenRead($ZipOut)
$entries = $zip.Entries | Select-Object -ExpandProperty FullName
$zip.Dispose()

$checks = @(
    @{ Name = "sin __pycache__"; Ok = -not ($entries | Where-Object { $_ -match '__pycache__' }) }
    @{ Name = "sin .pyc";        Ok = -not ($entries | Where-Object { $_ -match '\.pyc$' }) }
    @{ Name = "setup.bat";       Ok = [bool]($entries | Where-Object { $_ -match '^setup\.bat$' }) }
    @{ Name = "run.bat";         Ok = [bool]($entries | Where-Object { $_ -match '^run\.bat$' }) }
    @{ Name = "README.md";       Ok = [bool]($entries | Where-Object { $_ -match '^README\.md$' }) }
    @{ Name = "requirements";    Ok = [bool]($entries | Where-Object { $_ -match 'requirements\.txt$' }) }
    @{ Name = "modulos Pro";     Ok = [bool]($entries | Where-Object { $_ -match 'polarity_module\.py$' }) }
    @{ Name = "tracker";         Ok = [bool]($entries | Where-Object { $_ -match 'tracker\.py$' }) }
)
Write-Host ""
Write-Host "  Verificacion del ZIP:"
$failed = 0
foreach ($c in $checks) {
    if ($c.Ok) {
        Write-Host "    OK    $($c.Name)" -ForegroundColor Green
    } else {
        Write-Host "    FALLA $($c.Name)" -ForegroundColor Red
        $failed++
    }
}

Write-Host ""
Write-Host "  ----------------------------------------------------"
if ($failed -eq 0) {
    Write-Host "   Release listo" -ForegroundColor Green
} else {
    Write-Host "   Release con $failed problema(s)" -ForegroundColor Red
}
Write-Host ""
Write-Host "   ZIP      : $ZipOut"
Write-Host "   Tamano   : $sizeMb MB  ($($entries.Count) archivos)"
Write-Host "   SHA-256  : $hash"
Write-Host ""
Write-Host "   Publica el hash junto a la descarga - el README explica"
Write-Host "   al tester como verificarlo (mitigacion de SmartScreen)."
Write-Host "  ----------------------------------------------------"
Write-Host ""

exit $failed
