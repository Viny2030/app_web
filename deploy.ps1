# Script para desplegar la aplicación Streamlit a Backblaze B2
# Uso: .\deploy.ps1 [--static]

param (
    [switch]$static = $false
)

# Configuración
$env:PYTHONPATH = "$PWD"
$env:PYTHONIOENCODING = "UTF-8"

# Verificar si se debe usar el modo estático
if ($static) {
    Write-Host "🚀 Iniciando despliegue de versión estática..." -ForegroundColor Cyan
    python deploy_static.py
} else {
    Write-Host "🚀 Iniciando despliegue de aplicación completa..." -ForegroundColor Cyan
    python deploy_streamlit.py
}

# Verificar si hubo errores
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error durante el despliegue. Código de salida: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "✅ ¡Despliegue completado con éxito!" -ForegroundColor Green
