# Fix moi truong: ep numpy<2 + opencv-python<4.11.
#
# Ly do:
#   - mediapipe 0.10.x compile chong NumPy 1.x -> can numpy<2
#   - opencv-python >=4.11 compile chong NumPy 2.x -> can opencv<4.11
#   Ca 2 phai downgrade cung luc.
#
# Idempotent.

$ErrorActionPreference = "Stop"

$projRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$venvActivate = Join-Path $projRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
    Write-Host "Activated venv: $projRoot\venv"
}

Write-Host ""
Write-Host "=== Phien ban hien tai ==="
$nv = (python -c "import numpy; print(numpy.__version__)" 2>$null)
$cv = (python -c "import cv2; print(cv2.__version__)" 2>$null)
if (-not $nv) { $nv = "(chua cai)" }
if (-not $cv) { $cv = "(chua cai)" }
Write-Host "numpy   : $nv"
Write-Host "opencv  : $cv"

$numpyOk  = $nv -match "^1\."
$opencvOk = $false
if ($cv -match "^4\.(\d+)\.") {
    if ([int]$Matches[1] -lt 11) { $opencvOk = $true }
}

if ($numpyOk -and $opencvOk) {
    Write-Host ""
    Write-Host "numpy va opencv da dung version. Khong can fix."
    exit 0
}

Write-Host ""
Write-Host "=== Bat dau fix ==="
Write-Host "Uninstall opencv-contrib-python (neu co)..."
python -m pip uninstall -y opencv-contrib-python 2>$null

Write-Host ""
Write-Host "Reinstall numpy<2 + opencv-python<4.11..."
python -m pip install "numpy<2" "opencv-python<4.11" --force-reinstall

Write-Host ""
Write-Host "=== Sau khi fix ==="
python -c "import numpy; print('numpy   :', numpy.__version__)"
python -c "import cv2;   print('opencv  :', cv2.__version__)"

Write-Host ""
Write-Host "Xong. Chay: python src\main.py"
