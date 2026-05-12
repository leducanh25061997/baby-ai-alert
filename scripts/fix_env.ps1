# Fix moi truong numpy/opencv triet de.
#
# Van de:
#   - mediapipe 0.10.x compile chong NumPy 1.x -> can numpy<2
#   - opencv-python(-contrib)(-headless) >=4.11 compile chong NumPy 2.x
#   - 4 variant opencv share cv2 namespace, cai chung se xung dot.
#
# Action: uninstall TAT CA 4 variant, cai lai opencv-python<4.11 + numpy<2.

$ErrorActionPreference = "Stop"

$projRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$venvActivate = Join-Path $projRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
    Write-Host "Activated venv: $projRoot\venv"
}

Write-Host ""
Write-Host "=== Phien ban hien tai ==="
$nv = (python -c "import numpy; print(numpy.__version__)" 2>$null); if (-not $nv) { $nv = "(chua cai)" }
$cv = (python -c "import cv2; print(cv2.__version__)" 2>$null);    if (-not $cv) { $cv = "(chua cai)" }
Write-Host "numpy   : $nv"
Write-Host "cv2     : $cv"

Write-Host ""
Write-Host "=== Cac opencv package dang cai ==="
$opencvPkgs = (python -m pip list 2>$null | Select-String -Pattern "opencv")
if ($opencvPkgs) { Write-Host $opencvPkgs } else { Write-Host "(khong co)" }

Write-Host ""
Write-Host "=== Buoc 1: Uninstall tat ca opencv variant ==="
foreach ($pkg in @("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless")) {
    python -m pip uninstall -y $pkg 2>$null
}

Write-Host ""
Write-Host "=== Buoc 2: Cai numpy<2 + opencv-python<4.11 ==="
python -m pip install "numpy<2" "opencv-python<4.11" --force-reinstall

Write-Host ""
Write-Host "=== Sau khi fix ==="
python -c "import numpy; print('numpy   :', numpy.__version__)"
python -c "import cv2;   print('cv2     :', cv2.__version__)"
python -m pip list 2>$null | Select-String -Pattern "opencv"

Write-Host ""
Write-Host "Xong. Chay: python src\main.py"
