# WildGuard

Hi! This is my WildGuard project.

WildGuard is a wildlife monitoring app that can:
- run **live webcam detection**
- analyze **images/videos offline**
- detect people + animals
- classify animal species
- show a threat state when person + animal appear together

---

## Quick Start (Windows)

### 1) Extract the zip
Unzip the project folder and open it in terminal/VS Code.

### 2) Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```powershell
pip install --upgrade pip
pip install numpy opencv-python pillow
pip install speciesnet
pip install -r external/MegaDetector/envs/requirements.txt
```

### 4) Confirm model file exists
Make sure this file is present:
- `models/md/md_v5a.0.0.pt`

### 5) Run the app
```powershell
python src/phase3test.py
```

---

## How to use
- Click **LIVE WEBCAM** for camera mode.
- Click **LOAD FILE** for image/video mode.
- Click **STOP SESSION** to stop.

---

## Notes
- The app currently runs from `src/phase3test.py`.
- `external/MegaDetector` is already included in this project and used directly.
- SpeciesNet may download/load resources on first run depending on your setup.

---

## Common issues

### PowerShell blocks venv activation
Run this once in PowerShell:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Webcam not opening
- Close other apps using the camera.
- Re-run the script.

### Slow first run
- First load can be slower because models are being initialized.

---

## Project entry points (for reference)
- Main app: `src/phase3test.py`
- Earlier versions: `src/wildguardpro.py`, `src/phase3.py`, `src/wildguard_phase2_live.py`
