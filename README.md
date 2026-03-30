# IR Glasses Privacy Diagnostic

CPU-focused diagnostic tool to measure whether IR-blocking glasses prevent biometric identification from an IR camera stream.

## Key outcomes
- **Identity Detected (Privacy Fail):** face is found and matched to baseline.
- **Identity Blocked (Privacy Success):** face is found but does not match baseline.
- **Total Evasion (Ultimate Privacy):** no face found for sustained frames.

## Quick start
1. Create env and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Optional: install dlib backend support:
   - `sudo apt-get update && sudo apt-get install -y python3-dev build-essential cmake`
   - `pip install -r requirements-dlib.txt`
   - Note: `requirements-dlib.txt` pins `setuptools<81` for `face_recognition_models` compatibility on Python 3.12.
3. Probe camera:
   - `python -m src.main --probe --camera-index 2`
4. Run app:
   - `python -m src.main --camera-index 2 --backend mediapipe`

## Controls
- `c`: Capture/recalibrate baseline (without glasses)
- `b`: Switch backend (mediapipe <-> dlib)
- `a`: Toggle auto-calibration
- `q` or `ESC`: Quit

## Auto-calibration behavior
- Auto-calibration updates match/block thresholds from recent live distance samples.
- HUD shows live threshold values (`match`, `block`) and sample count so threshold changes are visible.
- It updates gradually (not every frame) to reduce jitter and is reset when baseline or backend changes.

## Data outputs
- Baseline artifacts in `data/baseline/`
- Session logs in `data/logs/` as JSONL + summary JSON

## Notes
- `mediapipe` is the default backend and is installed by `requirements.txt`.
- `face_recognition`/`dlib` is optional and can require system headers (`python3-dev`) to compile.
- If a backend package is missing, switch to available backend or install the dependency.
