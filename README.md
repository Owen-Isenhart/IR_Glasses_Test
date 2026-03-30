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
- `c`: Capture/add baseline (without glasses)
- `b`: Switch backend (mediapipe <-> dlib)
- `a`: Toggle auto-calibration
- `q` or `ESC`: Quit

## Multi-baseline behavior
- Each baseline capture is stored per backend and camera index (default: append mode).
- Matching uses the best (lowest-distance) result across all stored baselines for the active backend + active camera mode.
- Configure baseline capture mode in `config/config.yaml` under `runtime.append_baseline`:
   - `true` (default): add new capture to baseline set.
   - `false`: replace with only the latest capture.

## Auto-calibration behavior
- Auto-calibration derives match/block thresholds directly from recent live distance samples (quantile based).
- It does **not** capture or add baseline embeddings.
- HUD shows live threshold values (`match`, `block`) and sample count so threshold changes are visible.
- It updates every few frames after enough samples are collected and resets when baseline or backend changes.

## Backend default thresholds
- `mediapipe`: match ≈ `0.20`, blocked ≈ `0.35`
- `dlib`: match ≈ `0.01`, blocked ≈ `0.08`

## Dlib performance tuning
- Dlib now supports frame skipping, downscaled detection, and optional OpenCV tracking.
- Configure in `config/config.yaml` under `runtime.dlib`:
   - `detect_every_n_frames`: run full dlib detect+encode every N frames (cached result is reused between runs)
   - `detection_scale`: detect on a downscaled frame (e.g. `0.5`) and map bbox back to full resolution
   - `use_tracker`: use tracker updates between detector runs to avoid repeated full detections
   - `tracker_type`: tracker implementation (`MOSSE`, `KCF`, `CSRT`)
   - `lost_face_grace_frames`: keep the last known face bbox+embedding for this many missed frames to reduce flicker/dropouts
- For more stable boxes, prefer `CSRT` (best stability, higher CPU) or `KCF` (balanced).

## MediaPipe stability tuning
- Configure in `config/config.yaml` under `runtime.mediapipe`:
   - `min_detection_confidence`: lower values are less strict (more detections, potentially more false positives)
   - `model_selection`: `0` short-range, `1` longer-range detector
   - `lost_face_grace_frames`: number of consecutive missed detector frames to keep last bbox/embedding before declaring no face
- If status flips to evasion too aggressively, increase `thresholds.evasion_frames`.

## Data outputs
- Baseline artifacts in `data/baseline/`
- Session logs in `data/logs/` as JSONL + summary JSON

## Notes
- `mediapipe` is the default backend and is installed by `requirements.txt`.
- `face_recognition`/`dlib` is optional and can require system headers (`python3-dev`) to compile.
- If a backend package is missing, switch to available backend or install the dependency.
