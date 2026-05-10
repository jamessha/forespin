# Forespin

Forespin is a local tennis-video analyzer for mostly fixed elevated-baseline phone footage. It is aimed at one singles player at a time: give it a video, the tracked player side, and handedness, and it writes a machine-readable timeline plus an annotated overlay video.

This project was primarily written by Codex through iterative development sessions. Treat it as an experimental local tool, not a polished product or a validated analytics system.

## What It Does

Forespin runs an offline pipeline over a video:

- Calibrates the tennis court into normalized court coordinates.
- Tracks the tennis ball.
- Tracks the selected player.
- Detects hits and bounces.
- Produces shot-in rates for `serve`, `forehand`, `backhand`, and `volley`.
- Produces lost-point location rates by depth (`baseline`, `net`, `other`) and lateral side (`left`, `right`).
- Writes `timeline.json` and, unless disabled, `overlay.mp4`.

The expected input is singles footage from a stable baseline view. Heavy zooming, panning, missing court visibility, or poor ball visibility should be rejected or flagged rather than trusted.

## Install

Clone with submodules so the court-detector training code is available:

```bash
git clone --recurse-submodules <repo-url>
cd forespin
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Install the Python package and vision/UI dependencies:

```bash
python3 -m pip install -e ".[all]"
```

For package-only development without OpenCV/PyTorch/Streamlit:

```bash
python3 -m pip install -e .
```

## Model Weights

Model weights are not committed to the repo. You need three local artifacts before `forespin analyze` will run.

### TrackNet Ball Weights

TrackNet is manual/local only. Download or create TrackNet weights yourself from the [yastrebksv/TrackNet project](https://github.com/yastrebksv/TrackNet), then place the file here:

```text
models/tracknet/tracknetv2.torchscript.pt
```

The analyzer accepts either a TorchScript-exported TrackNet model or the compatible yastrebksv/TrackNet state-dict checkpoint. If you use another filename, pass it with `--tracknet-weights`.

### YOLO26 Pose Weights

YOLO26 pose weights are downloaded from Hugging Face:

```bash
forespin download yolo
```

This saves the default file here:

```text
models/yolo26/yolo26n-pose.pt
```

You can also pass another local pose checkpoint with `--player-pose-weights`.

### Court Detector Weights

The learned court calibrator uses the `tennis_court_detector` submodule, which points at:

```text
https://github.com/jamessha/TennisCourtDetector
```

Place a trained checkpoint here:

```text
models/court/tennis_court_detector.pt
```

Court detector weights trained for this project are available on [Google Drive](https://drive.google.com/file/d/1xLwRbsWz8blh7cFF9eZl3LJZb8p0VrC3/view?usp=sharing).

If `models/court/` contains exactly one `.pt`, `.pth`, or `.bin` file, Forespin will use it automatically. Otherwise pass the checkpoint explicitly with `--court-weights`.

## Run Analysis

Default model locations:

```bash
forespin analyze /path/to/match.mp4 \
  --player-side near \
  --handedness right \
  --output-dir outputs/session-001
```

Explicit model paths:

```bash
forespin analyze /path/to/match.mp4 \
  --player-side near \
  --handedness right \
  --tracknet-weights /path/to/tracknet.pt \
  --player-pose-weights /path/to/yolo26n-pose.pt \
  --court-weights /path/to/tennis_court_detector.pt \
  --output-dir outputs/session-001
```

Useful run flags:

- `--skip-overlay`: write JSON only.
- `--allow-low-quality`: persist metrics even if the quality gate rejects the clip.
- `--no-remove-net-for-court-calibration`: disable default OpenAI net removal and calibrate from raw frames.
- `--no-court-cache`: force court recalibration.
- `--no-trace-cache`: force ball/player tracking to rerun.

Outputs are written under:

```text
<output-dir>/<video-stem>/timeline.json
<output-dir>/<video-stem>/overlay.mp4
```

## Court Calibration Net Removal

Forespin removes the net from the calibration frame by default before running court calibration. This is the default because the court detector was trained primarily on professional video angles with much less net occlusion than amateur elevated-baseline phone footage.

Default behavior:

```bash
forespin analyze /path/to/match.mp4 \
  --player-side near \
  --handedness right \
  --output-dir outputs/session-001
```

Disable net removal for a fully local/raw-frame calibration run:

```bash
forespin analyze /path/to/match.mp4 \
  --player-side near \
  --handedness right \
  --no-remove-net-for-court-calibration \
  --output-dir outputs/session-001
```

Net removal requires `OPENAI_API_KEY` in the environment or a `.env` file. It sends one video frame to OpenAI and adds cost/latency.

## Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI uses the same model paths and defaults as the CLI.

## Court Detector Training

The court detector is kept as a Git submodule at `tennis_court_detector/`. To train it on generated courtside-style calibration data, first generate the dataset:

```bash
python3 scripts/augment_courtside_data.py --overwrite
```

This writes:

```text
calib_model_data/courtside_data/images/
calib_model_data/courtside_data/data_train.json
calib_model_data/courtside_data/data_val.json
calib_model_data/courtside_data/augmentation_report.json
```

Inspect label overlays:

```bash
python3 scripts/visualize_court_labels.py \
  --data-root calib_model_data/courtside_data \
  --max-images 100
```

Install the court-detector training dependencies and train:

```bash
python3 -m pip install -r tennis_court_detector/requirements.txt
python3 tennis_court_detector/main.py
```

Training writes experiment files under `tennis_court_detector/exps/`. Copy the best inference checkpoint into Forespin’s model directory when you want to use it:

```bash
cp tennis_court_detector/exps/default/model_best.pt models/court/tennis_court_detector.pt
```

Resume training:

```bash
python3 tennis_court_detector/main.py --exp_id default --resume
```

## Court Calibration Debugging

Run learned court calibration on a single image:

```bash
python3 scripts/infer_court_calibration.py /path/to/frame.jpg \
  --output-dir outputs/court_calibration_debug
```

With OpenAI net removal:

```bash
python3 scripts/infer_court_calibration.py /path/to/frame.jpg \
  --remove-net-with-openai \
  --output-dir outputs/court_calibration_debug
```

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall src tests
```
