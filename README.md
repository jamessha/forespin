# Forespin

`forespin` is a local batch analyzer for tennis videos recorded from a mostly fixed elevated baseline view. It focuses on one singles player and produces:

- shot-in percentages for `serve`, `forehand`, `backhand`, and `volley`
- lost-point percentages by depth (`baseline`, `net`, `other`) and lateral position (`left`, `right`)
- a machine-readable JSON timeline and an annotated overlay video

## What Is Implemented

- A Python package with a CLI and Streamlit entrypoint.
- Court calibration, ball tracking, and player tracking modules that require model-backed adapters.
- The default court calibration path now uses a learned keypoint detector with homography reconstruction, which is more tolerant of clipped back corners than the old line-only heuristic.
- Event detection for hits and bounces, point assembly, shot classification, quality gating, and metrics aggregation.
- A `download` command for fetching the default YOLO26 pose weights into the repo.
- Dev-set annotation scaffolding for building the 20-30 clip tuning set from the plan.
- Stdlib unit tests for the core shot, point, and metric logic.

## What Is Not Bundled

- Model weights are not checked into the repo.
- `TrackNetV2` weights are local/manual only. Place the yastrebksv/TrackNet state-dict checkpoint or a TorchScript-exported model in `models/tracknet/`, or provide a local path with `--tracknet-weights`.
- The learned court detector is also local/manual only. Place a checkpoint in `models/court/`, or provide a local path with `--court-weights`.
- The player tracker now targets `YOLO26-pose`, not `YOLOv8-pose`.
- The player model is local at analysis time. Use `forespin download yolo` to fetch the default `yolo26n-pose.pt` into `models/yolo26/`, or pass a local path with `--player-pose-weights`.

## Install

```bash
python3 -m pip install -e ".[all]"
```

If you only want the package and tests first:

```bash
python3 -m pip install -e .
```

## CLI Usage

Place your TrackNet weights here if you do not want to pass a flag. The app accepts the yastrebksv/TrackNet state-dict checkpoint as well as TorchScript-exported TrackNet models:

```text
models/tracknet/tracknetv2.torchscript.pt
```

Download the default YOLO26 pose weights here:

```bash
forespin download yolo
```

That downloads the default checkpoint to:

```text
models/yolo26/yolo26n-pose.pt
```

Place the learned court detector checkpoint here:

```text
models/court/tennis_court_detector.pt
```

If the filename differs, `forespin` will still auto-detect it when `models/court/` contains exactly one checkpoint file.

```bash
forespin analyze /path/to/match.mp4 \
  --player-side near \
  --handedness right \
  --output-dir outputs/session-001
```

That command works if the TrackNet file is present in `models/tracknet/tracknetv2.torchscript.pt`, the YOLO file is present in `models/yolo26/yolo26n-pose.pt`, and the learned court detector checkpoint is present in `models/court/`.

To use local files instead:

```bash
forespin analyze /path/to/match.mp4 \
  --player-side near \
  --handedness right \
  --tracknet-weights /models/tracknetv2.torchscript.pt \
  --player-pose-weights /models/yolo26n-pose.pt \
  --court-weights /models/tennis_court_detector.pt \
  --output-dir outputs/session-001
```

For real baseline footage where the net occludes court lines, you can run a first-frame net-removal pass before learned court calibration:

```bash
forespin analyze /path/to/match.mp4 \
  --player-side near \
  --handedness right \
  --remove-net-for-court-calibration \
  --output-dir outputs/session-001
```

This uses OpenAI image editing with `gpt-image-2`, loads `.env` via `python-dotenv`, and expects `OPENAI_API_KEY` to be available. The API call is opt-in because it sends the first video frame to OpenAI and adds cost/latency.

Court calibration is cached per input filename and calibration mode under:

```text
outputs/<video-stem>/cache/court_calibration/
```

Use `--no-court-cache` to force recalibration while testing.

## Streamlit Usage

```bash
streamlit run streamlit_app.py
```

The Streamlit UI accepts local model paths and also checks the default repo-local model locations for TrackNet, the learned court detector, and YOLO26.

## Outputs

- `timeline.json`: serialized analysis result
- `overlay.mp4`: annotated playback if OpenCV video writing is available

## Dev Set

See [devset/README.md](/Users/james/playground/forespin/devset/README.md) for the annotation contract used to build and validate the tuning set from the implementation plan.

## Courtside Calibration Data

The original TennisCourtDetector calibration data lives in:

```text
calib_model_data/tennis_court_detector/
```

Generate lower-height courtside-style calibration data:

```bash
python3 scripts/augment_courtside_data.py --overwrite
```

Generate one transformed sample into the same output dataset:

```bash
python3 scripts/augment_courtside_data.py --test --seed 1 --overwrite
```

By default this writes:

```text
calib_model_data/courtside_data/images/
calib_model_data/courtside_data/data_train.json
calib_model_data/courtside_data/data_val.json
calib_model_data/courtside_data/augmentation_report.json
```

The script assumes the original professional baseline camera is roughly 30 ft high, 21 ft behind the nearest baseline, and tilted down at about a 20% grade. For each generated image, it samples a target camera height between 4-6 ft, samples a downward angle between 8-10 degrees, jitters the target Y position by up to 1 ft around a 15 ft behind-baseline distance, keeps the original TennisCourtDetector JSON schema, maps all 14 labeled keypoints through the same homography used for the image warp, zero-pads newly exposed image regions, and writes a 75/25 train/validation split.

Render label overlays for inspection:

```bash
python3 scripts/visualize_court_labels.py --data-root calib_model_data/courtside_data --max-images 100
```

Run learned court calibration inference on a single image:

```bash
python3 scripts/infer_court_calibration.py /path/to/baseline-view-frame.jpg \
  --output-dir outputs/court_calibration_debug
```

The script auto-discovers the court checkpoint in `models/court/` unless `--weights` is provided. It writes a raw image copy, an overlay image, `summary.json`, and `calibration.json` when the image is accepted.

To test OpenAI net removal on a single frame before calibration:

```bash
python3 scripts/infer_court_calibration.py /path/to/baseline-view-frame.jpg \
  --remove-net-with-openai \
  --output-dir outputs/court_calibration_debug
```

Train the vendored court detector on the generated courtside data:

```bash
python3 -m pip install -r tennis_court_detector/requirements.txt
python3 tennis_court_detector/main.py
```

Training writes both inference weights and resumable checkpoints under the experiment directory. For example:

```text
tennis_court_detector/exps/default/model_last.pt
tennis_court_detector/exps/default/model_best.pt
tennis_court_detector/exps/default/checkpoint_last.pt
tennis_court_detector/exps/default/checkpoint_best.pt
```

Resume the latest checkpoint for an experiment:

```bash
python3 tennis_court_detector/main.py --exp_id default --resume
```

Or resume a specific checkpoint:

```bash
python3 tennis_court_detector/main.py --resume tennis_court_detector/exps/default/checkpoint_last.pt
```
