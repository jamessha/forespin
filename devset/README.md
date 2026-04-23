# Dev Set Contract

The MVP plan calls for a labeled development set of 20-30 representative clips. This directory defines the annotation format the analysis pipeline expects for tuning and offline evaluation.

## Recommended Clip Mix

- 8-10 clean baseline rally clips
- 4-6 serve-heavy clips
- 4-6 net-approach / volley clips
- 4-6 edge-case clips with mild shake, brief ball occlusion, or missed-bounce scenarios

## Required Per-Clip Labels

- `clip_id`
- `video_path`
- `tracked_player_side`
- `handedness`
- `court_corners_px`
- `hit_events`
- `bounce_events`
- `points`

## Field Notes

- `court_corners_px` must be ordered: top-left, top-right, bottom-right, bottom-left.
- `hit_events[*].shot_type` uses `serve`, `forehand`, `backhand`, `volley`, or `unknown`.
- `points[*].winner` uses `tracked` or `opponent`.
- `points[*].final_loss_zone` includes one depth label and one lateral label.

Validate an annotations file with:

```bash
python3 scripts/validate_devset.py devset/annotations_template.json
```

