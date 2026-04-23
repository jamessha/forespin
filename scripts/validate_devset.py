from __future__ import annotations

import json
import sys
from pathlib import Path


VALID_PLAYER_SIDES = {"near", "far"}
VALID_HANDEDNESS = {"right", "left"}
VALID_ACTORS = {"tracked", "opponent", "unknown"}
VALID_SHOTS = {"serve", "forehand", "backhand", "volley", "unknown"}
VALID_DEPTH = {"baseline", "net", "other"}
VALID_LATERAL = {"left", "right"}


def validate_devset(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text())
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["Root payload must be an object."]
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        return ["'clips' must be a non-empty array."]

    for clip_index, clip in enumerate(clips):
        prefix = f"clips[{clip_index}]"
        if clip.get("tracked_player_side") not in VALID_PLAYER_SIDES:
            errors.append(f"{prefix}.tracked_player_side must be one of {sorted(VALID_PLAYER_SIDES)}")
        if clip.get("handedness") not in VALID_HANDEDNESS:
            errors.append(f"{prefix}.handedness must be one of {sorted(VALID_HANDEDNESS)}")
        corners = clip.get("court_corners_px")
        if not isinstance(corners, list) or len(corners) != 4:
            errors.append(f"{prefix}.court_corners_px must contain exactly four points.")
        for hit_index, hit in enumerate(clip.get("hit_events", [])):
            hit_prefix = f"{prefix}.hit_events[{hit_index}]"
            if hit.get("actor") not in VALID_ACTORS:
                errors.append(f"{hit_prefix}.actor must be one of {sorted(VALID_ACTORS)}")
            if hit.get("shot_type") not in VALID_SHOTS:
                errors.append(f"{hit_prefix}.shot_type must be one of {sorted(VALID_SHOTS)}")
        for point_index, point in enumerate(clip.get("points", [])):
            point_prefix = f"{prefix}.points[{point_index}]"
            final_loss_zone = point.get("final_loss_zone", {})
            if final_loss_zone.get("depth") not in VALID_DEPTH:
                errors.append(f"{point_prefix}.final_loss_zone.depth must be one of {sorted(VALID_DEPTH)}")
            if final_loss_zone.get("lateral") not in VALID_LATERAL:
                errors.append(f"{point_prefix}.final_loss_zone.lateral must be one of {sorted(VALID_LATERAL)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 1:
        print("usage: python3 scripts/validate_devset.py <annotations.json>")
        return 1
    errors = validate_devset(args[0])
    if errors:
        for error in errors:
            print(error)
        return 1
    print("devset annotations are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

