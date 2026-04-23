from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from forespin.domain import InputConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKNET_WEIGHTS_DIR = REPO_ROOT / "models" / "tracknet"
DEFAULT_TRACKNET_WEIGHTS_PATH = DEFAULT_TRACKNET_WEIGHTS_DIR / "tracknetv2.torchscript.pt"
DEFAULT_PLAYER_POSE_WEIGHTS_DIR = REPO_ROOT / "models" / "yolo26"
DEFAULT_PLAYER_POSE_WEIGHTS_PATH = DEFAULT_PLAYER_POSE_WEIGHTS_DIR / "yolo26n-pose.pt"
DEFAULT_COURT_WEIGHTS_DIR = REPO_ROOT / "models" / "court"
DEFAULT_COURT_WEIGHTS_PATH = DEFAULT_COURT_WEIGHTS_DIR / "tennis_court_detector.pt"


class MissingModelWeightsError(RuntimeError):
    pass


@dataclass(slots=True)
class ResolvedModelWeights:
    tracknet_weights: str
    player_pose_weights: str
    court_weights: str


def resolve_model_weights(input_config: InputConfig) -> ResolvedModelWeights:
    tracknet_local_path = input_config.tracknet_weights
    if not tracknet_local_path and DEFAULT_TRACKNET_WEIGHTS_PATH.exists():
        tracknet_local_path = str(DEFAULT_TRACKNET_WEIGHTS_PATH)
    player_pose_local_path = input_config.player_pose_weights
    if not player_pose_local_path and DEFAULT_PLAYER_POSE_WEIGHTS_PATH.exists():
        player_pose_local_path = str(DEFAULT_PLAYER_POSE_WEIGHTS_PATH)
    court_local_path = input_config.court_weights
    if not court_local_path:
        court_candidate = discover_default_local_artifact(
            default_path=DEFAULT_COURT_WEIGHTS_PATH,
            search_dir=DEFAULT_COURT_WEIGHTS_DIR,
        )
        if court_candidate is not None:
            court_local_path = str(court_candidate)

    return ResolvedModelWeights(
        tracknet_weights=resolve_local_artifact_path(
            local_path=tracknet_local_path,
            model_label="TrackNetV2",
            missing_hint=(
                f"Place tracknetv2.torchscript.pt in {DEFAULT_TRACKNET_WEIGHTS_DIR} "
                "or provide --tracknet-weights."
            ),
        ),
        player_pose_weights=resolve_local_artifact_path(
            local_path=player_pose_local_path,
            model_label="YOLO26 pose",
            missing_hint=(
                f"Place yolo26n-pose.pt in {DEFAULT_PLAYER_POSE_WEIGHTS_DIR}, "
                "run `forespin download yolo`, or provide --player-pose-weights."
            ),
        ),
        court_weights=resolve_local_artifact_path(
            local_path=court_local_path,
            model_label="Learned court detector",
            missing_hint=(
                f"Place a tennis-court detector checkpoint in {DEFAULT_COURT_WEIGHTS_DIR} "
                "(the default filename is tennis_court_detector.pt) or provide --court-weights."
            ),
        ),
    )


def discover_default_local_artifact(*, default_path: Path, search_dir: Path) -> Path | None:
    if default_path.exists():
        return default_path
    if not search_dir.exists():
        return None
    candidates = sorted(
        path
        for path in search_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".bin"}
    )
    if len(candidates) == 1:
        return candidates[0]
    return None


def resolve_local_artifact_path(
    *,
    local_path: str | None,
    model_label: str,
    missing_hint: str,
) -> str:
    if local_path:
        expanded = Path(local_path).expanduser().resolve()
        if not expanded.exists():
            raise FileNotFoundError(f"{model_label} weights not found: {expanded}")
        return str(expanded)

    raise MissingModelWeightsError(f"{model_label} weights are required. {missing_hint}")


def has_default_tracknet_weights() -> bool:
    return DEFAULT_TRACKNET_WEIGHTS_PATH.exists()


def has_default_player_pose_weights() -> bool:
    return DEFAULT_PLAYER_POSE_WEIGHTS_PATH.exists()


def has_default_court_weights() -> bool:
    return discover_default_local_artifact(
        default_path=DEFAULT_COURT_WEIGHTS_PATH,
        search_dir=DEFAULT_COURT_WEIGHTS_DIR,
    ) is not None
