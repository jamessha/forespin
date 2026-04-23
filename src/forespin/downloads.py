from __future__ import annotations

from pathlib import Path

from forespin.deps import require_huggingface_hub
from forespin.model_weights import DEFAULT_PLAYER_POSE_WEIGHTS_DIR

DEFAULT_YOLO_HF_REPO = "Ultralytics/YOLO26"
DEFAULT_YOLO_HF_FILE = "yolo26n-pose.pt"


def download_yolo_weights(
    *,
    repo_id: str = DEFAULT_YOLO_HF_REPO,
    filename: str = DEFAULT_YOLO_HF_FILE,
    output_dir: str | Path = DEFAULT_PLAYER_POSE_WEIGHTS_DIR,
) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    hub = require_huggingface_hub()
    downloaded = hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(output_path),
    )
    return Path(downloaded).resolve()
