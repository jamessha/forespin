from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

from forespin.deps import require_module, require_vision_stack

DEFAULT_NET_REMOVAL_MODEL = "gpt-image-2"
DEFAULT_NET_REMOVAL_PROMPT = (
    "Remove the tennis net and net posts from this baseline-view tennis court frame. "
    "Preserve the exact camera framing, perspective, court geometry, court lines, "
    "players, ball, lighting, and background. Reconstruct only the court surface and "
    "court lines hidden behind the net. Do not stylize the image."
)


class OpenAINetRemovalPreprocessor:
    def __init__(
        self,
        *,
        model: str = DEFAULT_NET_REMOVAL_MODEL,
        prompt: str = DEFAULT_NET_REMOVAL_PROMPT,
        timeout_s: float = 120.0,
    ) -> None:
        dotenv = require_module("dotenv", "OpenAI net removal", "vision")
        dotenv.load_dotenv()
        openai = require_module("openai", "OpenAI net removal", "vision")
        self.client = openai.OpenAI(timeout=timeout_s)
        self.model = model
        self.prompt = prompt

    def preprocess_frame(self, frame: Any, *, debug_dir: Path | None = None, frame_index: int = 0) -> Any:
        cv2, np = require_vision_stack()
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"frame_{frame_index:03d}_net_removal_input.jpg"), frame)

        with tempfile.TemporaryDirectory(prefix="forespin-net-removal-") as temp_dir:
            input_path = Path(temp_dir) / "frame.png"
            if not cv2.imwrite(str(input_path), frame):
                raise RuntimeError("Unable to encode frame for OpenAI net removal.")
            with input_path.open("rb") as image_file:
                response = self.client.images.edit(
                    model=self.model,
                    image=image_file,
                    prompt=self.prompt,
                )

        edited = self._decode_response_image(response, cv2, np)
        if edited.shape[:2] != frame.shape[:2]:
            edited = cv2.resize(edited, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        if debug_dir is not None:
            cv2.imwrite(str(debug_dir / f"frame_{frame_index:03d}_net_removed.jpg"), edited)
        return edited

    @staticmethod
    def _decode_response_image(response: Any, cv2: Any, np: Any) -> Any:
        data = getattr(response, "data", None)
        if not data:
            raise RuntimeError("OpenAI image edit response did not include image data.")
        first = data[0]
        encoded = getattr(first, "b64_json", None)
        if encoded is None and isinstance(first, dict):
            encoded = first.get("b64_json")
        if not encoded:
            raise RuntimeError("OpenAI image edit response did not include b64_json image data.")

        raw = base64.b64decode(encoded)
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Unable to decode OpenAI net-removed image.")
        return decoded
