from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forespin.downloads import download_yolo_weights


class DownloadTests(unittest.TestCase):
    def test_download_yolo_weights_uses_huggingface_hub(self) -> None:
        fake_hub = SimpleNamespace(hf_hub_download=lambda **kwargs: str(Path(kwargs["local_dir"]) / kwargs["filename"]))
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("forespin.downloads.require_huggingface_hub", return_value=fake_hub):
                downloaded = download_yolo_weights(output_dir=temp_dir)
        self.assertTrue(str(downloaded).endswith("yolo26n-pose.pt"))


if __name__ == "__main__":
    unittest.main()
