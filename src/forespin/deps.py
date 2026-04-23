from __future__ import annotations

import importlib


class MissingDependencyError(RuntimeError):
    pass


def require_module(module_name: str, feature: str, install_extra: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            f"{feature} requires the optional dependency '{module_name}'. "
            f"Install with: python3 -m pip install -e '.[{install_extra}]'"
        ) from exc


def require_vision_stack() -> tuple[object, object]:
    cv2 = require_module("cv2", "Video analysis", "vision")
    numpy = require_module("numpy", "Video analysis", "vision")
    return cv2, numpy


def require_streamlit() -> object:
    return require_module("streamlit", "The Streamlit UI", "ui")


def require_huggingface_hub() -> object:
    return require_module("huggingface_hub", "Model downloads from Hugging Face", "vision")


def load_optional_module(module_name: str) -> object | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
