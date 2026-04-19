"""Standard Image32 UNet config presets."""

from typing import Callable, Dict

from prototyping.runners.image32_unet_configs import image32_128
from prototyping.runners.image32_unet_configs import image32_256
from prototyping.runners.image32_unet_configs import image32_32

DEFAULT_PRESET = "image32_32"

_PRESET_MAP: Dict[str, Callable[[], object]] = {
    "image32_32": image32_32.get_default_configs,
    "image32_128": image32_128.get_default_configs,
    "image32_256": image32_256.get_default_configs,
}


def list_presets():
    """Return sorted preset names."""
    return sorted(_PRESET_MAP.keys())


def get_default_configs(preset_name=DEFAULT_PRESET):
    """Return the requested preset `ConfigDict` for Image32 UNet."""
    if preset_name not in _PRESET_MAP:
        raise ValueError(f"Unknown Image32 UNet preset: {preset_name}")
    return _PRESET_MAP[preset_name]()
