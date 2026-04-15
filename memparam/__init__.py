"""
Deprecated staging namespace kept temporarily for safety during the `src/`
migration.

Canonical reusable code now lives under `src/`, with `mp_*` packages for the
thin experiment-support layer and `gdfm` for the model core.
"""

from mp_config import apply_overrides, config_to_dict, get_base_config

__all__ = ["apply_overrides", "config_to_dict", "get_base_config"]
