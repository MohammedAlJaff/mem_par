"""
Canonical configuration helpers for first-party studies.

This package exposes the small set of config utilities used throughout the
repo: building a base `ConfigDict`, cloning it safely, converting it to plain
Python containers for serialization, and applying command-line-style dotted
overrides.
"""

from mp_config.base import apply_overrides, clone_config, config_to_dict, get_base_config

__all__ = ["apply_overrides", "clone_config", "config_to_dict", "get_base_config"]
