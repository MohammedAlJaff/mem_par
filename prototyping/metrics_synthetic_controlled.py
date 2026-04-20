"""
Compatibility import path for synthetic controlled metrics.

The implementation lives with the prototype runners, but older notebooks and
the direct synthetic runner still import `prototyping.metrics_synthetic_controlled`.
"""

from prototyping.runners.metrics_synthetic_controlled import *  # noqa: F401,F403
