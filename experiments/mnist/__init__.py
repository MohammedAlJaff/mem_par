"""
Live MNIST study surface.

This package keeps two complementary entrypoints side by side:

- `mnist_walkthrough.ipynb` for the pedagogical checkpoint-replay walkthrough
- `run.py` for the official artifact-writing path
"""

def build_mnist_config(*args, **kwargs):
    """
    Re-export the MNIST config builder without eagerly importing `run.py`.
    """
    from experiments.mnist.run import build_mnist_config as _build_mnist_config

    return _build_mnist_config(*args, **kwargs)


def run_mnist_study(*args, **kwargs):
    """
    Re-export the MNIST saved-run helper without eager side effects.
    """
    from experiments.mnist.run import run_mnist_study as _run_mnist_study

    return _run_mnist_study(*args, **kwargs)

__all__ = ["build_mnist_config", "run_mnist_study"]
