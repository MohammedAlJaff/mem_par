from pathlib import Path

from experiments._shared.cli import train_main
from experiments.celeba32_grey.config import get_config


if __name__ == "__main__":
    train_main(get_config, Path(__file__).resolve().parent)
