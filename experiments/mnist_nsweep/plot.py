from pathlib import Path

from experiments._shared.cli import plot_main


if __name__ == "__main__":
    plot_main(Path(__file__).resolve().parent)
