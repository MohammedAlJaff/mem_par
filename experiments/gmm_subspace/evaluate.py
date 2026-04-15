from pathlib import Path

from experiments._shared.cli import evaluate_main


if __name__ == "__main__":
    evaluate_main(Path(__file__).resolve().parent)
