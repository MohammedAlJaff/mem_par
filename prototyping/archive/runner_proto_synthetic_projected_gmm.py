"""
Compatibility launcher for the projected-GMM MLP prototype runner.

The maintained implementation now lives in
`prototyping.runners.synthetic_mlp`.
"""

import sys
from pathlib import Path


HERE = Path(__file__).resolve()
for CANDIDATE in (HERE, *HERE.parents):
    if (CANDIDATE / "src").is_dir() and (CANDIDATE / "prototyping").is_dir():
        for PATH in (CANDIDATE, CANDIDATE / "src"):
            if str(PATH) not in sys.path:
                sys.path.insert(0, str(PATH))
        break

from prototyping.runners.synthetic_mlp import main


if __name__ == "__main__":
    main()
