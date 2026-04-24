import ast
import unittest
from pathlib import Path


class TestLiveTreeBoundaries(unittest.TestCase):
    def test_src_has_no_imports_from_live_or_legacy_study_trees(self):
        for path in sorted(Path("src").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue

            tree = ast.parse(path.read_text())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            alias.name == "experiments"
                            or alias.name.startswith("experiments.")
                            or alias.name == "reference"
                            or alias.name.startswith("reference.")
                            or alias.name == "archives"
                            or alias.name.startswith("archives."),
                            msg=f"{path}:{node.lineno} imports study-layer module {alias.name}",
                        )
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertFalse(
                        module == "experiments"
                        or module.startswith("experiments.")
                        or module == "reference"
                        or module.startswith("reference.")
                        or module == "archives"
                        or module.startswith("archives."),
                        msg=f"{path}:{node.lineno} imports study-layer module {module}",
                    )

    def test_active_runners_live_under_src(self):
        self.assertTrue(Path("src/mp_runners/synthetic_runner.py").is_file())
        self.assertTrue(Path("src/mp_runners/image_runner.py").is_file())
        self.assertTrue(Path("src/mp_metrics/metrics_synthetic.py").is_file())
        self.assertTrue(Path("src/mp_metrics/metrics_images.py").is_file())

        removed_active_paths = [
            "prototyping/runners/synthetic_runner.py",
            "prototyping/runners/image_runner.py",
            "prototyping/runners/metrics_synthetic.py",
            "prototyping/runners/metrics_synthetic_controlled.py",
            "prototyping/runners/metrics_image32.py",
        ]
        for path in removed_active_paths:
            self.assertFalse(Path(path).exists(), path)


if __name__ == "__main__":
    unittest.main()
