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

    def test_live_and_legacy_layout_match_submission_repo_split(self):
        self.assertTrue(Path("experiments").exists())
        self.assertFalse(Path("reference").exists())
        self.assertTrue(Path("archives/legacy_experiments_20260416.zip").exists())
        self.assertTrue(Path("archives/legacy_reference_20260416.zip").exists())


if __name__ == "__main__":
    unittest.main()
