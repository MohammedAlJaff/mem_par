import ast
import unittest
from pathlib import Path


class TestLiveTreeBoundaries(unittest.TestCase):
    def test_src_has_no_imports_from_experiments_or_reference(self):
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
                            or alias.name.startswith("reference."),
                            msg=f"{path}:{node.lineno} imports archived module {alias.name}",
                        )
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertFalse(
                        module == "experiments"
                        or module.startswith("experiments.")
                        or module == "reference"
                        or module.startswith("reference."),
                        msg=f"{path}:{node.lineno} imports archived module {module}",
                    )

    def test_archived_trees_are_removed(self):
        self.assertFalse(Path("experiments").exists())
        self.assertFalse(Path("reference").exists())


if __name__ == "__main__":
    unittest.main()
