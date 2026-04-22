import ast
import unittest
from pathlib import Path


class TestSrcDocstrings(unittest.TestCase):
    MAIN_EXEMPTIONS = {
        "src/gdfm/__init__.py",
        "src/gdfm/model.py",
        "src/gdfm/networks/__init__.py",
        "src/mp_artifacts/__init__.py",
        "src/mp_config/__init__.py",
        "src/mp_data/__init__.py",
        "src/mp_metrics/__init__.py",
        "src/mp_runners/__init__.py",
        "src/mp_training/__init__.py",
    }

    REQUIRED_METHOD_DOCSTRINGS = {
        "src/gdfm/core.py": {
            "GDFM": {"__init__", "compute_loss", "convert", "sample"},
        },
        "src/gdfm/networks/mlp.py": {
            "TimeCondMLP": {"forward"},
        },
        "src/gdfm/networks/unet.py": {
            "BonnaireUNet": {"forward"},
        },
        "src/gdfm/networks/jit.py": {
            "JiT": {"forward"},
        },
    }

    def test_every_src_module_has_a_docstring(self):
        for path in sorted(Path("src").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue

            tree = ast.parse(path.read_text())
            module_doc = ast.get_docstring(tree)
            self.assertTrue(
                module_doc,
                f"{path} is missing a module docstring",
            )

    def test_every_top_level_definition_has_a_docstring(self):
        for path in sorted(Path("src").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue

            tree = ast.parse(path.read_text())
            for node in tree.body:
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    continue

                docstring = ast.get_docstring(node)
                self.assertTrue(
                    docstring,
                    f"{path}:{node.lineno} top-level definition '{node.name}' is missing a docstring",
                )

    def test_selected_important_methods_have_docstrings(self):
        for rel_path, class_requirements in self.REQUIRED_METHOD_DOCSTRINGS.items():
            path = Path(rel_path)
            tree = ast.parse(path.read_text())

            classes = {
                node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
            }

            for class_name, required_methods in class_requirements.items():
                self.assertIn(class_name, classes, f"{path} is missing class {class_name}")
                class_node = classes[class_name]

                methods = {
                    node.name: node
                    for node in class_node.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }

                for method_name in required_methods:
                    self.assertIn(
                        method_name,
                        methods,
                        f"{path}:{class_name} is missing method {method_name}",
                    )
                    self.assertTrue(
                        ast.get_docstring(methods[method_name]),
                        f"{path}:{class_name}.{method_name} is missing a docstring",
                    )

    def test_substantive_src_modules_have_main_blocks(self):
        for path in sorted(Path("src").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue

            rel_path = path.as_posix()
            if rel_path in self.MAIN_EXEMPTIONS:
                continue

            tree = ast.parse(path.read_text())
            has_main_block = any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__"
                for node in tree.body
            )

            self.assertTrue(
                has_main_block,
                f"{path} is missing an if __name__ == '__main__' example block",
            )


if __name__ == "__main__":
    unittest.main()
