import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestSrcExamples(unittest.TestCase):
    MODULES = (
        "gdfm.core",
        "gdfm.model_setup",
        "gdfm.samplers",
        "gdfm.networks.jit",
        "mp_config.base",
        "mp_metrics.synthetic",
        "mp_metrics.images",
        "mp_runners.common",
        "mp_artifacts.artifacts",
        "mp_data.gmm",
    )

    def _run_module(self, module_name):
        env = os.environ.copy()
        src_path = str(ROOT / "src")
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            src_path if not current else src_path + os.pathsep + current
        )
        return subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_representative_src_examples_run(self):
        for module_name in self.MODULES:
            with self.subTest(module=module_name):
                result = self._run_module(module_name)
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=(
                        f"{module_name} example failed.\n"
                        f"stdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}"
                    ),
                )
                self.assertTrue(
                    result.stdout.strip(),
                    msg=f"{module_name} example did not print any output.",
                )


if __name__ == "__main__":
    unittest.main()
