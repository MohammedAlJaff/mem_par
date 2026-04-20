import contextlib
import unittest

from mp_config import get_base_config
from mp_runtime import autocast_context, configure_torch_runtime


class TestRuntimeNumerics(unittest.TestCase):
    def test_fp32_cpu_autocast_is_noop(self):
        config = get_base_config()
        config.runtime.precision = "fp32"

        metadata = configure_torch_runtime(config, "cpu")
        context = autocast_context(config, "cpu")

        self.assertIsInstance(context, contextlib.nullcontext)
        self.assertEqual(metadata["precision"], "fp32")
        self.assertFalse(metadata["tf32_enabled"])
        self.assertEqual(metadata["autocast_dtype"], "none")

    def test_bf16_rejects_cpu(self):
        config = get_base_config()
        config.runtime.precision = "bf16"

        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            configure_torch_runtime(config, "cpu")

    def test_pin_memory_resolves_to_nonblocking_only_for_cuda(self):
        config = get_base_config()
        config.runtime.precision = "fp32"
        config.runtime.pin_memory = True

        metadata = configure_torch_runtime(config, "cpu")

        self.assertFalse(metadata["pin_memory"])
        self.assertFalse(metadata["non_blocking_h2d"])


if __name__ == "__main__":
    unittest.main()
