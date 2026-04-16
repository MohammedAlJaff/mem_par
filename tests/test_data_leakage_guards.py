import unittest

try:
    import torch
    from mp_config import clone_config, get_base_config
    from mp_data import validate_data_bundle
except ModuleNotFoundError:
    torch = None
    clone_config = None
    get_base_config = None
    validate_data_bundle = None


@unittest.skipIf(
    torch is None or validate_data_bundle is None,
    "torch and data validation helpers are required",
)
class TestDataLeakageGuards(unittest.TestCase):
    def make_config(self):
        config = clone_config(get_base_config())
        config.data.dataset = "guard_demo"
        return config

    def make_bundle(self, *, same_source=True, overlap=False):
        source = "demo:train"
        reference_source = source if same_source else "demo:test"
        reference_indices = torch.tensor([2, 3]) if overlap else torch.tensor([4, 5])
        return {
            "train": torch.tensor([[0.0], [1.0]]),
            "reference": torch.tensor([[2.0], [3.0]]),
            "train_source_id": source,
            "reference_source_id": reference_source,
            "train_indices": torch.tensor([0, 2]),
            "reference_indices": reference_indices,
        }

    def test_same_source_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, "indices overlap"):
            validate_data_bundle(
                self.make_bundle(same_source=True, overlap=True),
                self.make_config(),
            )

    def test_same_source_disjoint_fails_by_default(self):
        with self.assertRaisesRegex(ValueError, "same source"):
            validate_data_bundle(
                self.make_bundle(same_source=True, overlap=False),
                self.make_config(),
            )

    def test_same_source_disjoint_requires_explicit_opt_in(self):
        config = self.make_config()
        config.data.allow_same_source_reference = True

        bundle = validate_data_bundle(
            self.make_bundle(same_source=True, overlap=False),
            config,
        )

        self.assertTrue(bundle["data_validation"]["same_source"])
        self.assertTrue(bundle["data_validation"]["same_source_allowed"])
        self.assertEqual(bundle["data_validation"]["overlap_count"], 0)

    def test_identical_full_tensors_fail_even_for_different_sources(self):
        bundle = self.make_bundle(same_source=False, overlap=False)
        bundle["reference"] = bundle["train"].clone()

        with self.assertRaisesRegex(ValueError, "exactly equal"):
            validate_data_bundle(bundle, self.make_config())


if __name__ == "__main__":
    unittest.main()
