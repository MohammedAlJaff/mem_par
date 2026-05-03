import unittest

try:
    from mp_visualization import DEFAULT_IMAGE_GRID_MAX_COLUMNS, _image_grid_layout
except ModuleNotFoundError:
    DEFAULT_IMAGE_GRID_MAX_COLUMNS = None
    _image_grid_layout = None


@unittest.skipIf(
    _image_grid_layout is None,
    "visualization helpers are required",
)
class TestVisualization(unittest.TestCase):
    def test_image_grid_layout_wraps_rows_at_default_columns(self):
        n_cols, visual_rows = _image_grid_layout(
            [23, 11],
            DEFAULT_IMAGE_GRID_MAX_COLUMNS,
        )

        self.assertEqual(n_cols, 10)
        self.assertEqual(visual_rows, [3, 2])

    def test_image_grid_layout_keeps_small_rows_compact(self):
        n_cols, visual_rows = _image_grid_layout([4, 2], 10)

        self.assertEqual(n_cols, 4)
        self.assertEqual(visual_rows, [1, 1])


if __name__ == "__main__":
    unittest.main()
