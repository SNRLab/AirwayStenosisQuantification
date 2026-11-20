"""Standalone tests for StenosisUtils.image_utils (no 3D Slicer required)."""
import os
import sys
import unittest

import numpy as np

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from StenosisUtils.image_utils import toGrayscale2D  # noqa: E402


class ToGrayscale2DTest(unittest.TestCase):
    def setUp(self):
        self.img = np.arange(6 * 4, dtype=np.uint8).reshape(6, 4)

    def assertImage(self, result, expected):
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_array_equal(result, expected)

    def test_2d_passthrough(self):
        self.assertImage(toGrayscale2D(self.img), self.img)

    def test_leading_singleton(self):
        self.assertImage(toGrayscale2D(self.img[np.newaxis, ...]), self.img)

    def test_trailing_singleton(self):
        self.assertImage(toGrayscale2D(self.img[..., np.newaxis]), self.img)

    def test_square_layouts_agree(self):
        img = np.random.default_rng(0).integers(0, 256, (1080, 1080), dtype=np.uint8)
        a = toGrayscale2D(img.reshape(1, 1080, 1080))
        b = toGrayscale2D(img.reshape(1080, 1080, 1))
        self.assertImage(a, img)
        self.assertImage(b, img)

    def test_rgb_leading_singleton(self):
        rgb = np.stack([self.img, self.img + 3, self.img + 6], axis=-1)[np.newaxis, ...]
        self.assertImage(toGrayscale2D(rgb), self.img + 3)

    def test_rgb_trailing_singleton(self):
        rgb = np.stack([self.img, self.img + 3, self.img + 6], axis=-1)[:, :, np.newaxis, :]
        self.assertImage(toGrayscale2D(rgb), self.img + 3)

    def test_rgba_ignores_alpha(self):
        alpha = np.full_like(self.img, 255)
        rgba = np.stack([self.img, self.img, self.img, alpha], axis=-1)[np.newaxis, ...]
        self.assertImage(toGrayscale2D(rgba), self.img)

    def test_single_component_vector(self):
        self.assertImage(toGrayscale2D(self.img[np.newaxis, ..., np.newaxis]), self.img)

    def test_multislice_takes_first_slice(self):
        vol = np.stack([self.img, self.img + 1, self.img + 2])
        self.assertImage(toGrayscale2D(vol), self.img)

    def test_float_input_cast_to_uint8(self):
        self.assertImage(toGrayscale2D(self.img.astype(np.float32)[np.newaxis]), self.img)

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            toGrayscale2D(np.zeros(5))
        with self.assertRaises(ValueError):
            toGrayscale2D(np.zeros((2, 2, 2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
