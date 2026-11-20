"""Standalone tests for StenosisUtils.depth_estimation (no 3D Slicer required).

Run from the 2.0 directory with either:
    python -m unittest discover -s tests -v
    python -m pytest tests
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from StenosisUtils.depth_estimation import (  # noqa: E402
    DepthEstimator,
    Discriminator,
    GeneratorResNet,
    ImageDataset,
    ResidualBlock,
    weights_init_normal,
)
import torchvision.transforms as transforms  # noqa: E402

WEIGHTS_PATH = os.path.join(MODULE_DIR, "Models", "6Level-ex-vivo-G_AB.pth")


def _synthetic_airway(size=200, seed=0):
    """Grayscale image: bright frame with a dark circular lumen in the middle."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    img = np.clip(220 - (r < size / 4) * 180 + rng.normal(0, 5, (size, size)), 0, 255)
    return img.astype(np.uint8)


class ImageDatasetTest(unittest.TestCase):
    def test_single_item_tensor(self):
        arr = _synthetic_airway(32)
        ds = ImageDataset(arr, transforms_=[transforms.ToTensor()])
        self.assertEqual(len(ds), 1)
        item = ds[0]["A"]
        self.assertEqual(tuple(item.shape), (1, 32, 32))
        self.assertEqual(item.dtype, torch.float32)
        self.assertGreaterEqual(item.min().item(), 0.0)
        self.assertLessEqual(item.max().item(), 1.0)
        np.testing.assert_allclose(item[0].numpy(), arr / 255.0, atol=1e-6)


class WeightsInitTest(unittest.TestCase):
    def test_conv_and_batchnorm_init(self):
        torch.manual_seed(0)
        conv = nn.Conv2d(4, 64, 3)
        bn = nn.BatchNorm2d(64)
        nn.init.constant_(conv.bias, 5.0)
        weights_init_normal(conv)
        weights_init_normal(bn)
        self.assertTrue(torch.all(conv.bias == 0))
        self.assertLess(abs(conv.weight.std().item() - 0.02), 0.005)
        self.assertLess(abs(bn.weight.mean().item() - 1.0), 0.01)
        self.assertTrue(torch.all(bn.bias == 0))

    def test_other_layers_untouched(self):
        lin = nn.Linear(3, 3)
        before = lin.weight.clone()
        weights_init_normal(lin)
        self.assertTrue(torch.equal(before, lin.weight))


class NetworkShapeTest(unittest.TestCase):
    def test_residual_block_preserves_shape(self):
        x = torch.randn(1, 8, 16, 16)
        self.assertEqual(ResidualBlock(8)(x).shape, x.shape)

    def test_generator_output_matches_input(self):
        gen = GeneratorResNet((1, 200, 200), num_residual_blocks=1).eval()
        with torch.no_grad():
            y = gen(torch.rand(1, 1, 200, 200))
        self.assertEqual(tuple(y.shape), (1, 1, 200, 200))
        # Tanh output range
        self.assertGreaterEqual(y.min().item(), -1.0)
        self.assertLessEqual(y.max().item(), 1.0)

    def test_discriminator_output_shape(self):
        disc = Discriminator((1, 64, 64))
        self.assertEqual(disc.output_shape, (1, 4, 4))
        with torch.no_grad():
            y = disc(torch.rand(2, 1, 64, 64))
        self.assertEqual(tuple(y.shape[1:]), disc.output_shape)


@unittest.skipUnless(os.path.isfile(WEIGHTS_PATH), "model weights not present")
class DepthEstimatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.estimator = DepthEstimator()

    def test_loads_default_weights_in_eval_mode(self):
        self.assertFalse(self.estimator.G_AB.training)
        self.assertEqual((self.estimator.img_height, self.estimator.img_width), (200, 200))

    def test_generate_depth_returns_grayscale_image(self):
        out = self.estimator.generateDepth(Image.fromarray(_synthetic_airway(), mode="L"))
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.mode, "L")
        self.assertEqual(out.size, (200, 200))

    def test_generate_depth_is_deterministic(self):
        img = Image.fromarray(_synthetic_airway(seed=1), mode="L")
        a = np.array(self.estimator.generateDepth(img))
        b = np.array(self.estimator.generateDepth(img))
        np.testing.assert_array_equal(a, b)

    def test_generate_depth_responds_to_input(self):
        flat = np.array(self.estimator.generateDepth(Image.new("L", (200, 200), 128)))
        airway = np.array(self.estimator.generateDepth(Image.fromarray(_synthetic_airway(), mode="L")))
        self.assertFalse(np.array_equal(flat, airway))
        self.assertGreater(airway.std(), 0)

    def test_explicit_model_dir(self):
        est = DepthEstimator(model_dir=os.path.dirname(WEIGHTS_PATH))
        self.assertFalse(est.G_AB.training)


class DepthEstimatorMissingWeightsTest(unittest.TestCase):
    def test_missing_weights_raises(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError):
                DepthEstimator(model_dir=empty_dir, n_residual_blocks=1)


if __name__ == "__main__":
    unittest.main()
