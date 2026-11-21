"""Pure-Python tests for decoder math, rotation detection, and letterboxing."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidindex import common as c  # noqa: E402


class DecoderMathTests(unittest.TestCase):
    def test_bucket_of(self):
        k = 4
        self.assertEqual(c.bucket_of(0.0, k), 0)
        self.assertEqual(c.bucket_of(0.24, k), 0)
        self.assertEqual(c.bucket_of(0.25, k), 1)
        self.assertEqual(c.bucket_of(0.50, k), 2)
        self.assertEqual(c.bucket_of(0.75, k), 3)
        self.assertEqual(c.bucket_of(1.00, k), 4)

    def test_rot90_k(self):
        self.assertEqual(c.rot90_k(0), 0)
        self.assertEqual(c.rot90_k(90), 1)
        self.assertEqual(c.rot90_k(180), 2)
        self.assertEqual(c.rot90_k(270), 3)
        self.assertEqual(c.rot90_k(-90), 3)
        self.assertEqual(c.rot90_k(-180), 2)
        self.assertEqual(c.rot90_k(-270), 1)

    def test_display_matrix_rotation(self):
        # Identity matrix: 0 degrees
        identity = [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824]
        self.assertAlmostEqual(c.display_matrix_rotation(identity), 0.0)

    def test_letterbox_params(self):
        # 16:9 1080p video into 640 input
        new_w, new_h, top, bottom, left, right = c.letterbox_params(1080, 1920, 640, 32)
        self.assertEqual(new_w, 640)
        self.assertEqual(new_h, 360)
        self.assertEqual(left, 0)
        self.assertEqual(right, 0)
        self.assertEqual(top, 12)
        self.assertEqual(bottom, 12)
        # Total height 360 + 12 + 12 = 384 (which is 12 * 32, stride aligned)
        self.assertEqual((new_h + top + bottom) % 32, 0)
        self.assertEqual((new_w + left + right) % 32, 0)

    def test_short_error(self):
        err = ValueError("A" * 150)
        short = c.short_error(err, 50)
        self.assertTrue(len(short) <= 50)
        self.assertTrue(short.endswith("..."))


if __name__ == "__main__":
    unittest.main()
