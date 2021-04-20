"""Test critical panorama stitching functions."""
import unittest
import numpy as np
import numpy.testing as npt

from blend import intrinsics
from bundle_adj import get_focal, rotation_to_mat


class TestHomography(unittest.TestCase):
    """Test homography decomposition."""

    @staticmethod
    def test_is_rotation():
        """Test if matrix from exp conversion is a valid rotation."""
        rot = rotation_to_mat()
        npt.assert_almost_equal(rot.T.dot(rot), np.eye(3))

    def test_focal(self):
        """Test extraction of focal from rotation + projection matrix."""
        kint = intrinsics(1e3, (0, 0))
        hom = kint.dot(rotation_to_mat().dot(np.linalg.inv(kint)))

        self.assertAlmostEqual(get_focal(hom), 1e3)
        self.assertAlmostEqual(get_focal(np.linalg.inv(hom)), 1e3)


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()
