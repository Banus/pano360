"""Test critical panorama stitching functions."""
import unittest
import numpy as np
import numpy.testing as npt

import blend as bl
from bundle_adj import get_focal, rotation_to_mat, mat_to_angle


class TestHomography(unittest.TestCase):
    """Test homography decomposition."""

    @staticmethod
    def test_is_rotation():
        """Test if matrix from exp conversion is a valid rotation."""
        rad = np.random.randn(3)
        rot = rotation_to_mat(rad)
        npt.assert_almost_equal(rot.T.dot(rot), np.eye(3))
        npt.assert_almost_equal(mat_to_angle(rot), rad)

    def test_focal(self):
        """Test extraction of focal from rotation + projection matrix."""
        kint = bl.intrinsics(1e3)
        hom = kint.dot(rotation_to_mat().dot(np.linalg.inv(kint)))

        self.assertAlmostEqual(get_focal(hom), 1e3)
        self.assertAlmostEqual(get_focal(np.linalg.inv(hom)), 1e3)


class TestWarp(unittest.TestCase):
    """Test warping functions."""

    @staticmethod
    def test_spherical_ok():
        """Check that forward + backward conversions is the identity."""
        pts = np.random.randn(10, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)

        new_pts = bl.SphProj.proj2hom(bl.SphProj.hom2proj(pts))
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        npt.assert_almost_equal(new_pts, pts)

    @staticmethod
    def test_cylindrical_ok():
        """Check that forward + backward conversions is the identity."""
        pts = np.random.randn(10, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)

        new_pts = bl.CylProj.proj2hom(bl.CylProj.hom2proj(pts))
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        npt.assert_almost_equal(new_pts, pts)


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()
