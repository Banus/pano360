"""Test critical panorama stitching functions."""
import unittest
import numpy as np
import numpy.testing as npt

import bundle_adj as ba
from stitcher import SphProj, CylProj, find_gains


class TestHomography(unittest.TestCase):
    """Test homography decomposition."""

    @staticmethod
    def test_is_rotation():
        """Test if matrix from exp conversion is a valid rotation."""
        rad = np.random.randn(3)
        rot = ba.rotation_to_mat(rad)
        npt.assert_almost_equal(rot.T.dot(rot), np.eye(3))
        npt.assert_almost_equal(ba.mat_to_angle(rot), rad)

    def test_focal(self):
        """Test extraction of focal from rotation + projection matrix."""
        kint = ba.intrinsics(1e3)
        hom = kint.dot(ba.rotation_to_mat().dot(np.linalg.inv(kint)))

        self.assertAlmostEqual(ba.get_focal(hom), 1e3)
        self.assertAlmostEqual(ba.get_focal(np.linalg.inv(hom)), 1e3)

    @staticmethod
    def test_camera_inverse():
        """Test if camera transform and its inverse are correct."""
        cam = ba.Image(None, ba.rotation_to_mat(), ba.intrinsics(1e3))
        npt.assert_almost_equal(cam.hom().dot(cam.proj()), np.eye(3))

    @staticmethod
    def test_straighten():
        """Test if straightening correctly recovers the vertical."""
        n_cams = 10
        step = np.array([0, 1, 0]) * 2 * np.pi / n_cams
        rots = [ba.rotation_to_mat(step * i) for i in range(n_cams)]

        tilt = ba.rotation_to_mat(np.array([0.1, 0, 0]))
        new_rots = [rot.dot(tilt) for rot in rots]     # change the vertical
        new_rots = np.stack(ba.straighten(new_rots), axis=0)
        new_rots = new_rots.dot(np.diag([-1, -1, 1]))  # orientation ambiguity
        npt.assert_almost_equal(new_rots, np.stack(rots, axis=0))

    @staticmethod
    def test_camera_params():
        """Test if conversion to camera and back gives the identity."""
        params = np.random.randn(6)
        new_params = ba.camera_to_params(ba.params_to_camera(params))
        npt.assert_almost_equal(new_params, params)


class TestWarp(unittest.TestCase):
    """Test warping functions."""

    @staticmethod
    def test_spherical_ok():
        """Check that forward + backward conversions is the identity."""
        pts = np.random.randn(10, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)

        new_pts = SphProj.proj2hom(SphProj.hom2proj(pts))
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        npt.assert_almost_equal(new_pts, pts)

    @staticmethod
    def test_cylindrical_ok():
        """Check that forward + backward conversions is the identity."""
        pts = np.random.randn(10, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)

        new_pts = CylProj.proj2hom(CylProj.hom2proj(pts))
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        npt.assert_almost_equal(new_pts, pts)

    @staticmethod
    def test_gain_correction():
        """Checks that the real gains are correctly recovered."""
        size = 10
        gains = 1 + 0.1 * np.random.randn(size)
        overlaps = 100 + 10 * np.random.randn(size, size)
        # simulate random voerlaps with consisten gain differences
        for i in range(size):
            for j in range(i+1, size):
                overlaps[i, j] = overlaps[j, i] * gains[j] / gains[i]

        # sanity check: verify that gains is the solution
        npt.assert_almost_equal(gains[:, None] * overlaps,
                                gains[None, :] * overlaps.T)

        sizes = np.random.randn(size, size) + 10
        ratio = find_gains(overlaps, sizes) / gains
        npt.assert_almost_equal(ratio, np.full(size, ratio[0]))


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()
