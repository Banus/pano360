"""Extract and match features."""
import os
import numpy as np

import scipy.ndimage as ndi
import cv2

DSIZE = 7  # descriptor size


def rot_mat(theta, pp_):
    """2D rotation matrix for the given angle."""
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[cos, sin, pp_[1]], [-sin, cos, pp_[0]], [0, 0, 1]],
                    dtype="float32")


def descriptors(src, xx_, yy_, scale):
    """Get oriented MSOP descriptors."""
    points, desc = [], []
    g_x = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=5)
    g_y = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=5)

    for pp_ in zip(xx_, yy_):
        theta = np.arctan2(g_x[pp_], g_y[pp_])
        points.append(tuple(scale*p for p in pp_) + (theta, scale))

        rmat = np.linalg.inv(rot_mat(theta, pp_))
        rmat[:2, 2] += DSIZE / 2  # center patch
        tile = cv2.warpPerspective(src, rmat, (DSIZE, DSIZE),
                                   flags=cv2.INTER_LINEAR)
        desc.append(tile.astype("uint8"))

    return points, desc


def msop(img, max_feat=(500, 50, 25, 10)):
    """Extract MSOP features."""
    gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    points, descs = [], []

    for lvl, maxf in enumerate(max_feat):
        # neighborhood size, Sobel aperture and trace coefficient
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        loc_max = np.where(ndi.maximum_filter(dst, size=3) == dst)
        idx = np.argsort(dst[loc_max])[-maxf:]

        x_lvl, y_lvl = loc_max
        pts, dsc = descriptors(gray, x_lvl[idx], y_lvl[idx], 2**lvl)
        points.append(pts)
        descs.append(dsc)

        gray = cv2.pyrDown(gray)
    return np.concatenate(points), np.concatenate(descs)


def plot_points(img, points):
    """Show points and the descriptor area."""
    rad = DSIZE / 2
    pts = np.array([[0, 0], [rad, 0], [rad, -rad], [-rad, -rad], [-rad, rad],
                    [rad, rad], [rad, 0]], dtype="float32")

    for pp_ in points:
        rmat = rot_mat(pp_[2], pp_[3] * pp_[:2])
        dst_pts = cv2.perspectiveTransform(pts[None, :] * pp_[3], rmat)
        dst_pts = dst_pts.squeeze().astype(np.int32)
        cv2.polylines(img, [dst_pts], False, color=(0, 0, 255), thickness=1)

    return img


def plot_descs(descs, side=25):
    """Plots the first 100 descriptors."""
    n_tiles = side * side
    if len(descs) < n_tiles:
        pad = np.zeros(
            (n_tiles - len(descs),) + descs.shape[1:]).astype("uint8")
        descs = np.concatenate([descs, pad])

    descs = descs.reshape((side, side, DSIZE, DSIZE)).transpose((0, 2, 1, 3))
    tiles = descs.reshape((side * DSIZE, side * DSIZE))
    return cv2.resize(tiles, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


# MSOP from: https://github.com/momohuang/Panorama/blob/master/MSOP.cpp
def main():
    """Script entry point."""
    base_path = "../data/ppwwyyxx/CMU2"

    img = cv2.imread(os.path.join(base_path, "medium00.JPG"))
    points, descs = msop(img)

    cv2.imshow('tiles', plot_descs(descs))
    cv2.imshow('dst', plot_points(img.copy(), points))
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
