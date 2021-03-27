"""Extract and match features."""
import math
import os
import numpy as np

import scipy.ndimage as ndi
import cv2

DSIZE = 8  # descriptor size


def gaussian_filter(img, sigma=1.0):
    """Compute the kernel size from sigma and smooths the image."""
    ksz = max(int((sigma - 0.35) / 0.15), 1)
    ksz += not ksz % 2  # must be odd
    return cv2.GaussianBlur(img, (ksz, ksz), sigma, sigma)


# from: https://github.com/BAILOOL/ANMS-Codes/blob/master/Python/ssc.py
def ssc(keypoints, im_size, n_points, tol=0.1):
    # pylint: disable=too-many-locals
    """Fast Adaptive Non-Maxima Suppression [1].

    [1] Bailo, Oleksandr, et al. "Efficient adaptive non-maximal suppression
    algorithms for homogeneous spatial keypoint distribution."
    Pattern Recognition Letters 106 (2018): 53-60.
    """
    cols, rows = im_size

    def _high():
        """Top range for binary search."""
        exp1 = rows + cols + 2 * n_points
        exp2 = (4 * cols
                + 4 * n_points
                + 4 * rows * n_points
                + rows * rows
                + cols * cols
                - 2 * rows * cols
                + 4 * rows * cols * n_points)
        exp3 = math.sqrt(exp2)
        exp4 = n_points - 1

        sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
        sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

        return max(sol1, sol2)

    # binary search range initialization with positive solutions
    high = _high()
    low = math.floor(math.sqrt(len(keypoints) / n_points))

    prev_width, complete, k = -1, False, n_points
    k_min, k_max = round(k - (k * tol)), round(k + (k * tol))

    result = []
    while not complete:
        width = low + (high - low) / 2
        # avoid repeating the same radius twice
        if (width == prev_width or low > high):
            # return the keypoints from the previous iteration
            break

        cgr = width / 2  # initializing the grid
        n_cell_cols = int(math.floor(cols / cgr))
        n_cell_rows = int(math.floor(rows / cgr))
        covered_vec = np.full((n_cell_rows+1, n_cell_cols+1), False)

        result = []
        for i, kpt in enumerate(keypoints):
            # get position of the cell current point is located at
            row = int(math.floor(kpt[1] / cgr))
            col = int(math.floor(kpt[0] / cgr))
            if not covered_vec[row][col]:  # if the cell is not covered
                result.append(i)
                # get range which current radius is covering
                row_min = int(max(row - math.floor(width / cgr), 0))
                row_max = int(min(row + math.floor(width / cgr), n_cell_rows))
                col_min = int(max(col - math.floor(width / cgr), 0))
                col_max = int(min(col + math.floor(width / cgr), n_cell_cols))
                # cover cells within the square bounding box with width w
                covered_vec[row_min:row_max+1, col_min:col_max+1] = True

        if k_min <= len(result) <= k_max:  # solution found
            complete = True
        elif len(result) < k_min:
            high = width - 1  # update binary search range
        else:
            low = width + 1
        prev_width = width

    return [keypoints[res] for res in result]


def rot_mat(theta, pp_):
    """2D rotation matrix for the given angle."""
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[cos, sin, pp_[1]], [-sin, cos, pp_[0]], [0, 0, 1]],
                    dtype="float32")


def descriptors(src, xx_, yy_, scale):
    """Get oriented MSOP descriptors."""
    points, desc = [], []
    g_x = gaussian_filter(cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3))
    g_y = gaussian_filter(cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3))
    blurred = gaussian_filter(src, 2.0)

    for pp_ in zip(xx_, yy_):
        theta = np.arctan2(g_x[pp_], g_y[pp_])
        points.append(tuple(scale*p for p in pp_) + (theta, scale))

        rmat = np.linalg.inv(rot_mat(theta, pp_))
        rmat[:2, 2] += DSIZE / 2  # center patch
        tile = cv2.warpPerspective(blurred, rmat, (DSIZE, DSIZE),
                                   flags=cv2.INTER_LINEAR)
        desc.append(tile.astype("uint8"))

    return points, desc


def msop(img, max_feat=(500, 50, 25, 10)):
    """Extract MSOP features."""
    gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    points, descs = [], []

    for lvl, maxf in enumerate(max_feat):
        # neighborhood size, Sobel aperture and trace coefficient
        hrs = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        loc_max = np.where(ndi.maximum_filter(hrs, size=3) == hrs)
        idx = np.argsort(hrs[loc_max])[-maxf*20:]  # slack for radii selection

        x_lvl, y_lvl = loc_max
        x_lvl, y_lvl = x_lvl[idx], y_lvl[idx]

        pts = ssc(np.stack([x_lvl, y_lvl], axis=1), gray.shape, maxf)
        x_lvl, y_lvl = np.stack(pts, axis=1)

        pts, dsc = descriptors(gray, x_lvl, y_lvl, 2**lvl)
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
    """Plot the first 100 descriptors."""
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
