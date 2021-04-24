"""Extract and match features.

Loosely based on: https://momohuang.github.io/Panorama/
"""
import argparse
import logging
import math
import os
from collections import defaultdict

import numpy as np
import scipy.ndimage as ndi
import cv2

DSIZE = 8         # descriptor size
N_MIN_MATCH = 8   # minimum number of point matches


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


def _msop_descriptors(src, xx_, yy_, scale):
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
        desc.append(tile.flatten())

    desc = np.array(desc)    # normalize
    desc = (desc - np.mean(desc, axis=1, keepdims=True)) / (
        np.std(desc, axis=1, keepdims=True) + 1e-8)

    return points, desc


def msop_detect(img, max_feat=(5000, 100, 25, 10)):
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

        pts, dsc = _msop_descriptors(gray, x_lvl, y_lvl, 2**lvl)
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
    else:
        descs = descs[:n_tiles]

    descs = descs.reshape((side, side, DSIZE, DSIZE)).transpose((0, 2, 1, 3))
    tiles = descs.reshape((side * DSIZE, side * DSIZE))
    tiles = 255 * (tiles - tiles.min())/(tiles.max() - tiles.min())

    return cv2.resize(tiles.astype(np.uint8), None, fx=4, fy=4,
                      interpolation=cv2.INTER_NEAREST)


def sift_detector():
    """Closure, return a SIFT detecting function."""
    sift = cv2.xfeatures2d.SIFT_create()

    def _detect(img):
        kp_, des = sift.detectAndCompute(img, None)
        des = np.sqrt(des/(des.sum(axis=1, keepdims=True) + 1e-7))  # RootSIFT
        return kp_, des

    return _detect


def msop_detector(max_feat=(5000, 100, 25, 10)):
    """Closure, returns a MSOP detector."""
    def _detect(img):
        kp_, des = msop_detect(img, max_feat)
        kp_ = [cv2.KeyPoint(p[1], p[0], p[2]) for p in kp_]
        des = des.reshape(-1, 64)

        return kp_, des
    return _detect


#
# Feature matching
#

FLANN_INDEX_KDTREE = 0   # FLANN strategy
FLANN_INDEX_LSH = 6


def flann_matching(des1, des2):
    """Given 2 lists of descriptors, match them with FLANN."""
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    return [m for m, n in matches if m.distance < 0.7*n.distance]


def _match_hom(pt1, pt2, des1, des2):
    """Match points, estimate homography and return inlier matches."""
    good = flann_matching(des1, des2)
    match = np.int32([(m.queryIdx, m.trainIdx) for m in good])
    if len(match) < N_MIN_MATCH:
        return None, None

    query_pts = np.float32([pt1[m] for m, _ in match])
    train_pts = np.float32([pt2[m] for _, m in match])
    hom, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC)

    return match[mask].squeeze(), hom


def _reverse(match, hom):
    """Find the matches and homography for the image is reverse order."""
    return np.fliplr(match), np.linalg.inv(hom)


def matching(imgs, detect=sift_detector()):
    """Find correspondences between images in a list."""
    kpts, descs = [], []
    for i, img in enumerate(imgs):
        logging.debug(f"Processing image #{i+1}")

        kp_, des = detect(img)
        cent = np.array([img.shape[1], img.shape[0]])
        kpts.append(np.float32([kp.pt - cent for kp in kp_]))
        descs.append(des)

    # match all possible pairs
    matches, n_imgs = defaultdict(dict), len(imgs)
    for src in range(n_imgs):
        for dst in range(src+1, n_imgs):
            logging.debug(f"Matching {src+1}-{dst+1}")
            match, hom = _match_hom(
                kpts[src], kpts[dst], descs[src], descs[dst])
            if hom is None:
                continue

            matches[src][dst] = (match, hom)
            matches[dst][src] = _reverse(match, hom)

    return np.array(kpts, dtype=np.object), np.array(matches, dtype=np.object)


def match_images(img1, img2, detect=sift_detector()):
    """Find MSOP correspondences between images."""
    (kp1, des1), (kp2, des2) = detect(img1), detect(img2)
    good = flann_matching(des1, des2)

    src_pts = np.array([kp1[m.queryIdx].pt for m in good]).reshape((-1, 1, 2))
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good]).reshape((-1, 1, 2))
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2,
                       matchesMask=mask)
    im_match = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imshow("Matches", im_match)


# MSOP from: https://github.com/momohuang/Panorama/blob/master/MSOP.cpp
def main():
    """Script entry point."""
    parser = argparse.ArgumentParser(description="Extract features.")
    parser.add_argument('--path', type=str, default="../data/ppwwyyxx/CMU1",
                        help="directory with the images to process.")
    args = parser.parse_args()

    exts = [".jpg", ".png", ".bmp"]
    exts += [ex.upper() for ex in exts]

    name = os.path.basename(args.path)
    files = [f for f in os.listdir(args.path)
             if any([f.endswith(ext) for ext in exts])]

    imgs = [cv2.imread(os.path.join(args.path, f)) for f in files]
    imgs = [cv2.resize(im, None, fx=0.5, fy=0.5) for im in imgs]

    kpts, matches = matching(imgs)
    np.savez(f"matches_{name}.npz", kpts=kpts, matches=matches)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
