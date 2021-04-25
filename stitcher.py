"""Stitches a set of images into a single panorama."""
import argparse
import logging
import os

import numpy as np
import cv2

from features import matching
from bundle_adj import traverse

MAX_RESOLUTION = 1400


#
# Reprojection
#

class SphProj:
    """Forward and backward spherical projection."""

    @staticmethod
    def hom2proj(pts):
        """Project the points in spherical coordinates."""
        hypot = np.sqrt(pts[:, 0]**2 + pts[:, 2]**2)
        return np.stack([np.arctan2(pts[:, 0], pts[:, 2]),
                        np.arctan2(pts[:, 1], hypot)], axis=-1)

    @staticmethod
    def proj2hom(pts):
        """Recover projective points from spherical coordinates."""
        return np.stack([np.sin(pts[:, 0]), np.tan(pts[:, 1]),
                         np.cos(pts[:, 0])], axis=-1)


class CylProj:
    """Forward and backward cylidrical projection."""

    @staticmethod
    def hom2proj(pts):
        """Project the points in cylindrical coordinates."""
        hypot = np.sqrt(pts[:, 0]**2 + pts[:, 2]**2)
        return np.stack([np.arctan2(pts[:, 0], pts[:, 2]),
                        pts[:, 1]/hypot], axis=-1)

    @staticmethod
    def proj2hom(pts):
        """Recover projective points from cylindrical coordinates."""
        return np.stack([np.sin(pts[:, 0]), pts[:, 1], np.cos(pts[:, 0])],
                        axis=-1)


def _proj_img_range_border(shape, hom):
    """Estimate the extent of the image after projection."""
    nel = 100
    height, width = shape

    side_x = np.linspace(0, width, nel)
    side_y = np.linspace(0, height, nel)
    borders = np.concatenate([
        np.stack([np.zeros(nel), side_y, np.ones(nel)], axis=1),
        np.stack([np.full(nel, width), side_y, np.ones(nel)], axis=1),
        np.stack([side_x, np.zeros(nel), np.ones(nel)], axis=1),
        np.stack([side_x, np.full(nel, height), np.ones(nel)], axis=1)])
    borders = borders - np.array([width/2, height/2, 0])

    pts = SphProj.hom2proj(hom.dot(borders.T).T)
    return np.min(pts, axis=0), np.max(pts, axis=0)   # range


def _proj_img_range_corners(shape, hom):
    """Estimate image extent from corners with check for angle wraparound."""
    height, width = shape
    pts = np.array([[-width/2, -height/2, 1], [width/2, -height/2, 1],
                    [-width/2, height/2, 1], [width/2, height/2, 1]])
    pts = SphProj.hom2proj(hom.dot(pts.T).T)

    xmin, xmax = min(pts[0, 0], pts[2, 0]), max(pts[1, 0], pts[3, 0])
    ymin, ymax = min(pts[0, 1], pts[1, 1]), max(pts[2, 1], pts[3, 1])
    if xmin > xmax:  # push to right
        xmax += 2 * np.pi
    if ymin > ymax:  # push on top
        ymax += np.pi

    return np.array([xmin, ymin]), np.array([xmax, ymax])


def estimate_resolution(regions):
    """Estimate the resolution of the final image."""
    min_r, max_r = zip(*[reg.range for reg in regions])
    min_r, max_r = np.min(min_r, axis=0), np.max(max_r, axis=0)
    size = max_r - min_r

    mid = regions[len(regions) // 2]   # central image
    im_shape = np.array(mid.img.shape[:2][::-1])
    resolution = (mid.range[1] - mid.range[0]) / im_shape

    max_side = np.max(size / resolution)
    if max_side > MAX_RESOLUTION:
        resolution *= max_side / MAX_RESOLUTION

    return resolution, (min_r, max_r)


def stitch(regions):
    """Stitch the images together."""
    for reg in regions:
        reg.range = _proj_img_range_border(reg.img.shape[:2], reg.hom())

    resolution, im_range = estimate_resolution(regions)
    target = (im_range[1] - im_range[0]) / resolution

    shape = tuple(int(t) for t in np.round(target))[::-1]  # y,x order
    mosaic = np.zeros(shape + (3,), dtype=np.uint8)        # RGBA image
    for idx, reg in enumerate(regions):
        bottom = np.round((reg.range[0] - im_range[0])/resolution)
        top = np.round((reg.range[1] - im_range[0])/resolution)
        bottom, top = bottom.astype(np.int32), top.astype(np.int32)
        hh_, ww_ = reg.img.shape[:2]  # original image shape

        # find pixel coordinates
        y_i, x_i = np.indices((top[1]-bottom[1], top[0]-bottom[0]))
        x_i = (x_i + bottom[0]) * resolution[0] + im_range[0][0]
        y_i = (y_i + bottom[1]) * resolution[1] + im_range[0][1]
        xx_ = SphProj.proj2hom(np.stack([x_i, y_i], axis=-1).reshape(-1, 2))

        # transform to the original image coordinates
        xx_ = reg.proj().dot(xx_.T).T.astype(np.float32)
        xx_ = xx_.reshape(top[1]-bottom[1], top[0]-bottom[0], -1)
        mask = xx_[..., -1] < 0  # behind the screen

        x_pr = xx_[..., :-1] / xx_[..., [-1]] + np.float32([ww_/2, hh_/2])
        mask |= (x_pr[..., 0] < 0) | (x_pr[..., 0] >= ww_) | \
                (x_pr[..., 1] < 0) | (x_pr[..., 1] >= hh_)
        x_pr[mask] = -1

        # paste only valid pixels
        warped = cv2.remap(reg.img, x_pr[:, :, 0], x_pr[:, :, 1],
                           cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        tile = mosaic[bottom[1]:top[1], bottom[0]:top[0]]
        mosaic[bottom[1]:top[1], bottom[0]:top[0]] = np.where(
            mask[..., None], tile, warped)

    return mosaic


def idx_to_keypoints(matches, kpts):
    """Replace keypoint indices with their coordinates."""
    def _i_to_k(matches, kpt1, kpt2):
        return np.concatenate([kpt1[matches[:, 0]], kpt2[matches[:, 1]]],
                              axis=1)

    # homogeneous coordinates
    kpts = [np.concatenate([kp, np.ones((kp.shape[0], 1))], axis=1)
            for kp in kpts]

    matches = matches.item()   # unpack dictionary
    # add match confidence (number of inliers)
    matches = {i: {j: (_i_to_k(m, kpts[i], kpts[j]), h, len(m))
                   for j, (m, h) in col.items()} for i, col in matches.items()}

    return matches


def main():
    """Script entry point."""
    parser = argparse.ArgumentParser(description="Stitch images.")
    parser.add_argument('--path', type=str, default="../data/ppwwyyxx/CMU2",
                        help="directory with the images to process.")
    args = parser.parse_args()

    exts = [".jpg", ".png", ".bmp"]
    exts += [ex.upper() for ex in exts]

    name = os.path.basename(args.path)
    files = [f for f in os.listdir(args.path)
             if any([f.endswith(ext) for ext in exts])]

    imgs = [cv2.imread(os.path.join(args.path, f)) for f in files]
    imgs = [cv2.resize(im, None, fx=0.5, fy=0.5) for im in imgs]

    try:  # cache
        arr = np.load(f"matches_{name}.npz", allow_pickle=True)
        kpts, matches = arr['kpts'], arr['matches']
    except IOError:
        kpts, matches = matching(imgs)
        np.savez(f"matches_{name}.npz", kpts=kpts, matches=matches)

    regions = traverse(imgs, idx_to_keypoints(matches, kpts))
    mosaic = stitch(regions)
    cv2.imshow("Mosaic", mosaic)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
