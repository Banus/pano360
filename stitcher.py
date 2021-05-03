"""Stitches a set of images into a single panorama.

Adapted from the C++ implementation in https://github.com/ppwwyyxx/OpenPano
"""
import argparse
import logging
import os
import pickle
import time

import numpy as np
import cv2

from features import matching
from bundle_adj import _hom_to_from, traverse

MAX_RESOLUTION = 1400


#
# Exposure adjustment
#

def find_gains(overlaps, sizes, stdn=0.1, stdg=2):
    """Find the gains minimizing discrepancies between mean intensities.

    Solving eq (29) in [1].
    """
    nsize1, nsize2 = (sizes+sizes.T)/(stdn*stdn), sizes/(stdg*stdg)
    aa_ = np.diag(np.sum(nsize1 * overlaps * overlaps + nsize2, axis=1))
    aa_ -= nsize1 * overlaps * overlaps.T

    return np.linalg.solve(aa_, np.sum(nsize2, axis=1))


def equalize_gains(regions):
    """Equalize the exposures by minimizing differences on overlaps."""
    n_imgs = len(regions)
    overlaps, sizes = np.zeros((n_imgs, n_imgs)), np.zeros((n_imgs, n_imgs))

    height, width = regions[0].img.shape[:2]
    tr_ = np.array([[1, 0, width/2], [0, 1, height/2], [0, 0, 1]])
    inv_tr = np.array([[1, 0, -width/2], [0, 1, -height/2], [0, 0, 1]])
    corners = np.array([[0, 0, 1], [width, 0, 1],
                        [width, height, 1], [0, height, 1]])

    logging.debug("Equalizing gain...")
    for i in range(n_imgs):
        for j in range(i+1, n_imgs):
            # translate image to (non-centered) pixel coordinates
            hom = tr_.dot(_hom_to_from(regions[i], regions[j])).dot(inv_tr)
            pts = hom.dot(corners.T).T

            if np.any(pts[:, 2] < 0):   # behind the screen, skip
                continue
            overlap = cv2.warpPerspective(regions[j].img, hom, (width, height),
                                          borderMode=cv2.BORDER_TRANSPARENT)
            mask = overlap[..., 3] != 0
            sizes[i, j] = sizes[j, i] = np.sum(mask)
            if sizes[i, j] == 0:  # no overlap
                continue
            overlaps[i, j] = np.mean(regions[i].img[mask, :3])
            overlaps[j, i] = np.mean(overlap[mask, :3])

    for reg, gain in zip(regions, find_gains(overlaps, sizes)):
        reg.img[..., :3] = np.clip(gain * reg.img[..., :3], 0, 1)


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
    mid_range = _proj_img_range_corners(mid.img.shape[:2], mid.hom())
    resolution = (mid_range[1] - mid_range[0]) / im_shape

    max_side = np.max(size / resolution)
    if max_side > MAX_RESOLUTION:
        resolution *= max_side / MAX_RESOLUTION

    return resolution, (min_r, max_r)


def no_blend(patches, shape):
    """Paste the patches to the mosaic without blending."""
    mosaic = np.zeros(shape + (3,), dtype=np.uint8)

    for warped, mask, irange in patches:
        mosaic[irange] = np.where(mask[..., None], mosaic[irange],
                                  (255 * warped[..., :3]).astype(np.uint8))

    return mosaic


def linear_blend(patches, shape):
    """Linearly blend patches."""
    mosaic = np.zeros(shape + (3,), dtype="float32")
    wsum = np.zeros(shape, dtype="float32")  # normalization
    for warped, mask, irange in patches:
        tile = np.where(mask[..., None], 0.0, warped[..., :3])
        mosaic[irange] += tile * warped[..., [3]]
        wsum[irange] += warped[..., 3]

    wsum[wsum == 0] = 1   # avoid division by zero
    mosaic /= wsum[..., None]

    return (255 * mosaic).astype(np.uint8)


def multiband_blend(patches, shape, n_levels=5):
    """
    Use multi-band blending [1] to merge patches.

    References
    ----------
    [1] Brown, Matthew, and David G. Lowe. "Automatic panoramic image stitching
    using invariant features." International journal of computer vision 74.1
    (2007): 59-73.
    """
    weights = np.zeros(shape + (len(patches),), dtype="float32")

    for idx, (warped, _, irange) in enumerate(patches):
        yrange, xrange = irange  # unpack to make numpy happy
        weights[yrange, xrange, idx] = warped[..., 3]
    # find maximum patch for each pixel
    valid = np.sum(weights, axis=-1) > 0
    weights = weights.argmax(axis=-1)
    weights[~valid] = -1

    # initialize sharp high-res masks for the patches
    for idx, (warped, _, irange) in enumerate(patches):
        warped[..., 3] = weights[irange] == idx

    # blur outside the valid region to reduce artifacts
    #  but then remove invalid pixels - compute only the first time
    allmask = np.zeros(shape, dtype=bool)

    mosaic = np.zeros(shape + (3,), dtype="float32")
    prevs = [None] * len(patches)
    for lvl in range(n_levels):
        logging.debug(f"Blending level #{lvl + 1}")
        sigma = np.sqrt(2*lvl + 1.0)*4
        layer = np.zeros(shape + (3,), dtype="float32")  # delta for this level
        wsum = np.zeros(shape, dtype="float32")
        is_last = lvl == (n_levels - 1)

        for idx, (warped, mask, irange) in enumerate(patches):
            tile = prevs[idx] if prevs[idx] is not None else warped.copy()
            if not is_last:
                blurwarp = cv2.GaussianBlur(warped, (0, 0), sigma)
                tile[..., :3] -= blurwarp[..., :3]
                tile[..., 3] = blurwarp[..., 3]   # avoid sharp masks
                prevs[idx] = blurwarp

            layer[irange] += tile[..., :3] * tile[..., [3]]
            wsum[irange] += tile[..., 3]
            if lvl == 0:
                allmask[irange] |= ~mask

        layer[~allmask, :] = 0
        wsum[wsum == 0] = 1
        mosaic += layer / wsum[..., None]

    mosaic = np.clip(mosaic, 0.0, 1.0)   # avoid saturation artifacts
    return (255 * mosaic).astype(np.uint8)


BLENDERS = {
    "none": no_blend,
    "linear": linear_blend,
    "multiband": multiband_blend
}


def _hat(size):
    """Triangular function 0-0.5-0 of a given size."""
    xx_ = np.arange(size) - size/2
    return 0.5 - np.abs(xx_ / size)


def _add_weights(img):
    """Add weights scaled as (x-0.5)*(y-0.5) in normalized coordinates."""
    img = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2RGBA)
    height, width = img.shape[:2]
    img[..., 3] = _hat(height)[:, None] * _hat(width)[None, :]

    return img


def _valid(patches, shape):
    """Area of validity (for crop)."""
    valid = np.zeros(shape, dtype=bool)
    for _, mask, irange in patches:
        valid[irange] |= ~mask
    return valid


def stitch(regions, blender=no_blend, equalize=False, crop=False):
    """Stitch the images together."""
    for reg in regions:
        reg.range = _proj_img_range_border(reg.img.shape[:2], reg.hom())
        reg.img = _add_weights(reg.img)

    if equalize:
        equalize_gains(regions)

    resolution, im_range = estimate_resolution(regions)
    target = (im_range[1] - im_range[0]) / resolution

    shape = tuple(int(t) for t in np.round(target))[::-1]  # y,x order
    patches = []
    for reg in regions:
        bottom = np.round((reg.range[0] - im_range[0])/resolution)
        top = np.round((reg.range[1] - im_range[0])/resolution)
        bottom, top = bottom.astype(np.int32), top.astype(np.int32)
        hh_, ww_ = reg.img.shape[:2]  # original image shape

        # pad image if multi-band to avoid sharp edges where the image ends
        if blender == multiband_blend:
            bottom = np.maximum(bottom - 10, np.int32([0, 0]))
            top = np.minimum(top + 10, target.astype(np.int32))

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
        mask |= (x_pr[..., 0] < 0) | (x_pr[..., 0] > ww_-1) | \
                (x_pr[..., 1] < 0) | (x_pr[..., 1] > hh_-1)

        # paste only valid pixels
        warped = cv2.remap(reg.img, x_pr[:, :, 0], x_pr[:, :, 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped[..., 3] = warped[..., 3] * (~mask)
        irange = np.s_[bottom[1]:top[1], bottom[0]:top[0]]
        patches.append((warped, mask, irange))

    mosaic = blender(patches, shape)
    if crop:
        logging.debug("Cropping...")
        valid = _valid(patches, shape)
        mosaic = crop_mosaic(mosaic, valid)

    return mosaic


def try_jit(*args, **kwargs):
    """Fall back to Python if Numba is not installed."""
    try:
        import numba
        return lambda f: numba.jit(f, *args, **kwargs)
    except ImportError:
        pass
    return lambda func: func


@try_jit(nopython=True, parallel=True, fastmath=True, cache=True)
def crop_mosaic(mosaic, valid):
    """Remove the black borders from the stitched image.

    Optimized in Numba; the first run will be slower.
    """
    height, width = valid.shape
    heights = np.zeros(width, dtype=np.int32)
    lefts = np.zeros(width, dtype=np.int32)
    rights = np.zeros(width, dtype=np.int32)

    area = 0
    for i in range(height):
        for j in range(width):
            heights[j] = (heights[j] + 1) if valid[i, j] else 0
        for j in range(width):
            lefts[j] = j
            while lefts[j] > 0 and heights[j] <= heights[lefts[j]-1]:
                lefts[j] = lefts[lefts[j] - 1]
        for j in range(width - 1, 0, -1):
            rights[j] = j
            while rights[j] < width - 1 and heights[j] <= heights[rights[j]+1]:
                rights[j] = rights[rights[j] + 1]
        for j in range(width):
            new_area = max(area, (rights[j] - lefts[j] + 1) * heights[j])
            if new_area > area:
                area = new_area
                ll, rr, hh, last = lefts[j], rights[j], heights[j], i

    return mosaic[last-hh+1:last+1, ll:rr+1, :]


def idx_to_keypoints(matches, kpts):
    """Replace keypoint indices with their coordinates."""
    def _i_to_k(match, kpt1, kpt2):
        return np.concatenate([kpt1[match[:, 0]], kpt2[match[:, 1]]],
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
    parser.add_argument('-p', '--path', type=str,
                        default="../data/ppwwyyxx/NSH",
                        help="directory with the images to process.")
    parser.add_argument("-s", "--shrink", type=float, default=2,
                        help="downsample the images by this amount.")
    parser.add_argument("--ba", default="incr",
                        choices=["none", "incr", "last"],
                        help="bundle adjustment type.")
    parser.add_argument("--equalize", "-e", action="store_true",
                        help="equalize image gain before stitching.")
    parser.add_argument("--crop", "-c", action="store_true",
                        help="remove the black borders.")
    parser.add_argument("-b", "--blend", default='multiband',
                        choices=list(BLENDERS.keys()),
                        help="blending algorithm.")
    parser.add_argument("-o", "--out", type=str,
                        help="save result to this file")
    args = parser.parse_args()

    exts = [".jpg", ".png", ".bmp"]
    exts += [ex.upper() for ex in exts]

    name = f"{os.path.basename(args.path)}_s{args.shrink}"
    files = [f for f in os.listdir(args.path)
             if any([f.endswith(ext) for ext in exts])]

    imgs = [cv2.imread(os.path.join(args.path, f)) for f in files]
    if args.shrink > 1:
        imgs = [cv2.resize(im, None, fx=1/args.shrink, fy=1/args.shrink)
                for im in imgs]

    try:  # cache
        arr = np.load(f"matches_{name}.npz", allow_pickle=True)
        kpts, matches = arr['kpts'], arr['matches']
    except IOError:
        kpts, matches = matching(imgs)
        np.savez(f"matches_{name}.npz", kpts=kpts, matches=matches)

    try:
        with open(f"ba_{name}.pkl", 'rb') as fid:
            regions = pickle.load(fid)
    except IOError:
        start = time.time()
        regions = traverse(imgs, idx_to_keypoints(matches, kpts),
                           badjust=args.ba)
        logging.info(f"Image registration, time: {time.time() - start}")
        with open(f"ba_{name}.pkl", 'wb') as fid:
            pickle.dump(regions, fid, protocol=pickle.HIGHEST_PROTOCOL)

    start = time.time()
    mosaic = stitch(regions, blender=BLENDERS[args.blend],
                    equalize=args.equalize, crop=args.crop)
    logging.info(f"Built mosaic, time: {time.time() - start}")

    if args.out:
        cv2.imwrite(args.out, mosaic)

    cv2.imshow("Mosaic", mosaic)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('numba').setLevel(logging.WARNING)  # silence Numba
    main()
