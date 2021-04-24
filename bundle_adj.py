"""Estimate the camera parameters with bundle adjustment."""
import argparse
import heapq
import logging
import os
from dataclasses import dataclass

import numpy as np
import cv2

from blend import intrinsics, SphProj


MAX_RESOLUTION = 1400

# bundle adjustment parameters
PARAMS_PER_CAMERA = 6
TERMS_PER_MATCH = 2
LM_LAMBDA = 5    # Levenberg–Marquardt regularization


@dataclass
class Image:
    """Patch with all the informations for stitching."""

    img: np.ndarray
    rot: np.ndarray
    intr: np.ndarray
    range: tuple = (np.zeros(2), np.zeros(2))

    def hom(self):
        """Homography to normalized coordinates."""
        return self.rot.dot(np.linalg.inv(self.intr))

    def inv_hom(self):
        """Return inverse camera transform."""
        return self.intr.dot(self.rot.T)


def _hom_to_from(cm1, cm2):
    """Homography between two cameras."""
    return (cm1.intr.dot(cm1.rot)).dot(cm2.rot.T.dot(np.linalg.inv(cm2.intr)))


def _focal(v1, v2, d1, d2):
    """Get focal from two squared estimates."""
    if v1 < v2:
        v1, v2 = v2, v1
    if v1 > 0 and v2 > 0:
        return np.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif v1 > 0:
        return np.sqrt(v1)
    return 0


def _get_focal(hom):
    """Run on the homography and its inverse to get a valid estimate."""
    hom = hom.ravel()

    d1, d2 = hom[6]*hom[7], (hom[7] - hom[6])*(hom[7] + hom[6])
    v1 = -(hom[0]*hom[1] + hom[3]*hom[4]) / d1
    v2 = (hom[0]*hom[0] + hom[3]*hom[3] - hom[1]*hom[1] - hom[4]*hom[4]) / d2
    f1 = _focal(v1, v2, d1, d2)

    d1 = hom[0]*hom[3] + hom[1]*hom[4]
    d2 = hom[0]*hom[0] + hom[1]*hom[1] - hom[3]*hom[3] - hom[4]*hom[4]
    v1, v2 = -hom[2]*hom[5] / d1, (hom[5]*hom[5] - hom[2]*hom[2]) / d2
    f0 = _focal(v1, v2, d1, d2)

    return np.sqrt(f0*f1)


def get_focal(hom):
    """Estimate the focal lenght from the homography [1].

    References
    ----------
    [1] Szeliski, Richard, and Heung-Yeung Shum. "Creating full view panoramic
    image mosaics and environment maps." Proceedings of the 24th annual
    conference on Computer graphics and interactive techniques. 1997.
    """
    f_ = _get_focal(hom)
    return f_ if f_ else _get_focal(np.linalg.inv(hom))


def _cross_mat(vec):
    """Skew symm. matrix for cross product."""
    return np.array(
        [[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])


def rotation_to_mat(rad=np.random.randn(3)):
    """Create a rotation matrices from the exponential representation."""
    ang = np.linalg.norm(rad)
    cross = _cross_mat(rad / ang if ang else rad)

    return np.eye(3) + cross*np.sin(ang) + (1-np.cos(ang))*cross.dot(cross)


def mat_to_angle(rot):
    """Exponential representation from rotation matrix."""
    rad = np.array(
        [rot[2, 1]-rot[1, 2], rot[0, 2]-rot[2, 0], rot[1, 0]-rot[0, 1]])
    mod = np.linalg.norm(rad)

    if mod < 1e-7:
        rad = 0
    else:
        theta = np.arccos(np.clip((np.trace(rot)-1) / 2, -1, 1))
        rad *= (theta / mod)
    return rad


def to_rotation(rot):
    """Find the closest rotation in the Frobenious norm."""
    uu_, _, vv_ = np.linalg.svd(rot)
    rot = uu_.dot(vv_)
    if np.linalg.det(rot) < 0:
        rot *= -1   # no reflections
    return rot


def straighten(rots):
    """Global rotation to have the x axis on the same plane."""
    cov = np.cov(np.stack([rot[0] for rot in rots], axis=-1))
    _, _, vv_ = np.linalg.svd(cov)
    v_y = vv_[2]
    v_z = np.sum(np.stack([rot[2] for rot in rots], axis=0), axis=0)
    v_x = np.cross(v_y, v_z)
    v_x /= np.linalg.norm(v_x)
    v_z = np.cross(v_x, v_y)

    rot_g = np.stack([v_x, v_y, v_z], axis=-1)
    return [rot.dot(rot_g) for rot in rots]


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


#
# Bundle adjustment
#

def traverse(imgs, matches):
    """Traverse connected matches by visiting the best matches first."""
    # find starting point
    idx, homs, scores = zip(*[(i, *matches[i][j][1:3]) for i in matches.keys()
                              for j in matches[i].keys()])
    src = idx[np.argmax(scores)]
    intr = intrinsics(np.median([get_focal(hom) for hom in homs]))

    iba = IncrementalBundleAdjuster(len(imgs))
    iba.cameras[src] = Image(imgs[src], np.eye(3), intr)

    qq_ = [(-matches[src][j][2], src, j) for j in matches[src].keys()]
    heapq.heapify(qq_)

    while True:
        try:
            score, src, dst = heapq.heappop(qq_)
        except IndexError:
            break
        if iba.cameras[dst] is not None:  # already estimated
            continue

        hom = matches[dst][src][1]
        rot = to_rotation(np.linalg.inv(intr).dot(hom.dot(intr)))
        rot = iba.cameras[src].rot.dot(rot)

        iba.cameras[dst] = Image(imgs[dst], rot, intr)
        iba.add(src, dst, matches[dst][src][0])

        for new in matches[dst].keys():
            heapq.heappush(qq_, (-matches[dst][new][2], dst, new))

    return iba.cameras


def residuals(cameras, matches):
    """Find estimation errors."""
    res = []
    for i, j, match in matches:
        hom = _hom_to_from(cameras[i], cameras[j])
        trans = hom.dot(match[:, 3:6].T)
        res.append((trans / trans[[-1], :] - match[:, :3].T)[:-1])
    return np.concatenate(res, axis=1).ravel()


def loss(res):
    """Error function: Residual Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean(np.square(res)))


def dr_dvi(rot):
    """Rotation derivative w.r.t. the exponential representation."""
    rad = mat_to_angle(rot)
    vsqr = np.sum(np.square(rad))
    if vsqr < 1e-14:
        return np.stack([_cross_mat([1, 0, 0]), _cross_mat([0, 1, 0]),
                         _cross_mat([0, 0, 1])])

    ire = np.eye(3) - rot
    res = np.stack([_cross_mat(rad)*r for r in rad])
    res[0] += _cross_mat(np.cross(rad, ire[:, 0]))
    res[1] += _cross_mat(np.cross(rad, ire[:, 1]))
    res[2] += _cross_mat(np.cross(rad, ire[:, 2]))

    return res.dot(rot) / vsqr


# derivatives of the intrinsic matrix w.r.t. its parameters
_DKDFOCAL = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
_DKDPPX = np.float32([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
_DKDPPY = np.float32([[0, 0, 0], [0, 0, 1], [0, 0, 0]])


def _jacobian_symbolic(cameras, matches):
    """Compute the symbolic Jacobian for the bundler."""
    m_offs = np.cumsum([0] + [len(m) for _, _, m in matches])
    n_match = m_offs[-1]

    cam_idx = [i for i, c in enumerate(cameras) if c is not None]
    np_cam = PARAMS_PER_CAMERA * len(cam_idx)

    jac = np.zeros((TERMS_PER_MATCH * n_match, np_cam))
    jac_t_jac = np.zeros((np_cam, np_cam))

    # cache rotation derivatives
    drs = [dr_dvi(cameras[i].rot) for i in cam_idx]
    for idx, (i, j, match) in enumerate(matches):
        m_slice = slice(m_offs[idx]*TERMS_PER_MATCH,
                        m_offs[idx+1]*TERMS_PER_MATCH)

        hom = _hom_to_from(cameras[i], cameras[j])
        from_R, to_R = cameras[i].rot, cameras[j].rot
        from_K, to_K = cameras[i].intr, cameras[j].intr

        pts = hom.dot(match[:, 3:6].T)
        inv_z = 1 / pts[2]
        dpdh = (pts[0]*inv_z*inv_z, pts[1]*inv_z*inv_z, inv_z)

        def drdv(xx_):
            """Differentiate different values w.r.t. the residuals."""
            return np.concatenate([-xx_[0]*dpdh[2] + xx_[2]*dpdh[0],
                                   -xx_[1]*dpdh[2] + xx_[2]*dpdh[1]])

        # Jacobian
        # first camera
        u2_ = from_R.dot(to_R.T.dot(np.linalg.inv(to_K))).dot(
            match[:, 3:6].T)

        c_off_i = cam_idx.index(i)*PARAMS_PER_CAMERA
        jac[m_slice, c_off_i] = drdv(_DKDFOCAL.dot(u2_))
        jac[m_slice, c_off_i + 1] = drdv(_DKDPPX.dot(u2_))
        jac[m_slice, c_off_i + 2] = drdv(_DKDPPY.dot(u2_))
        # rotation
        drdvi = drs[cam_idx.index(i)]
        jac[m_slice, c_off_i + 3] = drdv(from_K.dot(drdvi[0].dot(u2_)))
        jac[m_slice, c_off_i + 4] = drdv(from_K.dot(drdvi[1].dot(u2_)))
        jac[m_slice, c_off_i + 5] = drdv(from_K.dot(drdvi[2].dot(u2_)))

        # # second camera
        u2_ = -np.linalg.inv(to_K).dot(match[:, 3:6].T)

        c_off_j = cam_idx.index(j)*PARAMS_PER_CAMERA
        jac[m_slice, c_off_j] = drdv(hom.dot(_DKDFOCAL).dot(u2_))
        jac[m_slice, c_off_j + 1] = drdv(hom.dot(_DKDPPX).dot(u2_))
        jac[m_slice, c_off_j + 2] = drdv(hom.dot(_DKDPPY).dot(u2_))
        # rotation
        drdvi_t = drs[cam_idx.index(j)].transpose(0, 2, 1)
        jac[m_slice, c_off_j + 3] = drdv(hom.dot(drdvi_t[0].dot(u2_)))
        jac[m_slice, c_off_j + 4] = drdv(hom.dot(drdvi_t[1].dot(u2_)))
        jac[m_slice, c_off_j + 5] = drdv(hom.dot(drdvi_t[2].dot(u2_)))

        # J^T J
        i_slice = slice(c_off_i, c_off_i+PARAMS_PER_CAMERA)
        j_slice = slice(c_off_j, c_off_j+PARAMS_PER_CAMERA)

        jac_t_jac[i_slice, i_slice] = \
            jac[m_slice, i_slice].T.dot(jac[m_slice, i_slice])
        jac_t_jac[j_slice, j_slice] = \
            jac[m_slice, j_slice].T.dot(jac[m_slice, j_slice])

        cross_block = jac[m_slice, i_slice].T.dot(jac[m_slice, j_slice])
        jac_t_jac[i_slice, j_slice] = cross_block
        jac_t_jac[j_slice, i_slice] = cross_block.T

    return jac, jac_t_jac


class IncrementalBundleAdjuster:
    """Bundle adjustment one image at a time."""

    def __init__(self, n_cameras, incremental=True):
        """Set bundler parameters."""
        self.cameras = [None] * n_cameras
        self.matches = []
        self.is_incremental = incremental

    def add(self, src, dst, match):
        """Add a new image to the bundler."""
        self.matches.append((src, dst, match))

        if self.is_incremental and len(self.matches) <= 1:
            self.optimize()

    def optimize(self):
        """Refine the camera parameters."""
        errs = residuals(self.cameras, self.matches)
        logging.debug(f"Initial error: {loss(errs)}")

        jac, jac_t_jac = _jacobian_symbolic(self.cameras, self.matches)


#
# Reprojection
#

def _proj_img_range_border(rot, shape, kint):
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

    pts = SphProj.hom2proj(rot.dot(np.linalg.inv(kint).dot(borders.T)).T)
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
        reg.range = _proj_img_range_corners(reg.img.shape[:2], reg.hom())

    resolution, im_range = estimate_resolution(regions)
    target = (im_range[1] - im_range[0]) / resolution

    shape = tuple(int(t) for t in np.round(target))[::-1]  # y,x order
    mosaic = np.zeros(shape + (3,), dtype=np.uint8)        # RGBA image
    for reg in regions:
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
        xx_ = reg.inv_hom().dot(xx_.T).T.astype(np.float32)
        x_pr = xx_[:, :-1] / xx_[:, [-1]] + np.float32([ww_/2, hh_/2])
        x_pr = x_pr.reshape(top[1]-bottom[1], top[0]-bottom[0], -1)
        mask = (x_pr[..., 0] < 0) | (x_pr[..., 0] >= ww_) | \
               (x_pr[..., 1] < 0) | (x_pr[..., 1] >= hh_)
        x_pr[mask] = -1

        # paste only valid pixels
        warped = cv2.remap(reg.img, x_pr[:, :, 0], x_pr[:, :, 1],
                           cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        tile = mosaic[bottom[1]:top[1], bottom[0]:top[0]]
        mosaic[bottom[1]:top[1], bottom[0]:top[0]] = np.where(
            mask[..., None], tile, warped)

    return mosaic


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

    arr = np.load(f"matches_{name}.npz", allow_pickle=True)
    kpts, matches = arr['kpts'], arr['matches']

    regions = traverse(imgs, idx_to_keypoints(matches, kpts))
    # regions = bundle_adjustment(regions, kpts, matches)

    mosaic = stitch(regions)
    cv2.imshow("Mosaic", mosaic)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
