"""Estimate the camera parameters with bundle adjustment."""
import os

import numpy as np
import cv2

from blend import intrinsics, warp


def _focal(v1, v2, d1, d2):
    """Get focal from two squared estimates."""
    if v1 < v2:
        v1, v2 = v2, v1
    if v1 > 0 and v2 > 0:
        return np.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif v1 > 0:
        return np.sqrt(v1)
    return 0


def get_focal(hom):
    """Estimate the focal lenght from the homography [1].

    References
    ----------
    [1] Szeliski, Richard, and Heung-Yeung Shum. "Creating full view panoramic
    image mosaics and environment maps." Proceedings of the 24th annual
    conference on Computer graphics and interactive techniques. 1997.
    """
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


def rotation_to_mat(rot=np.random.randn(3)):
    """Create a rotation matrices from the exponential representation."""
    ang = np.linalg.norm(rot)
    rot /= ang
    cross = np.array(
        [[0, -rot[2], rot[1]], [rot[2], 0, -rot[0]], [-rot[1], rot[0], 0]])

    return np.eye(3) + cross*np.sin(ang) + (1-np.cos(ang))*cross.dot(cross)


def to_rotation(rot):
    """Find the closest rotation in the Frobenious norm."""
    uu_, _, vv_ = np.linalg.svd(rot)
    rot = uu_.dot(vv_)
    if np.linalg.det(rot) < 0:
        rot *= -1   # no reflections
    return rot


def absolute_rotations(homs, kint):
    """Find camera rotation w.r.t. a reference given homographies."""
    rots = [to_rotation(np.linalg.inv(kint).dot(hom.dot(kint)))
            for hom in homs]

    # mid point as reference to reduce drift
    mid = len(rots) // 2
    rot_r = [rots[mid]]
    for rot in rots[mid+1:]:
        rot_r.append(rot.dot(rot_r[-1]))
    rot_l = [np.eye(3)]
    for rot in rots[:mid]:
        rot_l.append(rot.T.dot(rot_l[-1]))
    return rot_l[::-1] + rot_r


def main():
    """Script entry point."""
    base_path = "../data/ppwwyyxx/CMU2"

    imgs = [cv2.imread(os.path.join(base_path, f"medium{i:02d}.JPG"))
            for i in range(3)]
    imgs = [cv2.resize(im, None, fx=0.5, fy=0.5) for im in imgs]

    arr = np.load("matches2.npz", allow_pickle=True)
    _, _, homs = arr['kpts'], arr['matches'], arr['homs']

    focals = [get_focal(np.linalg.inv(hom)) for hom in homs]
    foc = np.median([f for f in focals])
    height, width = imgs[1].shape[:2]
    # intrinsics
    intr = np.array([[foc, 0, width/2], [0, foc, height/2], [0, 0, 1]])

    height, width = imgs[0].shape[:2]
    tr_ = np.array([[1, 0, width/2], [0, 1, height/2], [0, 0, 1]])

    print(absolute_rotations(homs, intrinsics(foc)))
    hom1 = tr_.dot(homs[0].dot(np.linalg.inv(tr_)))
    hom2 = tr_.dot(homs[1].dot(np.linalg.inv(tr_)))

    proj1 = warp(imgs[0], intr, np.linalg.inv(hom1))[..., :3]
    proj2 = warp(imgs[2], intr, hom2)[..., :3]
    cv2.imshow('warp', np.uint8(proj1/3+proj2/3+imgs[1]/3))

    # im1 = cv2.warpPerspective(imgs[0], hom1,
    #                           (imgs[1].shape[1], imgs[1].shape[0]))
    # im2 = cv2.warpPerspective(imgs[2], np.linalg.inv(hom2),
    #                           (imgs[1].shape[1], imgs[1].shape[0]))
    # cv2.imshow('warp', np.uint8(im1/3+im2/3+imgs[1]/3))

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
