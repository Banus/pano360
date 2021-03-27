"""Warps and blend images."""
import os
import numpy as np
import cv2


# from: https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b
def warp(img, kint, spherical=True):
    """Warp the image in cylindrical/spherical coordinates."""
    hh_, ww_ = img.shape[:2]
    y_i, x_i = np.indices((hh_, ww_))  # pixel coordinates

    # homogeneous coordinates
    xx_ = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(-1, 3)
    xx_ = np.linalg.inv(kint).dot(xx_.T).T  # normalize coordinate

    if spherical:
        # calculate equirectangular coords (sin(theta), sin(phi), cos(theta))
        x_n = np.stack([np.sin(xx_[:, 0]), np.sin(xx_[:, 1]),
                        np.cos(xx_[:, 0])*np.cos(xx_[:, 1])], axis=-1)
    else:
        # calculate cylindrical coords (sin(theta), h, cos(theta))
        x_n = np.stack([np.sin(xx_[:, 0]), xx_[:, 1], np.cos(xx_[:, 0])],
                       axis=-1)

    # project back to image-pixels and pixel coordinates
    x_pr = kint.dot(x_n.reshape(-1, 3).T).T
    x_pr = x_pr[:, :-1] / x_pr[:, [-1]]
    # make sure warp coords only within image bounds
    x_pr[(x_pr[:, 0] < 0) | (x_pr[:, 0] >= ww_) |
         (x_pr[:, 1] < 0) | (x_pr[:, 1] >= hh_)] = -1
    x_pr = x_pr.reshape(hh_, ww_, -1)

    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # warp the image to cylindrical coords and transparent background
    return cv2.remap(img_rgba, x_pr[:, :, 0].astype(np.float32),
                     x_pr[:, :, 1].astype(np.float32), cv2.INTER_AREA,
                     borderMode=cv2.BORDER_TRANSPARENT)


def main():
    """Script entry point."""
    base_path = "../data/ppwwyyxx/CMU2"
    img1 = cv2.imread(os.path.join(base_path, "medium01.JPG"))
    img2 = cv2.imread(os.path.join(base_path, "medium00.JPG"))

    height, width = img1.shape[:2]
    foc, delta = 3e3, 975
    # intrinsics
    intr = np.array([[foc, 0, width/2], [0, foc, height/2], [0, 0, 1]])

    img1, img2 = warp(img1, intr), warp(img2, intr)
    blend_mask = np.linspace(1, 0, delta).reshape((1, delta, 1))
    overlap = (img1[:, -delta:]*blend_mask + img2[:, :delta]*(1-blend_mask))
    blended = np.concatenate(
        [img1[:, :-delta], overlap.astype("uint8"), img2[:, delta:]], axis=1)

    cv2.imshow('blend', blended)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
