"""Warps and blend images."""
import os
import heapq

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


def alpha_blend(img1, img2, mask=None):
    """Blend using an alpha ramp."""
    if mask is None:
        delta = img1.shape[1]
        mask = np.linspace(1, 0, delta).reshape((1, delta, 1))
    return (img1*mask + img2*(1-mask)).astype("uint8")


def graph_cut(img1, img2, shrink=5):
    # pylint: disable=too-many-locals
    """Blend two images using graph cuts with optional downsampling."""
    dd_ = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    diff = np.max(np.abs(img1 - img2), axis=2)
    if img1.shape[2] == 4:  # borders are low priority
        diff[img1[:, :, 3] == 0], diff[img2[:, :, 3] == 0] = -1, -1
    if shrink > 1:
        # when subsampling, the weakest difference matters
        hh_, ww_ = diff.shape
        hh_, ww_ = hh_ // shrink, ww_ // shrink
        diff = diff[:shrink*hh_, :shrink*ww_]
        diff = np.min(diff.reshape(hh_, shrink, ww_, shrink), axis=(1, 3))

    mask = np.zeros(diff.shape, dtype=np.int32)
    rows, cols = mask.shape[:2]

    qq_, border = [], int(13/shrink) + 1
    mask[:, :border] = -1
    mask[:, -border+1:] = 1

    for yy_ in range(rows):
        qq_ += [(-1e3, -1, border, yy_), (-1e3, 1, cols-border, yy_)]
    heapq.heapify(qq_)

    while True:
        try:
            _, clr, xx_, yy_ = heapq.heappop(qq_)
        except IndexError:
            break

        if mask[yy_, xx_] != 0:
            continue
        mask[yy_, xx_] = clr

        for dx_, dy_ in dd_:
            nx_, ny_ = xx_ + dx_, yy_ + dy_
            if not(0 <= nx_ < cols and 0 <= ny_ < rows):
                continue
            if mask[ny_, nx_] == 0:
                heapq.heappush(qq_, (-diff[ny_, nx_], clr, nx_, ny_))

    mask = cv2.resize((mask == -1).astype("float32"), img1.shape[:2][::-1])
    return (mask[..., None]*255).astype("uint8")


# adapted from: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/
#   py_pyramids/py_pyramids.html
def laplacian_blending(img1, img2, mask=None, n_levels=6):
    """Use a Laplacian pyramid on the images for blending."""
    if mask is None:
        hh_, ww_, cc_ = img1.shape
        mask = np.linspace(1, -1, ww_).reshape((1, ww_, 1))
        mask = 1.0 / (1 + np.exp(-100 * mask))  # sigmoid
        mask = np.tile(mask, (hh_, 1, cc_))

    if mask.shape[2] == 1:
        mask = np.repeat(mask, img1.shape[2], axis=2)

    def _gassian_pyr(img):
        pyr = [img]
        for _ in range(n_levels):
            img = cv2.pyrDown(img)
            pyr.append(img)
        return pyr

    def _laplacian_pyr(img):
        pyr = _gassian_pyr(img)
        lap = [pyr[-1]]
        for idx in range(n_levels, 0, -1):
            im_ = pyr[idx-1]
            lap.append(im_ - cv2.pyrUp(pyr[idx])[:im_.shape[0], :im_.shape[1]])
        return lap

    pyr1 = _laplacian_pyr(img1.astype("float32"))
    pyr2 = _laplacian_pyr(img2.astype("float32"))
    pyrm = _gassian_pyr(mask)[::-1]

    pyrs = [la * gm + lb * (1.0 - gm) for la, lb, gm in zip(pyr1, pyr2, pyrm)]
    blended = pyrs[0]
    for ls_ in pyrs[1:]:
        blended = ls_ + cv2.pyrUp(blended)[:ls_.shape[0], :ls_.shape[1]]

    return np.clip(blended, 0, 255).astype("uint8")


def main():
    """Script entry point."""
    base_path = "../data/ppwwyyxx/CMU2"
    img1 = cv2.imread(os.path.join(base_path, "medium01.JPG"))
    img2 = cv2.imread(os.path.join(base_path, "medium00.JPG"))

    height, width = img1.shape[:2]
    foc, delta = 3e3, 976

    # intrinsics
    intr = np.array([[foc, 0, width/2], [0, foc, height/2], [0, 0, 1]])
    img1, img2 = warp(img1, intr), warp(img2, intr)

    mask = graph_cut(img1[:, -delta:], img2[:, :delta])
    overlap = laplacian_blending(
        img1[:, -delta:], img2[:, :delta], mask / 255.0)

    blended = np.concatenate(
        [img1[:, :-delta], overlap.astype("uint8"), img2[:, delta:]], axis=1)

    cv2.imshow('blend', blended)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
