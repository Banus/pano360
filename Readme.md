Panorama stitching - Computer Vision Project
============================================

In this repository we implement a panorama stitching application in Python.
The code is heavily based on [OpenPano](https://github.com/ppwwyyxx/OpenPano),
which in turn is an open source implementation of
[AutoStitch](http://matthewalunbrown.com/autostitch/autostitch.html).
Compared to OpenPano, we add an exposure correction stage and better handling
of the multi-band blending.

Installation
------------

The code relies on OpenCV, Numpy, Scipy and (optionally) Numba.
It is suggested to use `conda` to install OpenCV, as it automatically handles
the installation of binary libraries and dependencies.
We reccomend to use a separate environment because OpenCV install several
dependent packages, that may interfere with your existing installation:


```bash
conda create -n pano python=3.7
conda activate pano
conda install opencv numba
```

Numba is not strictly necessary to run the application, but it significantly
increases the speed on the crop stage.

Code organization
-----------------

The application entry point is `stitcher.py`, which implements the projection
and blending stages of panorama generation.

`features.py` contains the SIFT feature extraction, FLANN matching and
homography computation with RANSAC. We also include an implementation of
*[Multi-Scale Oriented Patches (MSOP)](http://matthewalunbrown.com/mops/mops.html#)*
that we didn't use because of the superior performance of SIFT.

`bundle_adj.py` includes the panorama discovery, focal and rotation estimation,
and the bundle adjustment stages of panorama generation.

The remaining modules are experiments or accessory functions.
`pano_tests.py` are the unit tests for different critical functions; run them
to ensure that the functionality is correct if you plan to modify the code.
`profiler.py` is a small self-contained Python profiler to guide the
optimization of the slow sections of the code.
`blend.py` is a set of experiments on Laplacian blending, Poisson blending and
and seam detection with graph cuts; the blending code actually used in panorama
generation has been moved to `stitcher`.

Usage
-----

Run the application by passing the image path and options, e.g.:

```bash
    python stitcher.py data/CMU2 -s 2
```

The path is the only mandatory argument. The following optional arguments are
available:

*  `-s`, `--shrink SHRINK`: downsample the images by a factor of *SHRINK* to
   speed up generation; stitching may fail for large factors because there
   aren't enough features to match.
*  `--ba {none,incr,last}`: how to apply bundle adjustment; skip it (*none*),
   apply it incrementally after adding an image (*incr*, default) or only after
   adding all the images (*last*)
*  `--equalize`, `-e`: compensate the exposure differences between images;
   disabled by default because it may lead to chromatic aberration.
*  `--crop`, `-c`: crops the largest rectangle from the panorama; disabled by
   default
*  `-b`, `--blend {none,linear,multiband}`: how to blend the images, either by
   simply pasting them on the mosaic (*none*), using linear blending (*linear*)
   or multi-band blending (*multiband*, default)
*  `-o OUT`, `--out OUT`: saves the mosaic to the *OUT* file, with the format
   determined by the extension (JPG, PNG supported).

The application stores the features and matches in a Numpy NPZ file and the
result of bundle adjustment (intrinsic and extrinsic camera parameters) in a
PKL file, tagged by the dataset name and shrink factor; the application checks
if the files exist and loads them instead of re-running the pipeline.
Delete the files if you want to re-generate the panorama for scratch.

Data
----

We tested the application on OpenPano's sequences CMU0, CMU1, CMU2, UAV and
NSH in the [example data](https://github.com/ppwwyyxx/OpenPano/releases/tag/0.1);
on the Lunch Room sequence in 
[PASSTA](https://www.cvl.isy.liu.se/en/research/datasets/passta/); on the
example data of c060604's
[panorama implementation](https://github.com/c060604/panorama) and a sequence
we acquired (*Athens*).
All the sequences are stored in a shared folder, accessible from
[Google Drive](https://drive.google.com/drive/folders/1thfXSy8RJHM6LOm-FcgdunMNh5tF1O2D?usp=sharing)
and
[One Drive](https://indiana-my.sharepoint.com/:f:/g/personal/eplebani_iu_edu/EpTXf8vGS4tJoMG3MaL5na0BatsUrDXAQE8RxWSGX4XUQw?e=abWBwa).
