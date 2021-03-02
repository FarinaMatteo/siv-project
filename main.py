from __future__ import print_function
import cv2
import numpy as np
import argparse
import random as rng
import time
import colorsys
from utils import randint_from_time, random_colors


def build_argparser():
    parser = argparse.ArgumentParser(description='Code for Image Segmentation with Distance '
                                                 'Transform and Watershed Algorithm. Sample code '
                                                 'showing how to segment overlapping objects in '
                                                 'addition to Watershed and Distance '
                                                 'Transformation.')
    parser.add_argument('--input', help='Path to input image.', default='sample_imgs/mr-6.jpg')
    parser.add_argument('--no-show',
                        help='Include this argument to avoid imshow (except the last one, '
                             'see :arg: \'no-show-all\' for it.',
                        required=False,
                        action='store_true',
                        default=None)
    parser.add_argument('--no-show-all',
                        help='Avoids every imshow when running.',
                        required=False,
                        action='store_true',
                        default=None)
    return parser


rng.seed(randint_from_time(time.time_ns()))
args = build_argparser().parse_args()
src = cv2.imread(cv2.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(1)

# Show source image
if not args.no_show and not args.no_show_all:
    cv2.imshow('Source Image', src)
    cv2.waitKey(0)
src[np.all(src == 255, axis=2)] = 0

# Show output image
if not args.no_show and not args.no_show_all:
    cv2.imshow('Black Background Image', src)
    cv2.waitKey(0)
kernel = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]], dtype=np.float32)

# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
laplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
if not args.no_show and not args.no_show_all:
    cv2.imshow("Laplacian", laplacian)
    cv2.waitKey(0)

sharp = np.float32(src)
if not args.no_show and not args.no_show_all:
    cv2.imshow("Sharp", sharp)
    cv2.waitKey(0)

imgResult = sharp - laplacian
if not args.no_show and not args.no_show_all:
    cv2.imshow("imgResult", imgResult)
    cv2.waitKey(0)

# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
if not args.no_show and not args.no_show_all:
    cv2.imshow('New Sharped Image', imgResult)
    cv2.waitKey(0)

laplacian = np.clip(laplacian, 0, 255)
laplacian = np.uint8(laplacian)
if not args.no_show and not args.no_show_all:
    cv2.imshow('Laplace Filtered Image', laplacian)
    cv2.waitKey(0)

bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
if not args.no_show and not args.no_show_all:
    cv2.imshow('Binary Image', bw)
    cv2.waitKey(0)

dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
if not args.no_show and not args.no_show_all:
    cv2.imshow('Distance Transform Image', dist)
    cv2.waitKey(0)

# threshold the distance transform by setting at white every pixel greater than 0.4 distance
_, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)

# Dilate a bit the dist image (note that dilation is the dual of erosion in morph. operations).
# This allows to further enhance bright areas.
kernel1 = np.ones((3,3), dtype=np.uint8)
dist = cv2.dilate(dist, kernel1)
if not args.no_show and not args.no_show_all:
    cv2.imshow('Peaks', dist)
    cv2.waitKey(0)

# re-convert distance transform to uint8 so that contours can be found by cv22 fn
dist_8u = dist.astype('uint8')
# ... and actually find them
contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create the marker image for the watershed algorithm (white filled mask)
markers = np.zeros(dist.shape, dtype=np.int32)
# Draw the foreground markers
for i in range(len(contours)):
    cv2.drawContours(markers, contours, i, (i+1), -1) # '-1' param -> fills the contour
if not args.no_show and not args.no_show_all:
    cv2.imshow("Markers mask", markers.astype('uint8'))
    cv2.waitKey(0)

# Draw the background marker
cv2.circle(markers, (5,5), 3, (255,255,255), -1)
if not args.no_show and not args.no_show_all:
    cv2.imshow("Circle on Markers mask", markers.astype('uint8'))
    cv2.waitKey(0)

# apply watershed algorithm on the sharpened image with the given mask
cv2.watershed(imgResult, markers)
if not args.no_show and not args.no_show_all:
    cv2.imshow("Watershed", imgResult)
    cv2.waitKey(0)

# change color of the markers mask
mark = markers.astype('uint8')
mark = cv2.bitwise_not(mark)
if not args.no_show and not args.no_show_all:
    cv2.imshow("Markers Inverted", mark)
    cv2.waitKey(0)

# generate random colors for final img display
colors = random_colors(len(contours))
# Create the output image
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
# Fill labeled objects with random colors
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i,j]
        if index > 0 and index <= len(contours): # '0' is the background label
            dst[i,j,:] = colors[index-1]

# Visualize the final image
if not args.no_show_all:
    cv2.imshow('Final Result', dst)
    cv2.waitKey(0)

# clean graphical env. after script execution
cv2.destroyAllWindows()
