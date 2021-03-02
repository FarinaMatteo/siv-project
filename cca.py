import cv2
import numpy as np
import random as rng
import time
import argparse
from utils import randint_from_time, random_colors
from skimage.morphology import closing, disk
from skimage.measure import label


def build_argparser():
    parser = argparse.ArgumentParser(description='Code for Image Segmentation with Edge Detection '
                                                 'and Connected Component Analysis. ')
    parser.add_argument('--input', help='Path to input image.', default='sample_imgs/mr-6.jpg')
    parser.add_argument('--no-show',
                        help='Include this argument to avoid imshow (except the latest ones, '
                             'with the meaningful output. '
                             'See :arg: \'no-show-all\' for it.)',
                        required=False,
                        action='store_true',
                        default=None)
    parser.add_argument('--no-show-all',
                        help='Avoids every imshow when running.',
                        required=False,
                        action='store_true',
                        default=None)
    return parser


args = build_argparser().parse_args()
# random seed
rng.seed(randint_from_time(time.time_ns()))

# global variables definition
AREA_THRESH = 20  # no. pixels needed for an image section to be labeled
PERIMETER_THRESH = 40  # no. pixels needed for a contours to bound a region without being discarded
BLUE = (255, 0, 0)  # BGR code for 'blue' color
GREEN = (0, 255, 0)  # BGR code for 'green' color
RED = (0, 0, 255)  # BGR code for 'red' color
WHITE = (255, 255, 255)  # BGR code for 'white' color
GAUSS_KSIZE = (15, 15)  # kernel size for Gaussian Blur
ALPHA = 0.7  # default value for alpha blending

# Read image
img = cv2.imread(args.input)  # use this for img processing
src = img.copy()  # use this for final display
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Gaussian Blur to reduce variation a bit
img = cv2.GaussianBlur(img, GAUSS_KSIZE, sigmaX=2, sigmaY=2)  # std dev. in x,y direction is 2
if not args.no_show and not args.no_show_all:
    cv2.imshow("Gaussian Blur", img)
    cv2.waitKey(0)

# apply image thresholding (so that refined edge detection could be performed later on)
ret, threshed = cv2.threshold(img, 92, 255, cv2.THRESH_BINARY_INV)
if not args.no_show and not args.no_show_all:
    cv2.imshow("Thresh", threshed)
    cv2.waitKey(0)

# morphological closing on the thresholded image to compensate rocks sharpness
closed_img = closing(threshed, selem=disk(7))
if not args.no_show and not args.no_show_all:
    cv2.imshow("Morphology Closing", closed_img)
    cv2.waitKey(0)

# actually apply edge detection on the thresholded image
canny = cv2.Canny(closed_img, 0, 1)  # 0,1 params bc threshed is a binary image
if not args.no_show and not args.no_show_all:
    cv2.imshow("Canny Edge Detection on blurred image", canny)
    cv2.waitKey(0)

# after edge detection, apply contour extraction
cnts, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# get rid of designed 0 indexing in hierarchy (counterintuitive to me)
hierarchy_ = hierarchy[0]

# clean nested contours and generate mask for cca at the same time,
cnts_ = []
mask = np.zeros(src.shape[:2], dtype='uint8')
for cnt, h in zip(cnts, hierarchy_):
    if h[3] == -1:  # 3rd element in hierarchy object contains index of parent contours
        area = cv2.contourArea(cnt)
        per = cv2.arcLength(cnt, not h[2] < 0)
        if area > AREA_THRESH and per > PERIMETER_THRESH:
            cnts_.append(cnt)
            cv2.drawContours(mask, [cv2.approxPolyDP(cnt, 1, not h[2] < 0)], -1, 255, -1)

# display acceptable areas according to criteria
if not args.no_show and not args.no_show_all:
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)

# clean memory usage
del cnts
del hierarchy

# perform cca to draw colored segmented image
labels = label(mask, background=0)
colors = random_colors(len(labels)-1)  # discard bg
colored_mask = np.zeros_like(src)
# draw coloured mask by applying alpha blending
for i, label in enumerate(np.unique(labels)):
    if label == 0:
        continue
    for c in range(3):
        src[:, :, c] = np.where(labels == label,  # condition to be verified
                 src[:, :, c]*(1 - ALPHA) + ALPHA*colors[i][c]*255,  # expression(s) to be computed
                 src[:, :, c])  # slot(s) where value of expression(s) should be inserted

# display segmented image
if not args.no_show_all:
    cv2.imshow("Colored CCA", src)
    cv2.waitKey(0)

# display regions with boxes
for i in range(len(cnts_)):
    contours_poly = cv2.approxPolyDP(cnts_[i], 1, not hierarchy_[i][2] < 0)
    x, y, w, h = cv2.boundingRect(contours_poly)
    ctr = [x+w//2, y+h//2]
    cv2.drawContours(src, cnts_, i, RED, 2)
    cv2.rectangle(src, (x, y), (x+w, y+h), BLUE, 2, cv2.LINE_AA)

# show last output
if not args.no_show_all:
    cv2.imshow("CCA with Contours and Boxes", src)
    cv2.waitKey(0)

# clean graphical env. after script execution
cv2.destroyAllWindows()

