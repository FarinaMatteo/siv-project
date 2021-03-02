import random as rng
import colorsys
from skimage.morphology import closing
import numpy as np
from numba import jit


def randint_from_time(t):
    """
    Generates random seed given a timestamp
    """
    return rng.randint(int(t) % 10, t // 10)


def random_colors(n):
    """
    Generates :n: random colors encoded as an RGB tuple
    """
    colors = []
    for i in range(n):
        # generate random color in hsv space for higher visual distinction
        hue = rng.randint(0, 360)/360  # 360 degrees rotation
        saturation = rng.randint(50, 100)/ 100  # random saturation between 0.5 and 1.0
        colors.append(colorsys.hsv_to_rgb(h=hue, s=saturation, v=1.0))  # max brightness
    return colors
