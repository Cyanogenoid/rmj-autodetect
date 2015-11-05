import sys

import numpy as np
from scipy.misc import imread, imsave
from scipy.ndimage.filters import sobel
from skimage.feature import canny


class Hough(object):
    def __init__(self, bins, template):
        self.bins = bins
        self.rtable = self.make_rtable(template)

    def make_rtable(self, edges):
        table = [[] for _ in range(self.bins)]
        r_y, r_x = self._centre_point(edges)
        gradients = self._gradient(edges)

        for y, x in np.transpose(edges.nonzero()):
            dx, dy = (x - r_x), (y - r_y)
            r = np.linalg.norm((dx, dy))
            if dx:
                alpha = np.arctan(dy / dx)
            else:
                alpha = 0
            phi = int((gradients[y, x] + np.pi) * self.bins / (2*np.pi))
            if phi == len(table):
                phi = 0
            table[phi].append((r, alpha))
        return table

    def detect(self, edges):
        gradients = self._gradient(edges)
        accumulator = np.zeros(edges.shape)
        for y, x in np.transpose(edges.nonzero()):
            phi = int((gradients[y, x] + np.pi) * self.bins / (2*np.pi))
            for r, alpha in self.rtable[phi]:
                x_c =  x + int(r * np.cos(alpha + np.pi))
                y_c =  y + int(r * np.sin(alpha + np.pi))
                try:
                    accumulator[y_c, x_c] += 1
                except:
                    pass
        return accumulator

    def _gradient(self, image):
        dx = sobel(image, axis=0, mode='constant')
        dy = sobel(image, axis=1, mode='constant')
        return np.arctan2(dy, dx)

    def _centre_point(self, image):
        x_c, y_c = 0, 0
        n = 0
        for y, x in np.transpose(image.nonzero()):
            x_c += x
            y_c += y
            n += 1
        return (int(x_c / n), int(y_c / n))


canny_low = None
canny_high = 25

template = imread('images/template.png', flatten=True)
template_edges = canny(template)
#  template_edges = canny(template, sigma=0.5,
#                   low_threshold=canny_low, high_threshold=canny_high)
image = imread('images/a.png', flatten=True)
image_edges = canny(image)
h = Hough(40, template_edges)
acc = h.detect(image_edges)
imsave('test.png', acc)
