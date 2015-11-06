import os.path

import cv2
import numpy as np
from matplotlib import pyplot as plt


def otsu_threshold(image):
    # http://www.academypublisher.com/proc/isip09/papers/isip09p109.pdf
    # also see: http://stackoverflow.com/a/16047590
    flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    threshold, _ = cv2.threshold(image, 0, 255, flags)
    return threshold


def edge_detect(image, size=5, threshold=None):
    # blur image to get rid of noise
    # TODO: is this actually needed? screenshots by themselves have no noise
    blur = cv2.GaussianBlur(image, (size, size), 0)

    if not threshold:
        higher, lower = 7, 10

    # use Canny's algorithm for edge detection
    edges = cv2.Canny(blur, higher, lower)
    return edges


def load_image(*path):
    image_path = os.path.join('images', *path)
    return cv2.imread(image_path)


# load images
img = load_image('a.png')
tpl = load_image('template.png')

# template needs border around it for edge detection to work properly
tpl = cv2.copyMakeBorder(tpl, 2, 0, 2, 2, cv2.BORDER_CONSTANT, value=[255] * 3)
tpl = cv2.copyMakeBorder(tpl, 0, 2, 0, 0, cv2.BORDER_CONSTANT, value=[0] * 3)

# make image and template gray
tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create edges
img_edges = edge_detect(img_gray)
tpl_edges = edge_detect(tpl_gray)

w, h = tpl_gray.shape[::-1]
res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCORR_NORMED)
threshold = np.max(res) - 0.05
print('Accumulator threshold: {}'.format(threshold))
loc = np.where(res >= threshold)
i = 0
for pt in zip(*loc[::-1]):
    i += 1
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255, 127), 2)
print('{} spikes found'.format(i))

plt.subplot(221), plt.imshow(img_edges, cmap='gray')
plt.title('Image Edges'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(tpl_edges, cmap='gray')
plt.title('Template Edges'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(res, cmap='gray')
plt.title('Accumulator'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Detected positions'), plt.xticks([]), plt.yticks([])

plt.show()
