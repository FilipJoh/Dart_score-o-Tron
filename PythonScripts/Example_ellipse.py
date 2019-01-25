import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse, rescale
from skimage.draw import ellipse_perimeter
from scipy import ndimage as ndi

import pdb

import os
import cv2

from skimage import io

## Modified example from Scikit image.org about hough detector

# Load picture, convert to grayscale and detect edges
image_rgb = cv2.imread(os.path.abspath('../Images/board.jpg'))#data.coffee()[0:220, 160:420]

dimensions = image_rgb.shape
print("image size: ({},{})".format(dimensions[0], dimensions[1]))
# Scale down image to something more suitable
scale = 1.0
image_rgb = cv2.resize(image_rgb, None, fx=scale, fy=scale)

#fig1, axori = plt.subplots()
#axori.imshow(image_rgb)
#plt.show()

image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image_gray, 200, 500)
#edges = ndi.binary_fill_holes(edges)
#eroded_edges = ndi.binary_erosion(edges)
#edges = edges != eroded_edges
#fig3, axEdges = plt.subplots()
#axEdges.imshow(img_as_ubyte(edges))
#plt.show()

# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
"""result = hough_ellipse(edges, accuracy=20, threshold=100,
                       min_size=40, max_size=120)
result.sort(order='accumulator')

#pdb.set_trace()

# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)
"""
fig2, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(8, 4),
                                sharex=True, sharey=True)
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sortedConts = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

image_conts = image_gray.copy()
image_conts = cv2.cvtColor(image_conts, cv2.COLOR_GRAY2BGR)
image_ellipses = image_conts.copy()

nbr_conts = 5

cv2.drawContours(image_conts, sortedConts[:nbr_conts], -1, (0,255,0), 3)
ellipses =[]


for i in range(0,nbr_conts):
    ellipse = cv2.fitEllipse(sortedConts[i])
    cv2.ellipse(image_ellipses,ellipse, (255,0,255),5)
    ellipses.append(ellipse)

#print(len(contours))



ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('(un)Canny edges (?)')
ax2.imshow(img_as_ubyte(edges))

ax3.set_title('Edge (white) and result (red)')
ax3.imshow(image_conts)

ax4.set_title('Fitted ellipses 0:{}, starting from largest'.format(nbr_conts))
ax4.imshow(image_ellipses)

#plt.show()
