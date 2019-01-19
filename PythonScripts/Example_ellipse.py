import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse, rescale
from skimage.draw import ellipse_perimeter
from scipy import ndimage as ndi

import pdb

import os

from skimage import io

## Modified example from Scikit image.org about hough detector

# Load picture, convert to grayscale and detect edges
image_rgb = io.imread(os.path.abspath('../Images/board.jpg'))#data.coffee()[0:220, 160:420]

dimensions = image_rgb.shape
print("image size: ({},{})".format(dimensions[0], dimensions[1]))
# Scale down image to something more suitable
image_rgb = rescale(image_rgb, 1.0 / 4.0, False)

fig1, axori = plt.subplots()
axori.imshow(image_rgb)
plt.show()

image_gray = color.rgb2gray(image_rgb)
edges = canny(image_gray, sigma=2.0,
              low_threshold=0.55, high_threshold=0.8)
#edges = ndi.binary_fill_holes(edges)
#eroded_edges = ndi.binary_erosion(edges)
#edges = edges != eroded_edges
fig3, axEdges = plt.subplots()
axEdges.imshow(img_as_ubyte(edges))
plt.show()

# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
result = hough_ellipse(edges, accuracy=20, threshold=80,
                       min_size=40, max_size=120)
result.sort(order='accumulator')

pdb.set_trace()

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

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                sharex=True, sharey=True)

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()
