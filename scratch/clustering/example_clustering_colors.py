import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scratch.linear_algebra.vectors import Vector
from scratch.clustering.the_model import KMeans

# The VP of Swag has designed attractive DataSciencester stickers that he’d 
# like you to hand out at meetups.  Unfortunately, your sticker printer can 
# print at most five colors per sticker. And since the VP of Art is on sabbatical, 
# the VP of Swag asks if there’s some way you can take his design and modify it 
# so that it contains only five colors.

# Computer images can be represented as two-dimensional arrays of pixels, where 
# each pixel is itself a three-dimensional vector (red, green, blue) indicating 
# its color.
    # 1. Choosing five colors
    # 2. Assigning one of those colors to each pixel.

# It turns out this is a great task for k-means clustering, which can partition 
# the pixels into five clusters in red-green-blue space. If we then recolor the 
# pixels in each cluster to the mean color, we’re done.

iamge_path = "/workspaces/data-science-from-scratch/data-science-from-scratch/img/girl.jpg"
img = mpimg.imread(iamge_path) / 256 # rescale to between 0 and 1

# Behind the scenes img is a NumPy array, but for our purposes, we can treat it 
# as a list of lists of lists.

pixels = [pixel.tolist() for row in img for pixel in row]

# and then feed them to our clusterer:
clusterer = KMeans(5)
clusterer.train(pixels)

# Once it finishes, we just construct a new image with the same format:

def recolor(pixel: Vector) -> Vector:
    cluster = clusterer.classify(pixel)
    return clusterer.means[cluster]

new_img = [[recolor(pixel) for pixel in row]
           for row in img]

plt.imshow(img)
plt.imshow(new_img)
plt.axis("off")
plt.show()
