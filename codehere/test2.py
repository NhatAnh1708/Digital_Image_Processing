from matplotlib import pyplot as plt
import cv2
import numpy as np

# Part 1
print("**\tImage segmentation using pixel color as feature\t**\n\n")

# Reading the image
img = cv2.imread("../final3.jpg")

# Changing color code from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



# Reshaping the image to get the pixel color values
color_location_vals = img.reshape((-1, 3))


# Pixel color match value with mean
def pixel_proxim(vec, mean):
    val = 0
    for i in range(len(vec)):
        val += abs(vec[i] - mean[i]) ** 2

    return val


means = []

# Number of segments(color clusters) to be created
k = 15

# Declaring means with random values
for i in range(k):
    means.append([int((i / k) * 255)] * 3)


# Function to provide updated means
def give_means(cluster_li, color_location_vals, m):
    count, totals, means = [0] * m, [[0] * 3] * m, [0] * m
    for i, j in enumerate(cluster_li):
        totals[j] = np.add(totals[j], color_location_vals[i])
        count[j] += 1

    for j in range(m):
        means[j] = np.divide(totals[j], count[j])

    means = np.array(means)
    means = means.astype(int)
    print(means)
    return means


# Function which provides index list to assign which cluster each pixel belongs to
def give_cluster_li(means, color_location_vals):
    cluster_li = []
    for i in color_location_vals:
        proxim_vals = []
        for j in means:
            proxim_vals.append(pixel_proxim(i, j))
        cluster_li.append(proxim_vals.index(min(proxim_vals)))

    return cluster_li


cluster_li = give_cluster_li(means, color_location_vals)

# Iterating for n iterations. This will give us the most dominant shades in RGB format in the given image as the means vector
n = 3
for i in range(n):
    means = give_means(cluster_li, color_location_vals, k)  # New means
    cluster_li = give_cluster_li(means, color_location_vals)  # New clusters

# Segmented image using k-means
seg_img = []
for i in cluster_li:
    seg_img.append(means[i])

# Re-shaping seg_img in the shape of the original image
seg_img = np.reshape(seg_img, img.shape)

# Plotting the segmented image
plt.imshow(seg_img)
plt.show()