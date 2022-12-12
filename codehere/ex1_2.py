import numpy as np
import cv2 as cv
# AMF params
max_patch_size = 15

img = cv.imread("../final1.bmp", 0)
kernel = np.ones((3,3),np.uint8)
dilation = cv.dilate(img,kernel,iterations = 1)

def _extract_patch(matrix, x, y, patch_size=3):
    height, width = matrix.shape
    size = patch_size // 2

    # initialize x index
    if x - size >= 0:
        x_begin = x - size
    else:
        x_begin = 0

    if x + size < height:
        x_end = x + size
    else:
        x_end = height - 1

    # initialize y index
    if y - size >= 0:
        y_begin = y - size
    else:
        y_begin = 0

    if y + size < width:
        y_end = y + size
    else:
        y_end = width - 1

    # loop inside patch
    output = []
    for i in range(x_begin, x_end+1):
        for j in range(y_begin, y_end+1):
            output.append(matrix[i][j])
    return output


# Adaptive median filter function
def amf(matrix):

    # prepare output
    output = np.copy(matrix)
    height, width = matrix.shape

    for x in range(height):
        for y in range(width):
            patch_size = 3
            patch = _extract_patch(matrix, x, y, patch_size)

            # extract min, max and median value of patch
            patch_min = np.min(patch)
            patch_max = np.max(patch)
            patch.sort()
            patch_median = patch[len(patch) // 2]

            # check if pixel is corrupted
            if patch_min < matrix[x][y] < patch_max:
                output[x][y] = matrix[x][y]
            else:
                # check if median value is also corrupted
                finish = False
                while not finish:
                    if 0 < patch_median < 255:
                        output[x][y] = patch_median
                        finish = True
                    else:
                    # calculate new patch
                        patch_size = patch_size + 2
                        if patch_size <= max_patch_size:
                            patch = _extract_patch(matrix, x, y, patch_size)
                            patch.sort()
                            patch_median = patch[len(patch) // 2]
                        else:
                            finish = True

    return output
result = amf(dilation)
erosion = cv.erode(result,kernel,iterations = 1)
closing = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel) #performance khong tot cho lam
cv.imshow('results',closing)
cv.waitKey(0)