import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 15]

# read given image and convert to RGB color-space
image = cv2.imread("/home/sarvesh/Projects/Image Processing/IITG.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# save a grayscale version of the image to be used for corner detection
img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
img = img.astype('float64')

def calc_corner_response(image, alpha):

    h = len(image)
    w = len(image[0])

    # initialize a container to save the corner responses for every pixel
    corner_responses = np.zeros((h, w))

    for x in range(2, h - 2):
        for y in range(2, w - 2):

            # square of the gradient in horizontal direction, vertical direction,
            # and product of the two gradients
            Ixx, Iyy, Ixy = (0, 0, 0)

            # only considering 5x5 window for calculating the corner response
            for i in range(5):
                for j in range(5):
                    n_x = x + 2 - i
                    n_y = y + 2 - j

                    Ixx += (img[x][n_y] - img[x][y])**2
                    Iyy += (img[n_x][y] - img[x][y])**2
                    Ixy += (img[x][n_y] - img[x][y]) * (
                        img[n_x][y] - img[x][y])

            det = Ixx * Iyy - Ixy**2
            trace = Ixx + Iyy

            r = det - alpha * (trace**2)

            corner_responses[x][y] = r

    return corner_responses

def non_maximal_suppression(responses, threshold):
    h = len(responses)
    w = len(responses[0])

    # initialize containers to store the coordinates of the corners
    cornerListx = []
    cornerListy = []

    for y in range(2, h - 1):
        for x in range(2, w - 1):
            # the first condition checks whether the pixel is the  
            # local maxima in a 5x5 window
            # the second condition checks if the corner response is above threshold
            if (responses[y][x] == np.amax(responses[y - 2:y + 3, x - 2:x + 3])
                    and responses[y][x] > threshold):
                cornerListx.append(x)
                cornerListy.append(y)

    return cornerListx, cornerListy

def harris_corner_detector(source, alpha, thres):

    corner_scores = calc_corner_response(source, alpha)
    cX, cY = non_maximal_suppression(corner_scores, thres)

    # while displaying the image the corners are marked with a red cross
    plt.imshow(image)
    plt.scatter(cX, cY, c='red', marker='x')
    plt.title("Corners marked with red crosses")
    plt.show

harris_corner_detector(img, 0.05, 100000000)