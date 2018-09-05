import cv2
import numpy as np

#######################################################################
#This module contains the implementation of functions responsible to  #
#simulate the distortions caused by the process of impression followed#
#by digitalization.                                                   #
#######################################################################


# Function that adds error difusion into the image
def errorDiffusion(image):
        image = np.float32(image)

        # Insert a border in the image
        imageExp = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
        imageExp[1:image.shape[0] + 1, 1:image.shape[1] + 1] = image[:, :]

        for y in range(1, image.shape[1] + 1):
                for x in range(1, image.shape[0] + 1):
                        oldPixel = imageExp[x, y]
                        if oldPixel > 127:
                            imageExp[x, y] = 255
                        else:
                            imageExp[x, y] = 0
                        quantError = oldPixel - imageExp[x, y]
                        imageExp[x + 1, y] = imageExp[x + 1, y] + 7 / 16.0 * quantError
                        imageExp[x - 1, y + 1] = imageExp[x - 1, y + 1] + 3 / 16.0 * quantError
                        imageExp[x, y + 1] = imageExp[x, y + 1] + 5 / 16.0 * quantError
                        imageExp[x + 1, y + 1] = imageExp[x + 1, y + 1] + 1 / 16.0 * quantError

        # Removes the border previously inserted
        image[:, :] = imageExp[1:image.shape[0] + 1, 1:image.shape[1] + 1]

        return image


# Function that simulates the process of a printer and a scanner
def simulatePrintScan(image, K):

    # Scaling and Haftoning image simulating print and scaning in real life
    image = cv2.resize(
        image, (K * image.shape[1], K * image.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # Inserting error in the image
    image = errorDiffusion(image)

    # Removing haftone and going back to the original resolution (simulation case)
    filtered = cv2.blur(image, (K, K))
    image = cv2.resize(filtered, (int((1 / K) * filtered.shape[1]), int(
        (1 / K) * filtered.shape[0])), interpolation=cv2.INTER_LANCZOS4)

    return image
