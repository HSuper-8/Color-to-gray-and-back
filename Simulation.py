import cv2
import numpy as np


# Function that adds error difusion in the image
def ErrorDiffusion(img):
        img = np.float32(img)

        # Insert a border in the image
        img_exp = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
        img_exp[1:img.shape[0] + 1, 1:img.shape[1] + 1] = img[:, :]

        for y in range(1, img.shape[1] + 1):
                for x in range(1, img.shape[0] + 1):
                        old_pixel = img_exp[x, y]
                        if old_pixel > 127:
                            img_exp[x, y] = 255
                        else:
                            img_exp[x, y] = 0
                        quant_error = old_pixel - img_exp[x, y]
                        img_exp[x + 1, y] = img_exp[x + 1, y] + 7 / 16.0 * quant_error
                        img_exp[x - 1, y + 1] = img_exp[x - 1, y + 1] + 3 / 16.0 * quant_error
                        img_exp[x, y + 1] = img_exp[x, y + 1] + 5 / 16.0 * quant_error
                        img_exp[x + 1, y + 1] = img_exp[x + 1, y + 1] + 1 / 16.0 * quant_error

        # Removes the border previously inserted
        img[:, :] = img_exp[1:img.shape[0] + 1, 1:img.shape[1] + 1]

        return img


# Function that simulates the process of a printer and a scanner
def SimulateRealWorld(image, K):

    # Scaling and Haftoning image simulating print and scaning in real life
    image = cv2.resize(
        image, (K * image.shape[1], K * image.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # Inserting error in the image
    image = ErrorDiffusion(image)

    # Removing haftone and going back to the original resolution (simulation case)
    Filtered = cv2.blur(image, (K, K))
    image = cv2.resize(Filtered, (int((1 / K) * Filtered.shape[1]), int(
        (1 / K) * Filtered.shape[0])), interpolation=cv2.INTER_LANCZOS4)

    return image
