import cv2
import numpy as np

#######################################################################
#Esse módulo contém as funções responsáveis por simular as distorções #
#que a imagem sofre no processo de impressão seguida de digitalização.#
#                                                                     #
#######################################################################


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
def SimulatePrintScan(img, K):

    # Scaling and Haftoning image simulating print and scaning in real life
    img = cv2.resize(
        img, (K * img.shape[1], K * img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # Inserting error in the image
    img = ErrorDiffusion(img)

    # Removing haftone and going back to the original resolution (simulation case)
    Filtered = cv2.blur(img, (K, K))
    img = cv2.resize(Filtered, (int((1 / K) * Filtered.shape[1]), int(
        (1 / K) * Filtered.shape[0])), interpolation=cv2.INTER_LANCZOS4)

    return img
