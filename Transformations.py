import cv2
import numpy as np


# Transform an RGB image to YCbCr image
def RGB2YCBCR(img):
    imageY = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    imageCb = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    imageCr = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    imageY = np.float32(
        ((0.299 * img[:, :, 2]) + (0.587 * img[:, :, 1]) + (0.114 * img[:, :, 0])))
    imageCb = np.float32((0.564 * img[:, :, 0]) - (0.564 * imageY[:, :]))
    imageCr = np.float32((0.713 * img[:, :, 2]) - (0.713 * imageY[:, :]))
    img[:, :, 0] = imageY
    img[:, :, 1] = imageCr
    img[:, :, 2] = imageCb
    return img


# Transform an YCbCr image to RGB image
def YCbCr2BGR(img):
    imageR = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    imageG = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    imageB = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    imageR = np.float32(((img[:, :, 0]) + (1.402 * img[:, :, 1])))
    imageG = np.float32(
        ((img[:, :, 0]) - (0.344 * img[:, :, 2]) - (0.714 * img[:, :, 1])))
    imageB = np.float32(((img[:, :, 0]) + (1.772 * img[:, :, 2])))
    img[:, :, 0] = imageB
    img[:, :, 1] = imageG
    img[:, :, 2] = imageR
    return img


# Add error in the image
def ErrorDiffusion(image):
    for y in range(0, image.shape[1] - 1):
        for x in range(0, image.shape[0] - 1):
            oldpixel = image[x, y]
            if oldpixel > 127:
                image[x, y] = 255
            else:
                image[x, y] = 0
            quant_error = oldpixel - image[x, y]
            image[x + 1, y] = image[x + 1, y] + 7 / 16.0 * quant_error
            image[x - 1, y + 1] = image[x - 1, y + 1] + 3 / 16.0 * quant_error
            image[x, y + 1] = image[x, y + 1] + 5 / 16.0 * quant_error
            image[x + 1, y + 1] = image[x + 1, y + 1] + 1 / 16.0 * quant_error

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j] > 127:
                image[i, j] = 255
            else:
                image[i, j] = 0

    return np.uint8(image)


# Calculate PSNR's values
def getPSNR(true, pred):
    GiantTrueMatrix = np.concatenate((true[0], true[1], true[2]), 1)
    GiantPredMatrix = np.concatenate((pred[0], pred[1], pred[2]), 1)

    return cv2.PSNR(GiantTrueMatrix, GiantPredMatrix)
