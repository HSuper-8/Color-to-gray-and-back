import cv2
import numpy as np
import glob

# increases image saturation
def Saturation():
    i = 0
    for file in np.sort(glob.glob("./ImagesResults/*.png")):
        Image = cv2.imread(file, 3)
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(Image)
       
        for x in range(0, Image.shape[0]):
            for y in range(0, Image.shape[1]):
                S[x,y] = 1.2 * S[x,y]
                if S[x, y] > 255:
                    print(S[x, y])
                    S[x, y] = 255
        Image[:, :, 1] = S
        Image = cv2.cvtColor(Image, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./ImagesResults/%d.png" % i, Image)
        i += 1

# Transform an RGB image to YCbCr image
def BGR2YCrCb(img):
    img = np.float32(img)
    Y = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299
    Cr = 0.713 * img[:, :, 2] - 0.713 * Y[:, :]
    Cb = 0.564 * img[:, :, 0] - 0.564 * Y[:, :]
    img[:, :, 0] = Y
    img[:, :, 1] = Cr
    img[:, :, 2] = Cb

    return img


# Transform an YCbCr image to RGB image
def YCrCb2BGR(img):
    R = np.float32(((img[:, :, 0]) + (1.402 * img[:, :, 1])))
    G = np.float32(
        ((img[:, :, 0]) - (0.344 * img[:, :, 2]) - (0.714 * img[:, :, 1])))
    B = np.float32(((img[:, :, 0]) + (1.772 * img[:, :, 2])))
    img[:, :, 0] = B
    img[:, :, 1] = G
    img[:, :, 2] = R
    return img


# Calculate PSNR's values
def getPSNR(true, pred):
    GiantTrueMatrix = np.concatenate((true[0], true[1], true[2]), 1)
    GiantPredMatrix = np.concatenate((pred[0], pred[1], pred[2]), 1)

    return cv2.PSNR(GiantTrueMatrix, GiantPredMatrix)
