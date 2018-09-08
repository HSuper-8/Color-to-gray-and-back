import cv2
import numpy as np
import glob
import ColorEmbedding as ce
import ColorRecovery as cr
import Simulation as sm
import matplotlib.pyplot as plt

################################################################################
#This module defines the functions that are useful in the manipulation of      #
#the color spaces of colored images                                            #
################################################################################

# Function to process all images
def color2grayAndBack(k, simulation, argv):
    psnrs = []
    for file in np.sort(glob.glob("Images/*.png")):
        print("\nImage %s..." % file[7:],)
        image = cv2.imread('Images/%s' % file[7:])

        imageText = ce.incorporateTexture(image)
        if(simulation):
            print("Simulating Print and Scan...")
            imageText = sm.simulatePrintScan(imageText, k)
        cv2.imwrite("ImagesTextures/%s" % file[7:], imageText)

        # Trying to re-create original images
        imageText = cv2.imread('ImagesTextures/%s' % file[7:], 0)
        result = cr.recoverColor(imageText)
        #cv2.imwrite("ImagesResults/%s" % file[7:], result)
        #result = cv2.imread("ImagesResults/%s" % file[7:])
        if(simulation):
            result = saturation(result, 1.4)
        cv2.imwrite("ImagesResults/%s" % file[7:], result)

        # Reading restored images with simulations of the real world
        result = cv2.imread('ImagesResults/%s' % file[7:])
        original = cv2.imread("Images/%s" % file[7:])

        # Prints the original, textured, and resulting image
        if '-p' in argv:
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title("Imagem original")
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(imageText, cmap='gray')
            plt.title("Imagem Texturizada")
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title("Imagem Resultante")
            plt.axis('off')
            plt.show()

        # Calculating PSNR's values of the real results
        psnr = getPSNR(original, result)
        print('PSNR: %lf' % psnr)
        psnrs.append(psnr)
    return psnrs


# Function that separates an array into a positive and a negative array
def dividePlusMinus(image):

    # Separate the image in positive(plus) and negative(minus)
    height, width = image.shape

    plus = np.zeros((height, width), np.float32)
    minus = np.zeros((height, width), np.float32)

    # Separate the image in positive(plus) and negative(minus)
    minus = (image - np.abs(image)) / 2
    plus = (image + np.abs(image)) / 2

    return plus, minus


# Adjust image saturation
def saturation(image, fit):
    image = np.float32(image)
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)

    S *= fit
    S -= 255
    SPlus, SMinus = dividePlusMinus(S)
    S = SMinus + 255

    HSV = cv2.merge([H, S, V])
    image = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

    return image


# Transform an RGB image to YCrCb image
def BGR2YCrCb(image):
    Y = image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299
    Cr = 0.713 * image[:, :, 2] - 0.713 * Y[:, :]
    Cb = 0.564 * image[:, :, 0] - 0.564 * Y[:, :]

    return cv2.merge([Y, Cr, Cb])


# Transform an YCbCr image to BGR image
def YCrCb2BGR(image):
    R = ((image[:, :, 0]) + (1.402 * image[:, :, 1]))
    G = ((image[:, :, 0]) - (0.344 * image[:, :, 2]) - (0.714 * image[:, :, 1]))
    B = ((image[:, :, 0]) + (1.772 * image[:, :, 2]))

    return cv2.merge([B, G, R])


# Calculate PSNR's values
def getPSNR(true, pred):
    giantTrueMatrix = np.concatenate((true[0], true[1], true[2]), 1)
    giantPredMatrix = np.concatenate((pred[0], pred[1], pred[2]), 1)

    return cv2.PSNR(giantTrueMatrix, giantPredMatrix)
