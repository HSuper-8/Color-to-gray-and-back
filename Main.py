import cv2
import glob
import pathlib
import numpy as np
import ColorEmbedding as ce
import ColorRecovery as cr
import Simulation as sm
import ColorSpaceTools as cst
import sys
import matplotlib.pyplot as plt

######################################################################################
#This module does the color codification and decodification for a set of             #
#images, offering the option to simulate an impression and a digitalization and the  #
#option of different resize values. Besides that, it is calculated the PSNR values   #
#between the original image and its decodificated version, which is useful to        #
#test the eficiency of the the color recovery.                                       #
#                                                                                    #
######################################################################################


# This is the main program of the Color to Gray and Back algorithm
def main():
    # Creating directories to save images

    pathlib.Path('./Images').mkdir(parents=True, exist_ok=True)

    # Textured Images with simulations of the real world
    pathlib.Path('./ImagesTextures').mkdir(parents=True, exist_ok=True)

    # Final Images after the process with simulations of the real world
    pathlib.Path('./ImagesResults').mkdir(parents=True, exist_ok=True)

    simulation = bool(int(
        input("\nEnter (1) or (0) for the option:\n(1) With Simulation\n(0) Without Simulation\n")))
    if(simulation):
        k = int(input("Enter a resize order\n"))

    for file in np.sort(glob.glob("Images/*.png")):
        print("Image %s..." % file[7:],)
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
            result = cst.saturation(result, 1.4)
        cv2.imwrite("ImagesResults/%s" % file[7:], result)

        # Reading restored images with simulations of the real world
        result = cv2.imread('ImagesResults/%s' % file[7:])
        original = cv2.imread("Images/%s" % file[7:])

        # Prints the original, textured, and resulting image
        if '-p' in sys.argv:
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
        print('PSNR: %lf' % cst.getPSNR(original, result))


if __name__ == '__main__':
    main()
