import cv2
import pathlib
import numpy as np
import glob
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

        if(simulation):
            result = cst.saturation(result, 1.2)
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
        psnr = cst.getPSNR(original, result)
        print('PSNR: %lf' % psnr)
        psnrs.append(psnr)
    return psnrs


if __name__ == '__main__':
    # Creating directories to save images

    pathlib.Path('./Images').mkdir(parents=True, exist_ok=True)

    # Textured Images with simulations of the real world
    pathlib.Path('./ImagesTextures').mkdir(parents=True, exist_ok=True)

    # Final Images after the process with simulations of the real world
    pathlib.Path('./ImagesResults').mkdir(parents=True, exist_ok=True)

    k = 0

    simulation = bool(int(
        input("\nEnter (1) or (0) for the option:\n(1) With Simulation\n(0) Without Simulation\n")))
    if(('-c' not in sys.argv) & (simulation)):
        k = int(input("Enter a resize order\n"))

    if '-c' in sys.argv:
        eoq = color2grayAndBack(1, 0, sys.argv)
        psnrs = []
        for k in range(1, 9):
            print('\n-- Resize Order = ', k)
            psnrs.append(color2grayAndBack(k, simulation, sys.argv))
        psnrsArray = np.array(psnrs)
        #Display PSNR x Resize Order for each Image
        for k in range(0, len(psnrs[0])):

            #a = np.ones((15,))*eoq[k-1]
            plt.plot(range(1, 9), psnrsArray[:, k], 'bo')
            #plt.plot(range(1, 16), a)
            plt.title('Imagem %s' % (k + 1))
            plt.xlabel('PSNR')
            plt.ylabel('Resize Order')
            plt.show()
    else:
        color2grayAndBack(k, simulation, sys.argv)
