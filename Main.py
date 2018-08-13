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
#Nesse módulo é realizada a codificação e decodificação das cores para um conjunto   #
#de imagens, oferecendo as opções de simular impressão/digitalização e escolher      #
#diferentes valores de resize. Além disso, é calculado o valor do PSNR entre a imagem#
#original e a decodificada, permitindo testar a capacidade de recuperação de   cores #
#do algoritmo.                                                                       #
#                                                                                    #
######################################################################################


# This is the main program of the Color to Gray and Back algorithm
def main():
    # Creating directories to save images

    # Textured Images with simulations of the real world
    pathlib.Path(
        './ImagesTextures').mkdir(parents=True, exist_ok=True)

    # Final Images after the process with simulations of the real world
    pathlib.Path('./ImagesResults').mkdir(parents=True, exist_ok=True)

    simulation = bool(int(
        input("\nEnter (1) or (0) for the option:\n(1) With Simulation\n(0) No Simulation\n")))
    if(simulation):
        k = int(input("Enter a resize order\n"))

    for file in np.sort(glob.glob("Images/*.png")):
        print("Image %s..." % file[7:],)
        Image = cv2.imread('Images/%s' % file[7:])

        ImageText = ce.IncorporateTexture(Image)
        if(simulation):
            print("Simulating Print and Scan...")
            ImageText = sm.SimulatePrintScan(ImageText, k)
        cv2.imwrite("ImagesTextures/%s" % file[7:], ImageText)

        # Trying to re-create original images
        ImageText = cv2.imread('ImagesTextures/%s' % file[7:], 0)
        Result = cr.RecoverColor(ImageText)
        if(simulation):
            Result = cst.Saturation(Result)
        cv2.imwrite("ImagesResults/%s" % file[7:], Result)

        # Reading restored images with simulations of the real world
        Result = cv2.imread('ImagesResults/%s' % file[7:])
        Original = cv2.imread("Images/%s" % file[7:])

        # Prints the original, textured, and resulting image
        if '-p' in sys.argv:
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(Original, cv2.COLOR_BGR2RGB))
            plt.title("Imagem original")
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(ImageText, cmap='gray')
            plt.title("Imagem Texturizada")
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(Result, cv2.COLOR_BGR2RGB))
            plt.title("Imagem Resultante")
            plt.axis('off')
            plt.show()


        # Calculating PSNR's values of the real results
        print('PSNR: %lf' % cst.getPSNR(Original, Result))


if __name__ == '__main__':
    main()
