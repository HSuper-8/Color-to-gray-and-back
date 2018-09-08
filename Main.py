import cv2
import pathlib
import numpy as np
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
    if(('-c' not in sys.argv) & (simulation)):
        k = int(input("Enter a resize order\n"))

    if '-c' in sys.argv:
        psnrs = []
        for k in range(1, 9):
            print('\n-- Resize Order = ', k)
            psnrs.append(cst.color2grayAndBack(k, simulation, sys.argv))
        psnrsArray = np.array(psnrs)
        #Display PSNR x Resize Order for each Image
        for k in range (0, len(psnrs[0])):
            plt.plot(psnrsArray[:,k], range(1, 9), 'bo')
            plt.title('Imagem %s' %(k+1))
            plt.xlabel('PSNR')
            plt.ylabel('Resize Order')
            plt.show()
    else:
        cst.color2grayAndBack(0, simulation, sys.argv)

if __name__ == '__main__':
    main()
