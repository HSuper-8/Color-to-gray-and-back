# Color-to-gray-and-back

## Description
This Program was created to replicate the results achieved by Ricardo L. de Queiroz and Karen M. Braun in their paper "[Color to Gray and Back:Color Embedding Into Textured Gray Images](http://queiroz.divp.org/papers/color-to-bw.pdf)". The objective of this software was to insert the color information of a colored image inside its gray-scale format by using wavelet transform in order that we can recover the color afterwards. 


## Prerequisites
This program was written in Python 3.5 so the usage of it in another version is not guaranteed to work. Besides that, it is also required the following libraries to be installed:
- cv2
- numpy
- pywt
- glob
- pathlib
- sys
- matplotlib


## Running the Program

First, acess the directory of the source code via terminal and then execute the following command:
- python Main.py

By adding the '-p' flag in the above command, the original, textured, and rebuilt versions will be displayed in a window after rendering each image. It is required that the user closes the window displaying the current image to resume the execution.

In the program, it will be asked if the user wants to simulate the digitalization. If the user wants the simulation, then it will be necessary to inform the order of the resize that will be applied to the image. This resize is responsible to preserve part of the information that would have been lost in the process of digitalization.

In order that the software works, the images must be inserted in the 'Images' folder. The process will be aplied to all the images in this folder. Besides that, it is also reccomended that the images are in the PNG format, otherwise it is not guaranteed that the program will run properly.

### Output

The textured images will be outputed in the folder 'ImagesTextures' and the color recovered ones in the 'ImageResults' folder. The output images will have the same name as its original and will be in the PNG format.

At the end of the execution, it will be displayed in the terminal the recovered image and its equivalent PSNR value.	

## Authors
- Danilo Inácio dos Santos Silva
- Hevelyn Sthefany Lima de Carvalho 
- Felipe Lima Vaz
