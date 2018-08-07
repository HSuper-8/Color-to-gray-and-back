## Execução do programa

Acesse o diretório do código fonte pelo terminal e execute o seguinte comando:

- python Main.py

Será solicitado a escolha de simular ou não a distorção por impressão/digitalização. 
Caso seja confirmada a simulação, será solicitado a ordem do resize que será aplicado
na imagem, para tentar preservar parte da informação que se perderia no processo de 
impressão/digitalização.

Para o correto funcionamento do software, as imagens deverão ser inseridas na pasta ./Images
no diretório do código fonte. Além disso, recomenda-se que as imagens estejam no formato png,
pois não é garantida a generalidade do software para outros formatos. Os procedimentos do softwar, 
assim como a opção de simular e a ordem do resize, serão aplicados em todas as Imagens contidas na
pasta ./Images.As imagens codificas(Texturizadas) são salvas na pasta ./ImagesTextures e as 
decodificadas(Cor Recuperada) em ./ImagesResults. Ambas são salvas com o mesmo nome da imagem 
original e em formato png.

No final do processo de codificação e decodificação das imagens, é exibido no terminal o PSNR
para cada imagem de entrada e sua respectiva imagem resultante.



