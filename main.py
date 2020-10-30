# -*- coding: utf-8 -*-


import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob2 import glob
import os, os.path
import time

global folders_path
global path
import os

global cm
import imutils
import warnings
import pandas as pd
from skimage import feature
import mahotas as mt
import skimage
from skimage.feature import greycomatrix, greycoprops

pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')
features1 = []
labels = []
bin_n = 16
GLCM_step = 20
numPoints = 24
radius = 8


def haralick(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # extrai os recursos haralick para todos os 4 tipos de adjacência
    textures = mt.features.haralick(img_gray)

    # encontra a média de todos os 4 tipos de GLCM
    ht_mean = textures.mean(axis=0)
    ht_mean = cv2.normalize(ht_mean, ht_mean).flatten()
    # retorna o vetor de recurso resultante para a imagem que descreve a textura
    return ht_mean


def describeLbp(image, eps=1e-7):
    # calcular LBP da imagem e use a representação LBP para criar o histograma de padrões
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    # NORMALIZAÇÃO DO HISTOGRAMA
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # retorna o histograma
    return hist


def momentsHu(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    momentshu = cv2.HuMoments(cv2.moments(image)).flatten()

    # momentshu = cv2.normalize(momentshu, momentshu).flatten()
    return momentshu


def histogram(image):
    # extrai um histograma de cores 3D da região mascarada da imagem,
    # usando o número fornecido de posições por canal
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # equ = cv2.equalizeHist(image)
    hist = cv2.normalize(hist, hist).flatten()

    # retorna o histograma
    return hist


def hog2(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))
    # quantquantizando binvalues em (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist é um vetor de 64 bits
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def zernike(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # radius
    radius = 10
    degree = 10

    # computing zernike moments
    value = mt.features.zernike_moments(image, degree, radius)

    return value

def SIFT(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SURF extraction
    sift = cv2.SIFT_create()
    descriptors_1 = sift.detectAndCompute(img,None)



    print(descriptors_1)

    return np.array([descriptors_1], dtype=np.float32)






# Função que seleciona a base de dados de imagens para extração
# das características e já aplica o extrator, gerando o arquivo com as
# características de todas as imagens

count = 0
global folders_path, imageID, files_path
# medir o tempo de execução
t0 = time.time()
# Abre uma janela para selecinar a base de dados
# folders = filedialog.askdirectory(parent=window,initialdir="/",title='Por favor selecione o diretório das imagens')


# Ir até o caminho em que estamos atualmente
folders_path = os.path.realpath('/Users/admin/Documents/Jasiel/Database/CBIS-DDSM/Training/Calc') + '/'
# obter todas as pastas no caminho
folders = glob(folders_path + '**/')
# vetor de caracterpistica que vai receber as imagens
img_files = []

# abre arquivo csv para escrita
# abre o arquivo para salvar as características das imagens
output = open("ORB.csv", "w")
output1 = open("datasel.txt", "w")
countImage = 0
countPasta = 0

# Obter apenas os arquivos de imagem em cada pasta
for folder in folders:

    files_path = folder
    print('\nFolder: ' + files_path)
    img_files = []

    # texto.insert(INSERT,'\nFolder: ' + files_path)
    img_files.extend(glob(files_path + '*.JPG'))
    img_files.extend(glob(files_path + '*.JPEG'))
    img_files.extend(glob(files_path + '*.BMP'))
    img_files.extend(glob(files_path + '*.PNG'))
    img_files.extend(glob(files_path + '*.dcm'))
    percent = countPasta / len(folders) * 100
    percent_text = '\nCriando vetores de características ' + str(int(percent)) + '%'
    print(percent_text, end='\r', flush=True)
    for i, files_path in enumerate(img_files):
        # read image file

        countImage += 1
        # obter o nome da imagem e a pasta
        imageID = files_path[files_path.rfind("/") + 1:]
        image_ID = os.path.basename(imageID)
        label1 = files_path.split(os.path.sep)[-3].split("/")[0]
        label = os.path.join(label1, files_path.split(os.path.sep)[-2].split("/")[0])
        # ler imagem em escala de cinza
        image = cv2.imread(files_path, 1)

        # Aplicar os extratores em cada imagem
        # features = haralick(image)
        # features = describeLbp(image)
        # features = momentsHu(image)
        # features = histogram(image)
        # features = hog2(image)
        features = zernike(image)
        # features = SIFT(image)
        # features = CreateGLCM(image)

        # fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
        # features = feature.hog(image, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        # features = exposure.rescale_intensity(features, out_range=(0, 255))
        # features = features.astype("uint8")
        # exit

        # print(label)
        # print(label[14:21:])

        # para cada atributo da imagem
        features = ['{:f}'.format(f) for f in features]
        # Salve o vetor de característica em um arquivo
        output.write("%s,%s\n" % (label[14:30:], ",".join(features)))
        # output1.write("%s %s %s\n" % (countImage, countPasta, " ".join(features)))

        features1.append(features1)
        labels.append(label)
    countPasta += 1

# fecha o arquivo
output.close()
output1.close()

if len(img_files) > 0:
    print('\nBusca finalizada,', countImage, 'imagem(s) encontradas.')
    # texto.insert(INSERT,'\nBusca completa, total de imagens encontradas: ' , countImage)

else:
    print("Erro", "Nenhuma imagem encontrada")
    # texto.insert(INSERT, "\nErro", "Nenhuma imagem encontrada")
    exit()

t1 = time.time()
print('\nArquivo com as características criado, tempo:', t1 - t0)
# texto.insert(INSERT,'\nArquivo com as características criado, tempo:'+ str(t1-t0))
