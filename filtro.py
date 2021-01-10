import cv2 as cv
import argparse
import math  
import numpy as np
import time

time_sleep = 1
class ImageNotFound(Exception):
    pass

def defineArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbose', help="Mais mensagens", required=False, default=False)
    parser.add_argument('-m','--method',  help="Método escolhido", required=True, default='media', choices=['media', 'mediana', 'gauss'])
    parser.add_argument('-x','--matrix',  help='Matriz', required=True, default=None)
    parser.add_argument('-i','--image',   help='Caminho da imagem', required=True, default=None)

    return parser.parse_args()

def getMethod(method):
    methods = {
        'media'   : media,
        'mediana' : mediana,
        'gauss'   : gauss
    }

    return methods[method]

def getMethodOpenCV(method):
    methods = {
        'media'   : media,
        'mediana' : medianaOpenCV,
        'gauss'   : gaussOpenCV
    }

    return methods[method]
    
def getMatrix(file):
    file = open(file, 'r')
    matrix = file.readline()
    
    new_matrix = getNewMatrix(matrix)
    size = math.sqrt(len(new_matrix))

    if (size % 2 == 0):
        print('Erro, a matriz precisa ser 1x1, 3x3, 5x5...')
        exit()

    return matrix

def getImage(path):
    try:
        image = cv.imread(path)

        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    except(ImageNotFound):
        print ('Imagem não encontrada')
        quit()

def saveImage(image, path):
    cv.imwrite(path, image) 

def showImage(image):
    cv.imshow('Imagem', image)
    cv.waitKey()

def showMethod(method):
    print('Usando {}'.format(method))

def getNewMatrixOfZeros(altura, largura):
    return np.zeros((altura, largura,1), np.uint8)

def getNewMatrix (matrix):
    return [int(s) for s in matrix.strip().split(' ')]

def getConfiguration(image, matrix):
    new_matrix = getNewMatrix(matrix)

    altura  = image.shape[0]
    largura = image.shape[1]
    
    nova_imagem = getNewMatrixOfZeros(altura, largura)

    return altura, largura, new_matrix, nova_imagem

def getSubMatrix(image, size, i, j):
    subMatrix = image[i:i + size,j:j + size]
    
    return subMatrix


def media(image, matrix):
    altura, largura, new_matrix, nova_imagem = getConfiguration(image, matrix)
    size = int(math.sqrt(len(new_matrix)))

    for i in range(altura):
        for j in range(largura):
            nova_imagem[i][j] = getMedia(image, new_matrix, size, i, j, altura, largura)

    return nova_imagem

def getMedia(image, new_matrix, size, index_x, index_y, max_size_height, max_size_width):
    media = 0
    numberOfElements = 0
    tmp = int(size/2)
    for i in range(0, size):
        for j in range(0, size):
            new_index_x = index_x - tmp + i
            new_index_y = index_y - tmp + j
            if (isInRange(max_size_height, max_size_width, new_index_x, new_index_y)):
                media += image[new_index_x][new_index_y] * new_matrix[i * i + j]
                numberOfElements += new_matrix[i * i + j]

    return int(media/(numberOfElements))

def isInRange(max_height, max_width, new_index_x, new_index_y):
    return (new_index_x >= 0 and new_index_x < max_height) and (new_index_y >= 0 and new_index_y < max_width)

def mediana(image, matrix):
    altura, largura, new_matrix, nova_imagem = getConfiguration(image, matrix)
    size = int(math.sqrt(len(new_matrix)))

    for i in range(altura):
        for j in range(largura):
            nova_imagem[i][j] = getMediana(getSubMatrix(image, size, i, j))

    return nova_imagem

def getMediana(matrix):
    array = np.concatenate(matrix)
    
    return np.median(array)

def medianaOpenCV(image, size):
    return cv.medianBlur(image, size)

def gauss(image, matrix):
    print('gauss')
    pass

def gaussOpenCV(image):
    pass




if __name__ == "__main__":
    arguments = defineArguments()

    matrix = getMatrix(arguments.matrix)
    method = getMethod(arguments.method)
    methodOpenCV = getMethodOpenCV(arguments.method)
    image  = getImage(arguments.image)

    showMethod(arguments.method)
    newImage = method(image, matrix)
    saveImage(newImage, 'meu_metodo.jpg')

    showMethod("{} OPEN CV".format(arguments.method))
    newImage = methodOpenCV(image, 5)
    saveImage(newImage, 'metodo_openCV.jpg')
    pass