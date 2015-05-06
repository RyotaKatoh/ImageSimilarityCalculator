import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import copy
import math

# return current directory
def getpwd():
    FILE_PATH = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(FILE_PATH)
    if BASE_DIR[:-1] != "/":
        BASE_DIR += "/"
    return BASE_DIR

def getImagelist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

# resize img size to (width, height)
def resize(img, width, height):
    resizedImage = cv2.resize(img, (width, height))
    return resizedImage

# trim img to (x, y) to (x + w, y + h)
def trim(img, x, y, w, h):
    trimedImage = img[y:y+h, x:x+w]
    return trimedImage

# reduce RGB to (32, 96, 120, 224)
def reduceColor(img):
    return 0


def saveImage(filename, img):
    cv2.imwrite(filename, img)
    print filename + " is successfully saved."


def createLowResolutionImage(inputDir, outputDir, width, height):
    imagelist = getImagelist(inputDir)

    for imagename in imagelist:
        img = cv2.imread(imagename)
        w, h = img.shape[:2]

        if w != h:
            m = min(w, h)
            img = trim(img, w/2 - m/2, h/2 - m/2, m, m)

        img = resize(img, width, height)

        filename = imagename.split('/')[-1]
        saveImage(outputDir + filename, img)


def create64bitColorImage(img):
    reducedImage = copy.copy(img)
    w, h = reducedImage.shape[:2]
    for i in range(h):
        for j in range(w):
            for r in range(3):

                if img[i, j, r] < 64:
                    reducedImage[i, j, r] = 32

                elif img[i, j, r] < 128:
                    reducedImage[i, j, r] = 96

                elif img[i, j, r] < 192:
                    reducedImage[i, j, r] = 160

                else:
                    reducedImage[i, j, r] = 224

    return reducedImage

def calc64bitColorHistogram(reducedimg):
    histogram = np.zeros(64)

    w, h = reducedimg.shape[:2]
    for y in range(h):
        for x in range(w):
            r = reducedimg[y, x, 0]
            g = reducedimg[y, x, 1]
            b = reducedimg[y, x, 2]

            pattern = ((r+32)/64-1) + ((g+32)/64-1) * 4 + ((b+32)/64-1) * 16
            histogram[pattern] += 1

    return histogram

# create Nbit color Image
def createNbitColorImage(img, N):
    reducedImg = copy.copy(img)
    w,h = reducedImg.shape[:2]

    numDivision = math.floor( pow(N, 1/3.) + 0.5 )
    pixelWidth = 256 / numDivision

    for y in range(h):
        for x in range(w):
            for r in range(3):
                index = int(reducedImg[y, x, r] / pixelWidth)
                reducedImg[y, x, r] = pixelWidth / 2. + index * pixelWidth


    return reducedImg

def calcNbitColorHistogram(nBitImage, N):
    numDivision = math.floor( pow(N, 1/3.) + 0.5 )
    pixelWidth = 256 / numDivision

    histogram = np.zeros(N)
    w, h = nBitImage.shape[:2]
    for y in range(h):
        for x in range(w):
            r = nBitImage[y, x, 0]
            g = nBitImage[y, x, 1]
            b = nBitImage[y, x, 2]

            pattern = ((r+pixelWidth/2)/pixelWidth-1) + ((g+pixelWidth/2)/pixelWidth-1) * numDivision + ((b+pixelWidth/2)/pixelWidth-1) * (numDivision**2)
            histogram[pattern] += 1

    return histogram

def calcRGBHistogram(img, isNormalized=True):
    h_b = cv2.calcHist([img], [0], None, [256], [0, 255])
    h_g = cv2.calcHist([img], [1], None, [256], [0, 255])
    h_r = cv2.calcHist([img], [2], None, [256], [0, 255])

    if isNormalized == True:
        cv2.normalize(h_b, h_b, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(h_g, h_g, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(h_r, h_r, 0, 255, cv2.NORM_MINMAX)

    return h_b, h_g, h_r

def calcColorHistogramSimilarity(img1, img2):
    h_b1, h_g1, h_r1 = calcRGBHistogram(img1)
    h_b2, h_g2, h_r2 = calcRGBHistogram(img2)

    bSimilarity = cv2.compareHist(h_b1, h_b2, cv2.cv.CV_COMP_CORREL)
    gSimilarity = cv2.compareHist(h_g1, h_g2, cv2.cv.CV_COMP_CORREL)
    rSimilarity = cv2.compareHist(h_r1, h_r2, cv2.cv.CV_COMP_CORREL)

    return (bSimilarity + gSimilarity + rSimilarity)/3.0


# def calcAllSimilarity(imagelist, N):
#     histograms = []
#
#     for imagename in imagelist:
#         image = cv2.imread(imagename)
#         reducedImage = create64bitColorImage(image)
#
#         print "calc " + imagename.split("/")[-1] + " historgram"
#         hist = calc64bitColorHistogram(reducedImage)
#         histograms.append(hist)
#
#     result = []
#
#     for i, hist1 in enumerate(histograms):
#
#         similarities = []
#         dict = {}
#         dict["targetImageFilename"] = imagelist[i]
#         dict["similarImageFilename"] = ""
#         dict["similarity"] = 0.0
#         for j, hist2 in enumerate(histograms):
#             if i != j:
#                 similarity = calcSimilarity(hist1, hist2)
#
#                 if similarity > dict["similarity"]:
#                     dict["similarity"] = similarity
#                     dict["similarImageFilename"] = imagelist[j]
#
#                 # similarities.append(similarity)
#
#         result.append(dict)
#         # maxIdx = similarities.index(max(similarities))
#         # print imagelist[maxIdx].split("/")[-1] + ": " +str(max(similarities))
#
#
#     return result


# main function
# if __name__ == "__main__":
#
#     IMAGE_PATH = "sample/artwork/"
#     BASE_DIR = getpwd()
#     imagelist = getImagelist(BASE_DIR + IMAGE_PATH)
#
#     for i in range(15):
#         img1 = cv2.imread(imagelist[i])
#         print imagelist[i].split("/")[-1]
#         for j in range(15):
#             img2 = cv2.imread(imagelist[j])
#             print imagelist[j].split("/")[-1] + "..." + str(calcColorHistogramSimilarity(img1, img2))
