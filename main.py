import sys
import cv2
import numpy as np
import opencvUtil
import SIFTFeature

argvs = sys.argv
argc = len(argvs)

'''
mode number:
0: calcurate Bag of Features
1: calcurate SURF Histogram
2: calcurate Image Similarity
'''

if argc != 2 or int(argvs[1]) >2:
    print "Usage: $python %s [mode number]" % argvs[0]
    quit()


MODE = int(argvs[1])
BASE_DIR  = opencvUtil.getpwd()
INPUT_DIR = "sample/artwork/"
imagelist = opencvUtil.getImagelist(BASE_DIR + INPUT_DIR)

def calcColorAndBOFSimilarity(id1, id2):
    img1 = cv2.imread(imagelist[id1])
    img2 = cv2.imread(imagelist[id2])
    rgbSimilarity  = opencvUtil.calcColorHistogramSimilarity(img1, img2)
    surfSimilarity = SIFTFeature.calcBOFHistogramSimilarity(histogram[id1], histogram[id2])

    similarity = (rgbSimilarity + surfSimilarity) / 2.0
    return similarity


if MODE == 0:
    SIFTFeature.createBagOfFeatures(imagelist)

elif MODE == 1:
    BOF = np.load("BOF.npy")
    SIFTFeature.calcBOFHistograms(imagelist, BOF)

elif MODE == 2:

    histogram = np.load("SURFHistogram.npy")

    for i in range(histogram.shape[0]):
        cv2.normalize(histogram[i], histogram[i], 0, 255, cv2.NORM_MINMAX)


    imagelist = opencvUtil.getImagelist(INPUT_DIR + "resize/")
    for i in range(1):
        print imagelist[i].split("/")[-1]
        img1 = cv2.imread(imagelist[i])

        similarityList = []
        for j in range(len(imagelist)):
            if i == j:
                continue

            similarityDict = {}
            similarityDict['id'] = j

            img2 = cv2.imread(imagelist[j])

            similarity = calcColorAndBOFSimilarity(i, j)

            similarityDict['similarity'] = similarity

            similarityList.append(similarityDict)


        similarityList = sorted(similarityList, key=lambda s:s['similarity'], reverse=True)
        for j in range(12):
            print similarityList[j]['id'], similarityList[j]['similarity']
