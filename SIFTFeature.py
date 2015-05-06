import math
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import opencvUtil

SURF_PARAM = 400
MAX_CLUSTER = 500
surf = cv2.SURF(SURF_PARAM)

def calcSURFDescription(img):
    key, des = surf.detectAndCompute(img, None)
    return key, des

def measureFeatureDistance(feature1, feature2):
    if feature1.size != feature2.size:
        print "size of feature is different!"
        return 0

    return np.linalg.norm(feature1-feature2)


def findNearestFeature(BOF, feature):

    minDistance = 100000000
    idx = BOF.shape[0] + 1
    for i in range(BOF.shape[0]):
        dis = measureFeatureDistance(BOF[i], feature)
        if dis < minDistance:
            minDistance = dis
            idx = i

    return idx

def createBagOfFeatures(imagalist, isSave=True):
    totalDescription = np.array(0)

    for imagename in imagelist:
        print imagename.split("/")[-1] + " is calcurating..."
        im = cv2.imread(imagename)
        key, des = calcSURFDescription(im)

        # At first, create totalDescription array
        if totalDescription.size == 1:
            totalDescription = des.copy()

        else:
            totalDescription = np.r_[totalDescription, des]

    # K-means
    print "K-means is calcurating..."
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, BOF = cv2.kmeans(totalDescription, MAX_CLUSTER, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    if isSave == True:
        np.save("BOF", BOF)

    return BOF

def calcBOFHistograms(imagelist, BOF, isSave=True):
    # calc histogram
    histogram = np.zeros(len(imagelist)*BOF.shape[0])
    histogram = histogram.reshape((len(imagelist), BOF.shape[0]))

    for i, imagename in enumerate(imagelist):
        im = cv2.imread(imagename)
        key, des = calcSURFDescription(im)
        print "calcurate " + "No." + str(i) + "histogram."
        for j in range(des.shape[0]):
            idx = findNearestFeature(BOF, des[j])
            histogram[i][idx] += 1

    if isSave:
        np.save("SURFHistogram", histogram)

def calcBOFHistogramSimilarity(hist1, hist2):
    similarity = cv2.compareHist(hist1.astype('float32'), hist2.astype('float32'), cv2.cv.CV_COMP_CORREL)
    return similarity


# if __name__ == "__main__":
    #
    # BASE_DIR  = opencvUtil.getpwd()
    # INPUT_DIR = "sample/artwork/"
    #
    # imagelist = opencvUtil.getImagelist(BASE_DIR + INPUT_DIR)
    #
    # # here is to calcurate BOF imagelist
    # # BOF = np.load("BOF.npy")
    # # calcBOFHistograms(imagelist, BOF)
    #
    # histogram = np.load("SURFHistogram.npy")
    #
    # for i in range(histogram.shape[0]):
    #     cv2.normalize(histogram[i], histogram[i], 0, 255, cv2.NORM_MINMAX)
    #
    # # createReducedImage
    # # for imagename in imagelist:
    # #     if os.path.exists(INPUT_DIR + "resize/") == False:
    # #         os.mkdir(INPUT_DIR + "resize/")
    # #
    # #     img = cv2.imread(imagename)
    # #     img = opencvUtil.createNbitColorImage(img, 64)
    # #     opencvUtil.saveImage(INPUT_DIR + "resize/" + imagename.split("/")[-1], img)
    #
    #
    # imagelist = opencvUtil.getImagelist(INPUT_DIR + "resize/")
    # for i in range(1):
    #     print imagelist[i].split("/")[-1]
    #     img1 = cv2.imread(imagelist[i])
    #
    #     maxSimilarity = 0.0
    #     maxSimilarityIdx = 0
    #     similarityList = []
    #     for j in range(len(imagelist)):
    #         if i == j:
    #             continue
    #
    #         similarityDict = {}
    #         similarityDict['id'] = j
    #
    #         img2 = cv2.imread(imagelist[j])
    #         rgbSimilarity  = opencvUtil.calcColorHistogramSimilarity(img1, img2)
    #
    #         surfSimilarity = cv2.compareHist(histogram[i].astype('float32'), histogram[j].astype('float32'), cv2.cv.CV_COMP_CORREL)
    #         similarity = (rgbSimilarity + surfSimilarity) / 2.0 #opencvUtil.calcSimilarity(histogram[i], histogram[j])
    #
    #         similarityDict['similarity'] = similarity
    #
    #         if maxSimilarity < similarity:
    #             maxSimilarity = similarity
    #             maxSimilarityIdx = j
    #
    #         similarityList.append(similarityDict)
    #
    #
    #     #similarityList = sorted(similarityList, key=lambda s:s['similarity'], reverse=True)
    #     for j in range(12):
    #         print similarityList[j]['id'], similarityList[j]['similarity']
    #
    #
    #     #print imagelist[i].split("/")[-1] + " ... " + imagelist[maxSimilarityIdx].split("/")[-1] + "..." + str(maxSimilarity)
    #
    # # calc histogram similarity
    # # for i in range(len(imagelist)):
    # #     similarity = opencvUtil.calcSimilarity(histogram[0], histogram[i])
    # #     print str(i) + ": " + str(similarity)
    #
    # # xrange = np.arange(BOF.shape[0])
    # # plt.plot(xrange, histogram1)
    # # plt.plot(xrange, histogram2)
    # # plt.show()
