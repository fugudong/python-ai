from numpy import *
import operator
from os import listdir
import numpy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [1.0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
     key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def likeType2Int(likeType):
    if(likeType == "largeDoses"):
        return 2
    if(likeType == "smallDoses"):
        return 0
    if(likeType == "didntLike"):
        return 1

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if(filename == "datingTestSet.txt"):
            classLabelVector.append(likeType2Int(listFromLine[-1]))
        else:
            classLabelVector.append(int(listFromLine[-1]))
       # print(returnMat[index, :])
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    print(minVals)
    print(maxVals)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    print(ranges)
   # print(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals


def datingClass():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    print(datingLabels)
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)


def plotTestSetData():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * numpy.array(datingLabels),
               15.0 * numpy.array(datingLabels), 15.0 * numpy.array(datingLabels))
    plt.show()

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassProc():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print( "\nthe total number of errors is: %d" % errorCount)
    print( "\nthe total error rate is: %f" % (errorCount/float(mTest)))
if __name__ == '__main__':
    #group, labels = createDataSet()
    #print(classify0([0,0], group, labels, 3))

   #datingDataMat, dataingLabels = file2matrix("datingTestSet.txt")
   #print(datingDataMat)
   #print(dataingLabels)
   #normMat, ranges, minVals = autoNorm(datingDataMat)
   # handwritingClassProc()
    plotTestSetData()