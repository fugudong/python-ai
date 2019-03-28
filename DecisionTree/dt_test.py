from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('iris.txt')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
        labelMat.append(lineArr[4])
        #print(lineArr)
    label = list(set(labelMat))
    print(label)
    labelIntMat = []
    for data in labelMat:
        labelIntMat.append(label.index(data))
    return mat(dataMat),mat(labelIntMat).transpose(),label



if __name__ == '__main__':
    iris = load_iris()
    iris_feature, iris_target, label = loadDataSet()
    #iris_feature = iris.data
    #iris_target = iris.target
    accuracy = 0.0
    i = 0
    while i<10000:
        train_feature, test_feature, train_target, test_target = \
        train_test_split(iris_feature, iris_target, test_size=0.2, random_state=102)
        #clf = tree.DecisionTreeClassifier(criterion='entropy')
        print("gini")
        clf = tree.DecisionTreeClassifier(criterion='gini')
        clf = clf.fit(train_feature, train_target)
        predict = clf.predict(test_feature)
        accuracy += accuracy_score(predict, test_target)
        #print(accuracy_score(predict, test_target))
        i = i+1
    print(accuracy/10000)
    dot_data = tree.export_graphviz(clf, out_file=None,\
                                    class_names=label, filled=True ,rounded=True,special_characters=True)
                           #         class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    #graph.write_pdf("iris.pdf")