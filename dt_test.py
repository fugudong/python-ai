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
    #print(iris.target_names)
    print(shape(iris_feature))
    print(shape(iris_target))
    train_feature, test_feature, train_target, test_target = \
    train_test_split(iris_feature, iris_target, test_size=0.2, random_state=102)
    #print(test_target)
    print("entropy")
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    clf = clf.fit(train_feature, train_target)
    predict = clf.predict(test_feature)
    print(accuracy_score(predict, test_target))
    print("gini")
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf = clf.fit(train_feature, train_target)
    predict = clf.predict(test_feature)
    print(accuracy_score(predict, test_target))
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=iris.feature_names,\
                                    class_names=label, filled=True ,rounded=True,special_characters=True)
                           #         class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    #graph.write_pdf("iris.pdf")