from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from  matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap


def linearPredict():
    boston = load_boston()
    x = boston.data
    y = boston.target
    print(boston.keys())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # 划分训练集和测试集
    lineR = LinearRegression()  # 线性模型
    lineR.fit(x_train, y_train)
    y_pred = lineR.predict(x_test)
    plt.subplot(211)
    plt.xlim([0, 50])
    plt.plot(range(len(y_test)), y_test, 'r', label='y_verify')
    plt.plot(range(len(y_pred)), y_pred, 'g--', label='y_predict')
    plt.title('sklearn: Linear Regression')
    plt.savefig('linear.png')
    '''----------输出模型参数、评价模型-----------'''
    print("线性回归")
    print("权重向量:%s, b的值为:%.2f" % (lineR.coef_, lineR.intercept_))
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('预测的准确率:',lineR.score(x_test, y_test))

    ridgeRegression = linear_model.Ridge()
    ridgeRegression.fit(x_train, y_train)
    y_pred = ridgeRegression.predict(x_test)
    plt.subplot(212)
    plt.xlim([0, 50])
    plt.plot(range(len(y_test)), y_test, 'r', label='y_verify')
    plt.plot(range(len(y_pred)), y_pred, 'g--', label='y_predict')
    plt.title('sklearn: Ridge Regression')
    plt.savefig('ridge.png')
    plt.legend()
    #plt.show()
    '''----------输出模型参数、评价模型-----------'''
    print("岭回归")
    print("权重向量:%s, b的值为:%.2f" % (ridgeRegression.coef_, ridgeRegression.intercept_))
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('预测的准确率:', ridgeRegression.score(x_test, y_test))



def visualFeatureTarget():
    #pd.set_option('display.max_columns', None)
    # 显示所有行列
    #pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
    #pd.set_option('max_colwidth', 200)
    cm_cycle = ListedColormap(['#000aa', '#ff5050', '#50ff50', '#9040a0', '#ff000'])
    #%matplotlib inline
    boston = load_boston()
    x = boston.data
    y = boston.target
    print(boston.feature_names)
    boston_df = pd.DataFrame(boston['data'], columns = boston.feature_names)
    boston_df['Target'] = pd.DataFrame(boston['target'], columns = ['Target'])
    print(boston_df.head(3))
    print(boston_df.corr().sort_values(by = ['Target'], ascending= False))

    sns.set(palette="muted", color_codes=True)
    #sns.pairplot(boston_df, vars=['CRIM', 'Target'])
    #plt.legend()

    for i in range(0, 13):
        plt.scatter(boston_df[boston.feature_names[i]], boston_df["Target"])
        plt.xlabel(boston.feature_names[i])
        plt.ylabel("Target")
        plt.show()


if __name__ == '__main__':

    visualFeatureTarget()
    linearPredict()


