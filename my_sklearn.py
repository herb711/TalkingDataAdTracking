#### Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import numpy as np

class modle(object):
    def __init__(self, modle_n=0):
        #生成模型
        if modle_n == 0: #支持向量机 
            self.clf = svm.SVC()
        elif modle_n == 1:
            self.clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        elif modle_n == 2: #随机梯度下降
            self.clf = SGDClassifier()  # SGDClassifier的参数设置可以参考sklearn官网
        '''
        clf = svm.SVC(C=1.0, cache_size=200, class_weight=None,
            coef0=0.0, decision_function_shape=None,
            degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False,
            random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        ''' 
    def train(self, X, y):
        self.clf.fit(X,y) 
        self.out_clf()

    def train_batch(self, X, y, m=2, n=0):
        if m == 0: #支持向量机 
            pass
        elif m == 1:
            pass
        elif m == 2: #随机梯度下降
            # 使用 partial_fit ，并在第一次调用 partial_fit 的时候指定 classes
            self.clf.partial_fit(X, y, classes=np.array([0, 1]))
        print("train...{0}".format(n))  # 当前次数       
        self.out_clf()
        
    def in_clf(self):
        #导入模型
        self.clf = joblib.load('TalkingDataAdTracking/data/svm.pkl') 
        print('in_clf...ok')

    def out_clf(self): #保存模型  
        joblib.dump(self.clf, 'TalkingDataAdTracking/data/svm.pkl')
        print('out_clf...ok')


    def evaluate(self, X, y):
        e = self.clf.score(X, y) 
        print(e)
        return e

    def evaluate2(self, X, y):
        #简单评估
        e = cross_validation.cross_val_score(self.clf, X, y, cv=5)
        print(e)
        return e

    def predict(self, X):
        #预测
        return self.clf.predict(X)