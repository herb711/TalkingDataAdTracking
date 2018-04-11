#### Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn.utils.multiclass import _check_partial_fit_first_call
import pickle
from sklearn.externals import joblib

class modle(object):
    def __init__(self, modle_n=0):
        #生成模型
        if modle_n == 0: #支持向量机 
            self.clf = svm.SVC()
        elif modle_n == 1:
            self.clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        '''
        clf = svm.SVC(C=1.0, cache_size=200, class_weight=None,
            coef0=0.0, decision_function_shape=None,
            degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False,
            random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        ''' 
    def train(self, X, y, n=0):
        if n==0:
            #训练模型
            self.clf.fit(X,y) 
        else:
            #分批训练模型
            print("train...{0}".format(n))
            self.clf.partial_fit(X, y)
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