from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import pickle


class Classifier2D:
    def __init__(self):
        # TODO: make abstract property
        self.model = None

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_weights(self):
        return self.model.coef_[0]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


class LogReg(Classifier2D):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LogisticRegression(**kwargs)


class SVM(Classifier2D):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = SVC(**kwargs)


class RandomForest(Classifier2D):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(**kwargs)


class XGB(Classifier2D):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = XGBClassifier(**kwargs)


