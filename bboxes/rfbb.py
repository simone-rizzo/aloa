from sklearn.ensemble import RandomForestClassifier
from bboxes.bb_wrapper import SklearnClassifierWrapper
import pickle


class RandomForestBlackBox(SklearnClassifierWrapper):
    def __init__(self):
        filename = "C:/Users/Simone/Documents/MIA/models/rf/measures_rf_black_box_original.sav"
        self.model = pickle.load(open(filename, 'rb'))

    def model(self):
        return self.model()

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def train_model(self, x, y):
        rf = RandomForestClassifier()
        rf.fit(x, y)
        return rf



"""
Testing the wrapper Abstract class with the model file format .sav

import pandas as pd
from sklearn.metrics import classification_report
bb = RandomForestBlackBox()

test_set = pd.read_csv("../data/adult_original_test_set.csv", index_col=0)
test_label = pd.read_csv("../data/adult_original_test_label.csv", index_col=0)
predictions1 = bb.predict(test_set)
report = classification_report(test_label, predictions1)
print(report)"""