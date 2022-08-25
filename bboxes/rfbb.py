from bboxes.bb_wrapper import SklearnClassifierWrapper
import pickle


class RandomForestBlackBox(SklearnClassifierWrapper):
    def __init__(self):
        filename = "C:/Users/Simone/Documents/MIA/models/rf/measures_rf_black_box_original.sav"
        bb = pickle.load(open(filename, 'rb'))
        super().__init__(bb)

    def model(self):
        return super().model()

    def predict(self, x):
        return super().predict(x)

    def predict_proba(self, x):
        return super().predict_proba(x)


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