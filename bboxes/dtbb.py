from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from bboxes.bb_wrapper import SklearnClassifierWrapper
import pickle
import matplotlib.pyplot as plt


class DecisionTreeBlackBox(SklearnClassifierWrapper):
    def __init__(self, db_name, regularized, explainer=False):
        self.regularized = regularized
        self.model_name = "dt"
        if explainer:
            # We import the regularized explainer meaning it has less depth better explainability.

            filename = "../new_trepan/explainers/{}/rf/regularized/explainer{}.sav".format(db_name, "_lssdpt" if regularized else "")
            self.model = pickle.load(open(filename, 'rb'))
        else:
            if not regularized:
                filename = "../models/{}/dt/dt_blackbox.sav".format(db_name)
                self.model = pickle.load(open(filename, 'rb'))
            else:
                filename = "../models/{}/dt/dt_blackbox_regularized.sav".format(db_name)
                self.model = pickle.load(open(filename, 'rb'))

    def model(self):
        return self.model

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def train_model(self, x, y):
        dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=2)
        dt.fit(x, y)
        return dt


if __name__ == "__main__":
    db_name = "adult"
    bb = DecisionTreeBlackBox(db_name=db_name, regularized=False, explainer=False)
    import pandas as pd
    train_set = pd.read_csv("../data/{}/original_train_set.csv".format(db_name))
    test_set = pd.read_csv("../data/{}/original_test_set.csv".format(db_name))
    train_label = pd.read_csv("../data/{}/original_train_label.csv".format(db_name))
    test_label = pd.read_csv("../data/{}/original_test_label.csv".format(db_name))

    # Performances on training set
    out = bb.predict_proba(train_set.values)
    train_prediction = bb.predict(train_set.values)
    report = classification_report(train_label, train_prediction)
    print(report)

    # Performances on test set
    test_prediction = bb.predict(test_set.values)
    report = classification_report(test_label, test_prediction)
    print(report)

    # create ROC curve
    bb_r = DecisionTreeBlackBox(db_name=db_name, regularized=True, explainer=False)
    test_prediction_r = bb_r.predict(test_set.values)
    fpr, tpr, _ = metrics.roc_curve(test_label, test_prediction)
    plt.plot(fpr, tpr, label="overfitted")
    fpr, tpr, _ = metrics.roc_curve(test_label, test_prediction_r)
    plt.plot(fpr, tpr, label="regularized")
    plt.plot([0.10 * i for i in range(11)], [0.10 * i for i in range(11)], '--', c="red")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC curve on test")
    plt.legend()
    plt.grid()
    plt.show()