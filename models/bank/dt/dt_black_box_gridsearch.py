from sklearn import tree
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import classification_report
import pickle


train_set = pd.read_csv("./data/adult_original_train_set.csv")
test_set = pd.read_csv("./data/adult_original_test_set.csv")
train_label = pd.read_csv("./data/adult_original_train_label.csv")
test_label = pd.read_csv("./data/adult_original_test_label.csv")

tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': [20, 40, 80, 100, 300, 400, 500],
             'min_samples_split': [5, 10, 15, 25, 30, 50], 'min_samples_leaf': [3, 5, 15, 20, 40, 50],
             'max_features': [1, 3, 5, 'auto', 'sqrt', 'log2']}
grid = GridSearchCV(tree.DecisionTreeClassifier(), tree_para, cv=5, n_jobs=12, verbose=10, scoring='accuracy')
grid.fit(train_set, train_label)
y_pred_acc = grid.predict(test_set)
report = classification_report(test_label, y_pred_acc)
write_report = open("models/dt/measures_grid_search_dt.txt", "w")
write_report.write(report)
print(grid.best_params_)

title = "models/dt/best_param_gridsearch_dt.txt"
with open(title, 'wb') as fp:
    pickle.dump(str(grid.best_params_), fp)