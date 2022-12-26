from keras.models import Sequential
from keras.layers import Dense
import pandas
import numpy as np
import pandas as pd

# np.random.seed(200)
# from tensorflow import set_random_seed
# set_random_seed(2)

# load training data
from models.adult.trepan.trepan import Oracle, Trepan
from bboxes.nnbb import NeuralNetworkBlackBox

ds_name = "bank"
trainX = pd.read_csv("../../../data/{}/original_train_set.csv".format(ds_name)).values
testX = pd.read_csv("../../../data/{}/original_test_set.csv".format(ds_name)).values
trainY = pd.read_csv("../../../data/{}/original_train_label.csv".format(ds_name)).values
testY = pd.read_csv("../../../data/{}/original_test_label.csv".format(ds_name)).values
num_classes = 2
total_num_examples = trainX.shape[0]
print(num_classes, total_num_examples)

# build oracle
model = NeuralNetworkBlackBox(db_name=ds_name, regularized=False)
oracle = Oracle(model, num_classes, trainX)

# build tree with TREPAN
MIN_EXAMPLES_PER_NODE = 30
MAX_NODES = 200
root = Trepan.build_tree(MIN_EXAMPLES_PER_NODE, MAX_NODES, trainX, oracle)

# calculate train fidelity
num_test_examples = trainX.shape[0]
correct = 0
ann_prediction = oracle.get_oracle_label(trainX)
for i in range(0, num_test_examples):
    tree_prediction = root.classify(trainX[i, :])
    correct += (ann_prediction[i] == tree_prediction)

fidelity = float(correct) / num_test_examples
print("Fidelity of the model is : " + str(fidelity))


# calculate test fidelity
num_test_examples = testX.shape[0]
correct = 0
ann_prediction = oracle.get_oracle_label(testX)
for i in range(0, num_test_examples):
    tree_prediction = root.classify(testX[i, :])
    correct += (ann_prediction[i] == tree_prediction)

fidelity = float(correct) / num_test_examples
print("Test Fidelity of the model is : " + str(fidelity))
