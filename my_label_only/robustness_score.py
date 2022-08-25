import pandas as pd
import random
import math
import numpy as np
from tqdm import tqdm


def bernoulli_noise(bin_values, p):
    """

    :param bin_values: index of the column in which we have binary values.
    :param p: porbability of changing the value.
    :return:
    """
    for i in range(len(bin_values)):
        r = random.uniform(0, 1)
        if r <= p:
            bin_values[i] = math.fabs(bin_values[i]-1)
    return bin_values


def neighborhood_noise(values, pd):
    """

    :param values: continuous values to be perturbed
    :param pd: percentage deviation (min, max) is the percentage of the value to add or subtrack.
    :return:
    """
    for i in range(len(values)):
        r = random.uniform(pd[0], pd[1])
        r = round(r, 2)
        if np.random.randint(2, size=1)[0] == 1:  # 50% of probability to be added or subtracted
            r *= -1
        values[i] += round(values[i] * r, 3)
    return values


def robustness_score(model, dataset: pd.DataFrame, n):
    """
    Compute the robustness score for each row inside the dataset.
    :param model: model to get the labels.
    :param dataset:
    :param n: number of perturbations.
    :return: score of robustness is a value 0<rob_score<1
    """
    fb = 0.60  # probability of flipping one bit.
    con_vals = 6  # number of continues values before the bit ones.
    percentage_deviation = (0.1, 0.50)  # min max of the percentage of the value to add or subtrack.
    scores = []
    for row in tqdm(dataset):
        variations = []
        for i in range(n):
            perturbed_row = row.copy()
            perturbed_row[:con_vals] = neighborhood_noise(perturbed_row[:con_vals], percentage_deviation)
            perturbed_row[con_vals:] = bernoulli_noise(perturbed_row[con_vals:], fb)
            variations.append(perturbed_row)
        variations = np.array(variations)
        output = model.predict(variations, verbose=0)  # we pass the variations inside the model
        _, c = np.unique(output, return_counts=True)  # we obtain the count of the majority output.
        score = c.max()/n  # we scale the value by dividing it to n.
        # print(score)
        scores.append(score)
    return scores


def robustness_score_label(model, dataset: pd.DataFrame, label, n):
    """
    Compute the robustness score for each row inside the dataset with the true label passed
    as parameter and in case of miss classification we set the score to 0.
    :param model: model to get the labels.
    :param dataset:
    :param n: number of perturbations.
    :return: score of robustness is a value 0<rob_score<1
    """
    fb = 0.60  # probability of flipping one bit.
    con_vals = 6  # number of continues values before the bit ones.
    percentage_deviation = (0.1, 0.50)  # min max of the percentage of the value to add or subtrack.
    scores = []
    index = 0
    for row in tqdm(dataset):
        variations = []
        y_true = label[index]
        y_predicted = model.predict(np.array([row]))[0]
        # y_predicted = np.argmax(y_predicted) if len(y_predicted) > 1 else y_predicted
        if y_true == y_predicted:
            for i in range(n):
                perturbed_row = row.copy()
                perturbed_row[:con_vals] = neighborhood_noise(perturbed_row[:con_vals], percentage_deviation)
                perturbed_row[con_vals:] = bernoulli_noise(perturbed_row[con_vals:], fb)
                variations.append(perturbed_row)
            variations = np.array(variations)
            output = model.predict(variations)  # we pass the variations inside the model
            # output = np.argmax(output, axis=1) if output.shape[1] > 1 else output
            score = np.mean(np.array(list(map(lambda x: 1 if x == y_true else 0, output))))
            # print(score)
            scores.append(score)
        else:
            scores.append(0)
        index += 1
    return scores


"""TS_PERC = 0.2
N_SHADOW_MODELS = 8
train_set = pd.read_csv("../data/adult_noise_shadow_labelled")
train_label = train_set.pop("class")
bb = RandomForestBlackBox()
print(train_label[0])
robustness_score(bb, train_set, 1000)"""

