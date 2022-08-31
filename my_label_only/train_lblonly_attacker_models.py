import pickle
import numpy as np
import pandas as pd
import os
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree

path = "../data/shadow_label_only"
# path = "../data/shadow_label_only_truelabel"
attack_dataset = pd.concat([pd.read_csv(path+"/"+n) for n in os.listdir(path)])
classes = list(attack_dataset['class_label'].unique())
for c in classes:
    print("Class:{}".format(c))
    tr = attack_dataset[attack_dataset['class_label'] == c]
    tr.pop('class_label')
    tr_label = tr.pop('target_label')

    # Print of the unbalanced dataset
    unique, counts = np.unique(tr_label, return_counts=True)
    print(np.asarray((unique, counts)).T)

    # Undersampling and splitting
    undersample = RandomUnderSampler(sampling_strategy="majority")
    tr, tr_label = undersample.fit_resample(tr, tr_label)

    # Print after the balancing.
    unique, counts = np.unique(tr_label, return_counts=True)
    print(np.asarray((unique, counts)).T)

    train_set, test_set, train_label, test_label = train_test_split(tr, tr_label, stratify=tr_label,
                                                                    test_size=0.20, random_state=1)

    # We train the attacker model.
    # mdl = tree.DecisionTreeClassifier()
    mdl = RandomForestClassifier()
    mdl.fit(train_set.values, train_label.values)

    # Prediction and report of the performances.
    pred = mdl.predict(test_set.values)
    report = classification_report(test_label, pred)
    print(report)

    # Saving of the model.
    filename = '../attacker/lblonly_attacker_class_{}.sav'.format(c)
    # filename = '../attacker/lblonly_attacker_truelabel_class_{}.sav'.format(c)
    pickle.dump(mdl, open(filename, 'wb'))

