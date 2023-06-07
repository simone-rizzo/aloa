import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import make_interp_spline
import pandas as pd


def estract_accuracy(testo):
    righe = testo.split('\n')
    acc = righe[5][41:43]
    return int(acc)


if __name__ == "__main__":
    """attack_accuracy = []
    depths = []
    test_fidelity = []
    train_fidelity = []
    folders = os.listdir("./xai_tradeoff")
    for depth in range(2, 80, 2):
        depths.append(depth)
        with open("./xai_tradeoff/{}/tr_fidelity.txt".format(depth), "r") as f:
            testo = f.read()
            acc = estract_accuracy(testo)
            train_fidelity.append(acc)
        with open("./xai_tradeoff/{}/ts_fidelity.txt".format(depth), "r") as f:
            testo = f.read()
            acc = estract_accuracy(testo)
            test_fidelity.append(acc)
        with open("./xai_tradeoff/{}/attack_ts.txt".format(depth), "r") as f:
            testo = f.read()
            acc = estract_accuracy(testo)
            attack_accuracy.append(acc)
    print(len(attack_accuracy))
    print(len(depths))
    print(len(test_fidelity))
    print(len(train_fidelity))
    df = pd.DataFrame(depths, columns=["depth"])
    df['att_acc'] = attack_accuracy
    df['tr_fide'] = train_fidelity
    df['ts_fide'] = test_fidelity
    df.to_csv("./trade_off.csv", index=False)"""
    df = pd.read_csv("./trade_off.csv")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.plot(df['depth'].values, df['att_acc'].values, 'r--', label='Attack accuracy')
    ax.plot(df['depth'].values, df['att_acc'].values, 'r.')
    ax.plot(df['depth'].values, df['att_acc'].ewm(span=4).mean().values, 'r', label='Attack accuracy interpolation')
    ax2.plot(df['depth'].values, df['ts_fide'].values, 'g--', label="Fidelity")
    ax2.plot(df['depth'].values, df['ts_fide'].values, 'g.')
    ax2.plot(df['depth'].values, df['ts_fide'].ewm(span=4).mean().values, 'g', label='Fidelity interpolation')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.xticks(np.arange(2, 80, 2.0))
    # plt.yticks(np.arange(df['att_acc'].min(), df['ts_fide'].max() + 1, 1.0))
    ax.set(xlabel='depth of the Tree')
    ax.set_ylabel('Attack accuracy', color='r')
    ax2.set_ylabel('Fidelity', color='g')
    plt.title("Showing the trade off between Privacy risk, Explainability and Fidelity")
    plt.grid(True)
    plt.show()