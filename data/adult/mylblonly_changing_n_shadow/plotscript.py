import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

n_shadow_models = [1, 2, 4, 8, 16, 32]
n_noise_examples = [1, 10, 50, 100, 200, 500, 1000]
path = "data/mylblonly_changing_n_shadow/"

my_lblonly = []
for s in n_shadow_models:
    dictionary = {"p": [], "r": [], "f1": []}
    for n in n_noise_examples:
        with open(path+"mtlblonly_report_{}_{}.txt".format(s, n), 'r') as file:
            data = file.read().split('\n')
            values = data[3].split(" ")
            p = float(values[18])
            r = float(values[24])
            f1 = float(values[30])
            dictionary["p"].append(p)
            dictionary["r"].append(r)
            dictionary["f1"].append(f1)
    my_lblonly.append(dictionary)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.suptitle('Mylbl only with different shadown models and noise examples.')
for i in range(len(my_lblonly)):
    choosen_dictionary = my_lblonly[i]
    rsmoothed = gaussian_filter1d(choosen_dictionary['r'], sigma=2)
    f1smoothed = gaussian_filter1d(choosen_dictionary['f1'], sigma=2)
    psmoothed = gaussian_filter1d(choosen_dictionary['p'], sigma=2)
    ax1.plot(n_noise_examples, psmoothed, label=n_shadow_models[i])
    ax2.plot(n_noise_examples, rsmoothed, label=n_shadow_models[i])
    ax3.plot(n_noise_examples, f1smoothed, label=n_shadow_models[i])
ax1.legend()

ax1.set_title("Precision comparison")
ax1.grid(visible=True)

ax2.legend()

ax2.set_title("Recall comparison")
ax2.grid(visible=True)

ax3.legend()
ax3.set(xlabel='number of noise examples', ylabel='percentage')
ax3.set_title("F1 comparison")
ax3.grid(visible=True)
plt.show()
