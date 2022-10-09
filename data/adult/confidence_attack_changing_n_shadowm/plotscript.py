import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

n_shadow_models = [i+1 for i in range(32)]
performances_dictionary = {"p": [], "r": [], "f1": []}
for s in n_shadow_models:
    with open("data/confidence_attack_changing_n_samples/confidence_report_nshadow{}.txt".format(s), 'r') as file:
        data = file.read().split('\n')
        values = data[3].split(" ")
        p = float(values[18])
        r = float(values[24])
        f1 = float(values[30])
        performances_dictionary["p"].append(p)
        performances_dictionary["r"].append(r)
        performances_dictionary["f1"].append(f1)

# fig, (ax1, ax2) = plt.subplots(1, 2)
plt.title("Confidence attack by changing the number of shadow model")
plt.plot(n_shadow_models, performances_dictionary['p'], label="precision")
plt.plot(n_shadow_models, performances_dictionary['r'], label="recall")
plt.plot(n_shadow_models, performances_dictionary['f1'], label="f1")
rsmoothed = gaussian_filter1d(performances_dictionary['r'], sigma=2)
f1smoothed = gaussian_filter1d(performances_dictionary['f1'], sigma=2)
psmoothed = gaussian_filter1d(performances_dictionary['p'], sigma=2)
plt.plot(n_shadow_models, rsmoothed,'--')
plt.plot(n_shadow_models, f1smoothed,'--')
plt.plot(n_shadow_models, psmoothed,'--')
plt.legend()
plt.xlabel('number of shadow models')
plt.ylabel('percentage')
plt.grid(True)
plt.yticks(np.arange(min(performances_dictionary['p']), max(performances_dictionary['r'])+0.001, 0.01))
plt.show()