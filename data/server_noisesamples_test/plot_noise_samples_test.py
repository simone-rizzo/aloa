import matplotlib.pyplot as plt
import numpy as np

samples = [1, 10, 50, 100, 200, 500, 1000]
path1 = "./my_lblonly/"
path2 = "./original_lblonly/"

my_lblonly_dict = {"p": [], "r": [], "f1": []}
lblonly_dict = {"p": [], "r": [], "f1": []}
for s in samples:
    with open(path1+"mtlblonly_report_N_SAMPLES{}.txt".format(s), 'r') as file:
        data = file.read().split('\n')
        values = data[3].split(" ")
        p = float(values[18])
        r = float(values[24])
        f1 = float(values[30])
        my_lblonly_dict["p"].append(p)
        my_lblonly_dict["r"].append(r)
        my_lblonly_dict["f1"].append(f1)

    with open(path2+"report_N_SAMPLES{}.txt".format(s), 'r') as file:
        data = file.read().split('\n')
        values = data[3].split(" ")
        p = float(values[16])
        r = float(values[22])
        f1 = float(values[28])
        lblonly_dict["p"].append(p)
        lblonly_dict["r"].append(r)
        lblonly_dict["f1"].append(f1)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Comparison between my_lblonly and original_lblonly implementation')
ax1.plot(samples, my_lblonly_dict['p'], label="precision")
ax1.plot(samples, my_lblonly_dict['r'], label="recall")
ax1.plot(samples, np.array(my_lblonly_dict['f1']), label="f1")
ax2.plot(samples, lblonly_dict['p'], label="precision")
ax2.plot(samples, lblonly_dict['r'], label="recall")
ax2.plot(samples, lblonly_dict['f1'], label="f1")
ax1.legend()
ax1.set(xlabel='number of noise examples', ylabel='percentage')
ax1.set_title("My label only implementation")
ax1.grid(visible=True)
ax2.legend()
ax2.set(xlabel='number of noise examples', ylabel='percentage')
ax2.set_title("Origianl label only impl")
plt.grid(True)
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Comparison between my_lblonly and original_lblonly implementation')
ax1.plot(samples, my_lblonly_dict['p'], label="mylblonly")
ax1.plot(samples, lblonly_dict['p'], label="original_lblonly")
ax1.legend()
ax1.set(xlabel='number of noise examples', ylabel='percentage')
ax1.set_title("Precision comparison")
ax1.grid(visible=True)

ax2.plot(samples, my_lblonly_dict['r'], label="mylblonly")
ax2.plot(samples, lblonly_dict['r'], label="original_lblonly")
ax2.legend()
ax2.set(xlabel='number of noise examples', ylabel='percentage')
ax2.set_title("Recall comparison")
ax2.grid(visible=True)

ax3.plot(samples, my_lblonly_dict['f1'], label="mylblonly")
ax3.plot(samples, lblonly_dict['f1'], label="original_lblonly")
ax3.legend()
ax3.set(xlabel='number of noise examples', ylabel='percentage')
ax3.set_title("F1 comparison")
ax3.grid(visible=True)
plt.show()
