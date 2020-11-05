import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axisartist.axislines import Subplot
import seaborn as sns

accuracies = [ 
"./log_vehicle/qffedavg_samp2_run1_q0_20_test.csv",
"./log_vehicle/qffedavg_samp2_run1_q5_20_test.csv",
"./log_vehicle/qffedavg_samp6_run1_q0_20_test.csv"
]

flag = "Testing"
dataset = ["Vehicle"]
label_ = ["q=0", "q>0", 'uniform']

def get_bar_y(filename, num_clients, num_bar=39):
    """
    Get bar chart.

    Args:
        filename: (str): write your description
        num_clients: (int): write your description
        num_bar: (int): write your description
    """


    num_runs = 1
    num_user = num_clients
    clients = np.zeros((num_runs, num_bar))
    accuracies = np.zeros((num_runs, num_clients))
    edges = np.linspace(0, 1, num_bar, endpoint=True)
    idx = 0
    for line in open(filename, 'r'):
        accu = float(line.strip())
        accuracies[idx/num_clients][idx % num_clients] = accu
        for j in range(num_bar):
            if accu > edges[j] and accu <= edges[j+1]:
                clients[idx/num_clients][j] += 1
        idx += 1

    mean_clients = np.mean(clients, axis=0)
    std_clients = np.std(clients, axis=0)


    
    num_good = int(num_user * 0.1)
    num_bad = num_good
    worst = np.zeros(num_runs)
    best = np.zeros(num_runs)
    variance = np.zeros(num_runs)

    for i in range(num_runs):
        worst[i] = np.mean(np.sort(accuracies[i])[:num_bad])
        best[i] = np.mean(np.sort(accuracies[i])[num_user-num_good:])
        variance[i] = np.var(accuracies[i]) * 10000

    avg_b = np.mean(worst)
    avg_g = np.mean(best)

    std_b = np.std(worst)
    std_g = np.std(best)

    avg_var = np.mean(variance)
    std_var = np.std(variance)

    print("############################################")

    print("file: {}\n, worst 10: {}, std: {}; best 10: {}, std: {}; variance: {}, std: {}".format(\
        filename, avg_b, std_b, avg_g, std_g, avg_var, std_var))

    print("############################################")

    mean_accuracies = np.mean(accuracies, axis=0)  

    return mean_clients, std_clients, mean_accuracies



def get_dis(filename):
    """
    Read a numpy array from a file

    Args:
        filename: (str): write your description
    """

    accuracies = []
    for line in open(filename, 'r'):
        accuracies.append(float(line.strip()))
    num_clients = len(accuracies)

    hist = np.asarray(accuracies)
    return hist

bws=[0.5, 0.6, 0.6]
ax2_y = [3.5, 3.5]
num_clients=[23]


f = plt.figure(figsize=[11, 4.5])

baseline = accuracies[0]
method = accuracies[1]
uniform = accuracies[2]


mean_y1, std_y1, mean_accu1 = get_bar_y(baseline, num_clients[0])
mean_y2, std_y2, mean_accu2 = get_bar_y(method, num_clients[0])
mean_y3, std_y3, mean_accu3 = get_bar_y(uniform, num_clients[0])

########################### q-FFL compared with fedavg (q=0) ###############################
ax1 = plt.subplot(1, 2, 1)
binEdges = np.linspace(0, 1, 40, endpoint=True)
bincenters = np.zeros(39)
for i in range(len(bincenters)):
    bincenters[i] = (binEdges[i] + binEdges[i+1]) * 0.5
width = bincenters[1] - bincenters[0]
plt.bar(bincenters, mean_y1, width=width, color='#17becf',  alpha=0.4, label = label_[0])
plt.bar(bincenters, mean_y2, width=width, color='#d62728', alpha=0.5, label = label_[1])


plt.xlabel(flag + " accuracy", fontsize=22)
plt.ylabel("# Clients", fontsize=22)
plt.title(dataset[0], fontsize=22, fontweight='bold')

plt.legend(frameon=False, loc=2)
ax2 = ax1.twinx()
ax2.set_ylim(0, ax2_y[0])

ax2.get_yaxis().set_visible(False)
sns.kdeplot(mean_accu1, linestyle='--', ax=ax2, bw=bws[0], color="#17becf")
sns.kdeplot(mean_accu2, ax=ax2, bw=bws[1], color="#d62728")
plt.xlim(0,1)
plt.tight_layout()
plt.ylim(0, 8)


########################### q-FFL compared with uniform ###############################
ax1 = plt.subplot(1, 2, 2)
binEdges = np.linspace(0, 1, 40, endpoint=True)
bincenters = np.zeros(39)
for i in range(len(bincenters)):
    bincenters[i] = (binEdges[i] + binEdges[i+1]) * 0.5
width = bincenters[1] - bincenters[0]
plt.bar(bincenters, mean_y3, width=width, color='#17becf',  alpha=0.4, label = label_[2])
plt.bar(bincenters, mean_y2, width=width, color='#d62728', alpha=0.5, label = label_[1])


plt.xlabel(flag + " accuracy", fontsize=22)
plt.ylabel("# Clients", fontsize=22)
plt.title(dataset[0], fontsize=22, fontweight='bold')

plt.legend(frameon=False, loc=2)
ax2 = ax1.twinx()
ax2.set_ylim(0, ax2_y[1])

ax2.get_yaxis().set_visible(False)
sns.kdeplot(mean_accu1, linestyle='--', ax=ax2, bw=bws[2], color="#17becf")
sns.kdeplot(mean_accu2, ax=ax2, bw=bws[1], color="#d62728")
plt.xlim(0, 1)
plt.tight_layout()
plt.ylim(0, 8)

f.savefig("fairness_vehicle.pdf")



