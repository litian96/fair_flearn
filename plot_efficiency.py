import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axisartist.axislines import Subplot


matplotlib.rc('xtick', labelsize=17) 
matplotlib.rc('ytick', labelsize=17) 


def parse_log(file_name):
    rounds = []
    accu = []
    loss = []
    sim = []
    accu_train = []

    test_accu_1 = []
    test_accu_2 = []
    train_accu_1 = []
    train_accu_2 = []

    for line in open(file_name, 'r'):

        search_test_accu = re.search( r'At round (.*) testing accuracy: (.*)', line, re.M|re.I)
        if search_test_accu:
            rounds.append(int(search_test_accu.group(1)))
            accu.append(float(search_test_accu.group(2)))
            
        search_loss = re.search(r'At round (.*) training loss: (.*)', line, re.M|re.I)
        if search_loss:
            loss.append(float(search_loss.group(2)))

        search_loss = re.search(r'gradient difference: (.*)', line, re.M|re.I)
        if search_loss:
            sim.append(float(search_loss.group(1)))

    return rounds, loss, accu, accu_train



accuracies = [ 
"./log_vehicle/ffedavg_run1_q5",
"./log_vehicle/ffedsgd_run1_q5",
]

dataset = ["Vehicle"]


f = plt.figure(figsize=[5.5, 4.5])

sampling_rate=[1]


rounds0, losses0, test_accuracies0, train_accuracies0 = parse_log(accuracies[0])
rounds1, losses1, test_accuracies1, train_accuracies1 = parse_log(accuracies[1])

plt.plot(np.asarray(rounds0)[::sampling_rate[0]], np.asarray(test_accuracies0)[::sampling_rate[0]], linewidth=3.0, label=r'q-FedAvg', color="#d62728")
plt.plot(np.asarray(rounds1)[::sampling_rate[0]], np.asarray(test_accuracies1)[::sampling_rate[0]],  '--', linewidth=3.0, label=r'q-FedSGD')
    
plt.ylabel('Testing accuracy', fontsize=22)
plt.xlabel('# Rounds', fontsize=22)

plt.legend(loc='best', frameon=False)
plt.title(dataset[0], fontsize=22, fontweight='bold')

plt.xlim(0, 10)
plt.tight_layout()

f.savefig("efficiency_qffedavg.pdf")
