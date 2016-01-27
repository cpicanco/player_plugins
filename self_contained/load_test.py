import os
import numpy as np

scapp_output = np.load(os.path.join(os.getcwd(),"scapp_output.npy"))
self_scapp_output = [[]]
for line in scapp_output:
    trial_no = line[0] 
    timestamp = line[1]
    event = line[2]

    i = int(trial_no)

    if i > len(self_scapp_output):
        self_scapp_output.append([])
    self_scapp_output[i - 1].append((timestamp, event))

print self_scapp_output