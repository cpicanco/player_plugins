# -*- coding: utf-8 -*-
'''
  Copyright (C) 2015 Rafael Picanço.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# plot timestamped events using a cummulative frequency graph

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

try:
    timestamps1_path = sys.argv[0]
    timestamps2_path = sys.argv[1]
except IndexError, e:
    print "No arguments provided: 1) response timestamps 2) stimuli onset timestamps"
    print e, "using hardcoded paths"
    output_path = "/home/rafael/documents/doutorado/data_doc/005-Marco/2015-05-20/002"
    timestamps1_path = os.path.join(output_path, "Data_002.timestamps")
    timestamps2_path = os.path.join(output_path, "scapp_output.npy")

if not os.path.isfile(timestamps1_path):
    raise IOError, "Source 1 were not found."

if not os.path.isfile(timestamps2_path):
    raise IOError, "Source 2 were not found."

print "responses:",timestamps1_path
print "stimuli__:",timestamps2_path

# load timestamps2, stimuli onset
timestamps2 = np.load(timestamps2_path)
stimuli_by_trial = [[]]
for line in timestamps2:
    trial_no = line[0] 
    timestamp = line[1]
    event = line[2]

    i = int(trial_no)

    if i > len(stimuli_by_trial):
        stimuli_by_trial.append([])
    stimuli_by_trial[i - 1].append((timestamp, event))

# for trial in stimuli_by_trial:
#     for stimulus in trial:
#         print "timestamp", stimulus[0]
#         print "code", stimulus[1]

begin = stimuli_by_trial[0][0][0]
end = stimuli_by_trial[-1][-1][0] 

# load timestamps1, responses
responses = []
cummulative = []
i = 1
with open(timestamps1_path, 'r') as f:
    for line in f:
        (trial_no, timestamp, event) = literal_eval(line)
        if "R" in event:
            responses.append(float(timestamp)-float(begin))
            cummulative.append(i)
            i += 1

ymax = len(responses)
print "found",ymax,"responses"

###########
# drawing #
###########

figure = plt.figure()
axes = figure.add_axes([0.1, 0.1, 0.8, 0.8], frameon = 0)
axes.step(responses, cummulative,color='black')

# lets check the results
x_label = 'Time (s)'
y_label = 'Cumulative Responses'
title = 'Cumulative Response by Time'
axes.set_xlabel(x_label)
axes.set_ylabel(y_label)
axes.set_title(title);
# plt.plot((begin, begin),(0,len(responses)), 'r-')
# end = float(end)-float(begin)

for trial in stimuli_by_trial:
    xline = float(trial[0][0])-float(begin)
    if trial[0][1] == '1':
        axes.plot((xline, xline),(0,ymax), 'r:')
    if trial[0][1] == '2':
        axes.plot((xline, xline),(0,ymax), 'b:')

plt.ylim(ymax = ymax, ymin = 0)
axes.xaxis.set_ticks_position('none')
axes.yaxis.set_ticks_position('none') 
plt.show()


# for line in output:
#     print line
# np.save(os.path.join(output_path,"scapp_output"), output)