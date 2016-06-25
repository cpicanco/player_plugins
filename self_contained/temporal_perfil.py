# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

# plot timestamped events using a cumulative frequency graph

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

def load_data_from_path(path):
    timestamps1_path = os.path.join(path, "scapp_output.timestamps")
    timestamps2_path = os.path.join(path, "scapp_output.npy")

    if not os.path.isfile(timestamps1_path):
        raise IOError, "Source 1 were not found."

    if not os.path.isfile(timestamps2_path):
        raise IOError, "Source 2 were not found."

    print "responses:",timestamps1_path
    print "stimuli__:",timestamps2_path

    # load timestamps2, stimuli onset
    stimuli_by_trial = [[]]
    timestamps2 = np.load(timestamps2_path)
    for line in timestamps2:
        trial_no = line[0] 
        timestamp = line[1]
        event = line[2]

        i = int(trial_no)

        if i > len(stimuli_by_trial):
            stimuli_by_trial.append([])
        stimuli_by_trial[i - 1].append((timestamp, event))

    print timestamps2
    responses = []
    with open(timestamps1_path, 'r') as f:
        for line in f:
            (trial_no, timestamp, event) = literal_eval(line)
            if "R" in event:
                responses.append(float(timestamp))

    ymax = len(responses)
    print ymax, 'responses'
    return stimuli_by_trial, responses, # cumulative, # begin #, end

def standard_plot(axis):
    # vertical lines
    red_onset = []
    blu_onset = []
    for trial in stimuli_by_trial:
        value = float(trial[0][0])
        if trial[0][1] == '1':
            red_onset.append(value)

        if trial[0][1] == '2':
            blu_onset.append(value)

    r_rate = []
    for red, blue in zip(red_onset,blu_onset):
        r_inside = []
        for r in responses:
            if (r >= red) and (r <= blue):
                r_inside.append(r)

        r_rate.append(len(r_inside)/(blue-red))


    # the actual data
    axis.plot(r_rate,color='red',label='Red')

    del red_onset[0]

    r_rate = []
    for red, blue in zip(red_onset,blu_onset):
        r_inside = []
        for r in responses:
            if (r >= blue) and (r <= red):
                r_inside.append(r)

        r_rate.append(len(r_inside)/(red-blue))

    axis.plot(r_rate,color='blue',label='Blue')

    # remove frame
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['right'].set_visible(False)

    #remove ticks
    axis.xaxis.set_ticks_position('none')
    axis.yaxis.set_ticks_position('none') 


if __name__ == '__main__':

###########
# drawing #
###########

    # rpath = '/home/rafael/documents/doutorado/data_doc/006-Renan/2015-05-20/'
    # rpath = '/home/rafael/documents/doutorado/data_doc/005-Marco/2015-05-19/'
    # rpath = '/home/rafael/documents/doutorado/data_doc/005-Marco/2015-05-20/' # beautiful
    # rpath = '/home/rafael/documents/doutorado/data_doc/007-Gabriel/2015-05-20/' # beautiful
    # rpath = '/home/rafael/documents/doutorado/data_doc/010-Iguaracy/2015-05-25/' # beautiful
    # rpath = '/home/rafael/documents/doutorado/data_doc/011-Priscila/2015-05-26/'
    # rpath = '/home/rafael/documents/doutorado/data_doc/013-Oziele/2015-05-26/'

    rpath = '/home/pupil/_rafael/data_doc/007-Gabriel/2015-05-20/'
    paths = [
        os.path.join(rpath, "000"),
        os.path.join(rpath, "001"),
        os.path.join(rpath, "002")
    ]

    # global vars
    data = []
    ymax = []
    for path in paths:
        stimuli_by_trial, responses = load_data_from_path(path)
        data.append((stimuli_by_trial, responses))
        ymax.append(len(responses))

    ymax = np.amax(ymax)
    x_label = 'Time block'
    y_label = 'Response rate'
    title = 'Response rate by time block'

    n_plots = len(paths)
    if n_plots == 1:
        figsize = (6, 4)
    elif n_plots == 2:
        figsize = (11, 4)
    else:
        figsize = (16, 4)

    # figure.add_axes([0.1, 0.1, 0.8, 0.8], frameon = 0)
    figure, axarr = plt.subplots(1, n_plots, sharey=True, sharex=False, figsize=figsize) 
    figure.suptitle(title);
    figure.text(0.5, 0.02, x_label)
    #figure.text(0.014, 0.5, y_label, rotation='vertical',verticalalignment='center',horizontalalignment='right')

    for i, d in enumerate(data):
        (stimuli_by_trial, responses) = d
        standard_plot(axarr[i])
        #plt.xlim(xmax = 300)

    axarr[0].set_ylabel(y_label)
    axarr[0].legend(loc=(0.0,0.73))

    # axarr[1].set_xlabel(x_label)
    #plt.ylim(ymax = ymax, ymin = 0)

    figure.subplots_adjust(wspace=0.1,left=0.05, right=.98,bottom=0.1,top=0.92)
    #figure.tight_layout()
    plt.show()

    # for line in output:
    #     print line
    # plt.savefig('test.png', bbox_inches='tight')
    # np.save(os.path.join(output_path,"scapp_output"), output)