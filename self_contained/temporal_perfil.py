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
from methods import all_responses,stimuli_onset

def load_data_from_path(path):
    beha_events_path = os.path.join(path, "behavioral_events.txt")
    if not os.path.isfile(beha_events_path):
        print beha_events_path
        raise IOError, "behavioral_events were not found."

    return np.genfromtxt(beha_events_path, delimiter="\t",missing_values=["NA"],
      filling_values=None,names=True, autostrip=True, dtype=None)

def plot_temporal_perfil(axis,onsets,timestamps):
  def is_inside(rangein, rangeout):
    inside = []
    for t in timestamps:
      if (t >= rangein) and (t <= rangeout):
        inside.append(t)

    return inside

  def get_rate_line(onsets,aLabel,plot=True):
    g_rate = []
    for begin, end in zip(onsets[0], onsets[1]):
      g_rate.append(len(is_inside(begin, end))/(end-begin))

    if plot:
      axis.plot(g_rate,color=aLabel.lower(),label=aLabel)

  # red line
  get_rate_line(onsets, 'Red')

  # deleting and reversing give us the blue one 
  del onsets[0][0]
  get_rate_line(list(reversed(onsets)), 'Blue')
  
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

    # rpath = '/home/pupil/_rafael/data_doc/006-Renan/2015-05-20/'
    # rpath = '/home/pupil/_rafael/data_doc/005-Marco/2015-05-19/'
    # rpath = '/home/pupil/_rafael/data_doc/005-Marco/2015-05-20/' # beautiful
    # rpath = '/home/pupil/_rafael/data_doc/007-Gabriel/2015-05-20/' # beautiful
    # rpath = '/home/pupil/_rafael/data_doc/010-Iguaracy/2015-05-25/' # beautiful
    rpath = '/home/pupil/_rafael/data_doc/011-Priscila/2015-05-26/raw_data_organized/'
    # rpath = '/home/pupil/_rafael/data_doc/013-Oziele/2015-05-26/'

    # rpath = '/home/pupil/_rafael/data_doc/014-Acsa/2015-05-26'
    paths = [
        os.path.join(rpath, "000"),
        os.path.join(rpath, "001"),
        os.path.join(rpath, "002")
        #os.path.join(rpath, "003")
    ]

    # global vars
    data = []
    ymax = []
    for path in paths:
        be = load_data_from_path(path)
        responses = all_responses(be)
        data.append((stimuli_onset(be), responses))
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
        (onsets, responses) = d
        plot_temporal_perfil(axarr[i], onsets, responses)
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