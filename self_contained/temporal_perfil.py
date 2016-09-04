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
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from methods import all_responses,stimuli_onset

def load_data_from_path(path):
    beha_events_path = os.path.join(path, "behavioral_events.txt")
    if not os.path.isfile(beha_events_path):
        print beha_events_path
        raise IOError, "behavioral_events were not found."

    return np.genfromtxt(beha_events_path, delimiter="\t",missing_values=["NA"],
      filling_values=None,names=True, autostrip=True, dtype=None)

def is_inside(timestamps,rangein, rangeout):
  # inside = []
  # for t in timestamps:
  #   if (t >= rangein) and (t <= rangeout):
  #     inside.append(t)

  return [t for t in timestamps if (t >= rangein) and (t <= rangeout)]

def get_rate(time_pairwise,timestamps):
  return [len(is_inside(timestamps, begin, end))/(end-begin) for begin, end in time_pairwise]

def plot_temporal_perfil(axis,onsets,timestamps, onsets_style='colors', c1="red", c2="blue", doreversed=False, nsize=None):
  # def adjust():
  #   if doreversed:
  #     data = [-x for x in data]
  w = 0.2
  if 'colors' in onsets_style:
    # red
    
    data = get_rate(zip(onsets[0], onsets[1]),timestamps)
    N = len(data)
    if doreversed:
      data = [-x for x in data]
    R = range(1,N+1)
    # axis.plot(data, color=c1, label="During Red")
    axis.bar(R,data,w, color=c1, label="Durante Verm.")
    

    # removing the first element and reversing give us the blue one
    data = get_rate(zip(onsets[1], onsets[0][1:]), timestamps)
    N = len(data)
    R = range(1,N+1)
    if doreversed:
      data = [-x for x in data]
    # axis.plot(data, color=c2, label="During Blue")
    axis.bar([x+w for x in R],data,w, color=c2, label="Durante Azul")
    axis.set_xlim([1,len(R)+w*2.])


  elif 'pair' in onsets_style:
    data = get_rate(zip(onsets[0], onsets[1]),timestamps)
    N = len(data)
    R = range(N)
    if doreversed:
      data = [-x for x in data]
      
    var = w * nsize
    print var, N
    axis.bar([x+var for x in R],data,w, color=c1)
    axis.set_xlim([1,len(R)+w*var])

  elif 'positions' in onsets_style:
    # left
    data = get_rate(zip(onsets,onsets[1:]),timestamps[0])
    axis.plot(data, color="black", label="Left")

    # right
    data = get_rate(zip(onsets,onsets[1:]),timestamps[1])
    axis.plot(data, color="grey", label="Right")

  # remove white space at the end
  

  # remove frame
  axis.spines['top'].set_visible(False)
  axis.spines['bottom'].set_visible(False)
  axis.spines['left'].set_visible(False)
  #axis.spines['right'].set_visible(False)

  # axis.set_xticklabels([round(1.3*x,2) for x in range(0,5)])

  #remove ticks
  axis.xaxis.set_ticks_position('none')
  axis.yaxis.set_ticks_position('none')

def draw_single(src_dir, show=False):
  print os.path.dirname(src_dir)
  ID = os.path.basename(os.path.dirname(src_dir))
  paths = sorted(glob(os.path.join(src_dir,'0*')))
  # global vars
  data = []
  ymax = []
  for path in paths:

      be = load_data_from_path(path)
      responses = all_responses(be)
      data.append((stimuli_onset(be), responses))

      ymax.append(len(responses))

  ymax = np.amax(ymax)
  x_label = 'Ciclos'
  y_label = 'Taxa (respostas por segundo)'
  title = 'Particip. '+ID+': taxa de resp. durante as cores Verm. e Azul.'

  n_plots = len(paths)
  if n_plots == 1:
      figsize = (4, 4)
  elif n_plots == 2:
      figsize = (8, 4)
  elif n_plots == 3:
      figsize = (12, 4)
  else:
      figsize = (14, 4)

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
  # axarr[0].legend(loc='upper right')
  # axarr[1].set_xlabel(x_label)
  #plt.ylim(ymax = ymax, ymin = 0)
  #plt.text(-16,4,ID,fontsize=18)
  #figure.subplots_adjust(wspace=0.05,left=0.05, right=.98,bottom=0.1,top=0.92)
  figure.tight_layout()
  
  # save/plot figure
  if show:
    plt.show()

def draw_all(src, draw_func, output_name,save=True):
  source_dir = src
  inner_paths = [
      'P001/2015-05-19',
      'P001/2015-05-27',
      'P002/2015-05-19',
      'P002/2015-05-20',
      'P003/2015-05-20',
      'P004/2015-05-20',
      'P005/2015-05-19',
      'P006/2015-05-25',
      'P007/2015-05-25',
      'P008/2015-05-26',
      'P009/2015-05-26',
      'P010/2015-05-26'
  ]

  source_directories = [os.path.join(source_dir,s) for s in inner_paths]
  
  for src_dir in source_directories:
      draw_func(src_dir)
      output_path = os.path.join(src_dir,output_name)
      plt.savefig(output_path, bbox_inches='tight')

if __name__ == '__main__':

  ###########
  # drawing #
  ###########
  font = {'family' : 'serif',
          'size'   : 16}

  matplotlib.rc('font', **font)
  #matplotlib.rcParams.update({'font.size': 22})

  draw_all('/home/rafael/git/abpmc-2016/',draw_single,'taxa_de_respostas_ao_botao_A.png')

  #draw_single('/home/rafael/git/abpmc-2016/P006/2015-05-25', True)