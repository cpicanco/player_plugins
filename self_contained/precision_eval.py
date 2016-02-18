# -*- coding: utf-8 -*-
'''
  Copyright (C) 2015 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
# precision is defined as the Standard Deviation of successive observations

# seccessive observations = every point inside the central 1 second temporal gap of 2 seconds recorded = cluster
# Standard Deviation assumes that the mean of each cluster is at zero.  

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import join
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

global width, height
width, height = 1280,764

# here we expect 24 clusters (2 responses- 4 angles - 3 distance)
# path = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/precision_report/data_ordered_by_metatag.npy'

# here we expect 96 clusters (2 responses- 4 angles - 3 distance - 4 - trials)
source = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/precision_report/'
paths = [
    join(source,'data_ordered_by_trial.npy'),
    join(source,'data_ordered_by_trial_no_time_window.npy')
]

def load_data(path):
    global all_gaze
    all_gaze = []
    clusters = np.load(path)
    sds = []
    for cluster in clusters:
        by_trial = []
        for gaze in cluster:
            by_trial.append(gaze['norm_pos'])

        by_trial = np.vstack(by_trial)
        MX = np.mean(by_trial[:,0])
        MY = np.mean(by_trial[:,1])
        by_trial[:,0] = MX - by_trial[:,0]
        by_trial[:,1] = MY - by_trial[:,1]

        sds.append(np.std(by_trial, ddof=1))
        all_gaze.append(by_trial)

    all_gaze = np.vstack(all_gaze)

    sd = np.std(all_gaze, ddof=0)
    return sd, sds, all_gaze

def custom_subplot(im, sd, sds):
    im.spines['top'].set_visible(False)
    im.spines['bottom'].set_visible(False)
    im.spines['left'].set_visible(False)
    im.spines['right'].set_visible(False)
    im.xaxis.set_ticks_position('none')

    im.plot(sds,color=(.0,.0,.0,1.),label='1s')
    im.plot([0,len(sds)], [sd, sd], color=(0.,0.,0.,.4))
    im.yaxis.set_ticks([min(sds), sd, max(sds)])
    #im.set_ylabel('Desvio padrão (min., média, max.)')
    return max(sds)

def show_points(g,s, denormalize=True, pixels_per_degree=None):
    figure = plt.figure()
    axes = figure.add_axes([.12, .10, .85, .85], axisbg=(.2, .2, .2, .3))
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    X = g[:,0]
    Y = g[:,1]
    if denormalize:
        X *= width
        Y *= height
        plt.ylim(ymax=height-(height/2), ymin=-(height/2))
        plt.xlim(xmax=width-(width/2), xmin=-(width/2))
    else:
        plt.ylim(ymax=(height-(height/2))/pixels_per_degree, ymin=-(height/2)/pixels_per_degree)
        plt.xlim(xmax=(width-(width/2))/pixels_per_degree, xmin=-(width/2)/pixels_per_degree)

    
    axes.plot(X, Y, '.', label=s)
    plt.legend()

figsize = (6, 6)
figure, axes = plt.subplots(2, 1, sharey=False, sharex=True, figsize=figsize)

sd1, sds1, g1 = load_data(paths[0])
custom_subplot(axes[0], sd1, sds1)

sd2, sds2, g2 = load_data(paths[1])
custom_subplot(axes[1], sd2, sds2)

axes[0].yaxis.set_ticks([min(sds1),sd1,max(sds2)])
axes[1].set_xlabel('Tentativas')

plt.xticks(xrange(0,len(sds2)+1,16)) 
plt.ylim(ymax=max(sds2), ymin = 0)
plt.xlim(xmax=96, xmin=0)

figure.tight_layout()

show_points(g1, '1s')
show_points(g2, '2s')

x1 = 14.7720863025
y1 = 8.81708901183
# 1.38052358382 86.6499134779

x2 = 15.3336085236
y2 = 9.15224758754
# 1.43300057679 83.4767626961

pixels_per_degree = np.sqrt((width**2)+(height**2))/np.sqrt((x1**2)+(y1**2))
X = g1[:,0]
Y = g1[:,1]
# X += abs(min(X))
# Y += abs(min(Y))
g1[:,0] = X/pixels_per_degree
g1[:,1] = Y/pixels_per_degree
show_points(g1, '1sB', False,pixels_per_degree)

print np.sqrt(np.mean(g1**2)), pixels_per_degree

plt.show()