# -*- coding: utf-8 -*-
'''
  Copyright (C) 2015 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

clusters = np.load('/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/precision_report/data_ordered_by_trial.npy')
i = 0
all_gaze = []
for cluster in clusters:
    by_trial = []
    for gaze in cluster:
        by_trial.append(gaze['norm_pos'])

    by_trial = np.vstack(by_trial)
    MX = np.mean(by_trial[:,0])
    MY = np.mean(by_trial[:,1])
    by_trial[:,0] = MX - by_trial[:,0]
    by_trial[:,1] = MY - by_trial[:,1]

    all_gaze.append(by_trial)

im = fig.add_axes([.08, .08, .85, (1/1.67539267016)], axisbg=(.2, .2, .2, .3))
im.spines['top'].set_visible(False)
im.spines['bottom'].set_visible(False)
im.spines['left'].set_visible(False)
im.spines['right'].set_visible(False)
im.xaxis.set_ticks_position('none')

def load_standard_frame():
    plt.ylim(ymax=.2, ymin=-.2)
    plt.xlim(xmax=.2, xmin=-.2)


def updatefig(*args):
    global all_gaze,i
    if i < len(all_gaze):
        data = np.vstack(all_gaze[i])
        X = data[:,0]
        Y = data[:,1]
        im.plot(X, Y, '.')
        i+=1
    else:
        i=0
        im.clear()
        load_standard_frame()
    return im,
print 1280./764.
ani = animation.FuncAnimation(fig, updatefig, interval=200, blit=True)
load_standard_frame()
plt.show()