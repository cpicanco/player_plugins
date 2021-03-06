# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
# precision is defined as the Standard Deviation of successive observations

# seccessive observations = every point inside the temporal gaps along the 2 seconds recorded = cluster
# Standard Deviation assumes that the mean of each cluster is at zero.  

import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import join
import sys  

from glob import glob

import constants as K
from methods import normalized_to_pixel, pixel_to_degree, move_mean_to_zero, root_mean_square

reload(sys)  
sys.setdefaultencoding('utf8')

# here we expect 24 clusters (2 responses- 4 angles - 3 distance)
# path = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/precision_report/data_ordered_by_metatag.npy'

# here we expect 96 clusters (2 responses- 4 angles - 3 distance - 4 - trials)
source = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/precision_report/'
paths = sorted(glob(join(source,'data_ordered_by_trial*')))

def load_data(path):
    all_gaze = []
    clusters = np.load(path)
    sds = []
    for cluster in clusters:
        by_trial = []
        for gaze in cluster:
            by_trial.append(gaze['norm_pos'])

        by_trial = np.vstack(by_trial)
        by_trial = move_mean_to_zero(by_trial)
        by_trial = normalized_to_pixel(by_trial)
        #by_trial = normalized_to_degree(by_trial)
        #by_trial = normalized_to_pixel(by_trial)
        by_trial = pixel_to_degree(by_trial)

        # intermediate precision defined as the Standard Deviation of observations in the cluster/trial 
        sds.append(np.std(by_trial, ddof=1, dtype=np.float64))
        all_gaze.append(by_trial)

    # overall precision defined as the Standard Deviation of all observations
    all_gaze = np.vstack(all_gaze)
    
    sd = np.std(all_gaze, ddof=0, dtype=np.float64)
    print 'sd:',round(sd,3)

    # overall variance
    var = np.var(all_gaze,ddof=0, dtype=np.float64)

    # overall precision defined as the RMS of all observations (should be equal to SD when the mean = zero)
    rms = root_mean_square(all_gaze)
    print 'rms:',rms
    print ''

   
    return var, sd, sds, all_gaze

def show_points(g,s, dimension):
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
    if 'pixels' in dimension:
        plt.ylim(ymax=K.SCREEN_HEIGHT_PX-(K.SCREEN_HEIGHT_PX/2), ymin=-(K.SCREEN_HEIGHT_PX/2))
        plt.xlim(xmax=K.SCREEN_WIDTH_PX-(K.SCREEN_WIDTH_PX/2), xmin=-(K.SCREEN_WIDTH_PX/2))
    elif 'degrees' in dimension:
        plt.ylim(ymax=(K.SCREEN_HEIGHT_DEG-(K.SCREEN_HEIGHT_DEG/2)), ymin=-(K.SCREEN_HEIGHT_DEG/2))
        plt.xlim(xmax=(K.SCREEN_WIDTH_DEG-(K.SCREEN_WIDTH_DEG/2)), xmin=-(K.SCREEN_WIDTH_DEG/2))
    elif 'normalized' in dimension:
        plt.ylim(ymax=(1.-(1./2.)), ymin=-(1./2.))
        plt.xlim(xmax=(1.-(1./2.)), xmin=-(1./2.))
    axes.plot(X, Y, '.', label=s)
    plt.legend()

def get_data(axes, paths):
    data = [[[] for cols in range(3)] for rows in range(6)]
    letters = string.uppercase[:18]
    index = 0
    for j, axcol in enumerate(axes):
        for i, axrow in enumerate(axcol):   
            #print paths[index]
            var, sd, sds, g = load_data(paths[index])
            data[j][i] = {'var':var,'sd':sd, 'sds':sds, 'g':g, 'l':letters[index]}
            index+=1        
    return data

def custom_subplots(axes, data, show_p=None):
    container = []
    for j, axcol in enumerate(axes):
        for i, axrow in enumerate(axcol):
            container.append(max(data[j][i]['sds']))

    yMAX = max(container)
    print round(yMAX, 2)
    container = []
    for j, axcol in enumerate(axes):
        for i, axrow in enumerate(axcol):
            #print j, i
            axrow.spines['top'].set_visible(False)
            axrow.spines['bottom'].set_visible(False)
            #axrow.spines['left'].set_visible(False)
            axrow.spines['right'].set_visible(False)
            axrow.xaxis.set_ticks_position('none')
            axrow.yaxis.set_ticks_position('none')

            sd = data[j][i]['sd']
            sds = data[j][i]['sds']
            letter = data[j][i]['l']
            container.append(data[j][i]['g'])
            axrow.plot(sds,color=(.0,.0,.0,1.))
            axrow.plot([0,len(sds)], [sd, sd], color=(0.,0.,0.,.4))
            axrow.set_ylim(ymax=yMAX, ymin=0)
            axrow.set_xlim(xmax=96, xmin=0)
            axrow.text(0.05, .9,letter, ha='center', va='center', transform=axrow.transAxes)
            axrow.text(0.85, .9,'~%s°'%(round(sd,2)), ha='center', va='center', transform=axrow.transAxes)
    
    ax = axes[3][0]
    ax.set_ylabel('Desvio padrão (graus)')
    ax.yaxis.set_label_coords(-0.2, 1.)
    ax.yaxis.set_ticks([0,round(yMAX, 2)])

    ax = axes[5][1]
    ax.set_xlabel('Tentativas')
    ax.xaxis.set_ticks([0,48,96])
    # ax.xaxis.set_label_coords(0.0, 0.0)

    if show_p:
        for data in container:
            show_points(data, 't', show_p)

def custom_subplots_points(axes, data):
    container = []
    for j, axcol in enumerate(axes):
        for i, axrow in enumerate(axcol):
            #print j, i
            axrow.spines['top'].set_visible(False)
            #axrow.spines['bottom'].set_visible(False)
            #axrow.spines['left'].set_visible(False)
            axrow.spines['right'].set_visible(False)
            axrow.xaxis.set_ticks_position('none')
            axrow.yaxis.set_ticks_position('none')

            letter = data[j][i]['l']
            sd = data[j][i]['sd']
            g = data[j][i]['g']
            X = g[:,0]
            Y = g[:,1]
            axrow.plot([0,0], [K.SCREEN_HEIGHT_DEG-(K.SCREEN_HEIGHT_DEG/2),-K.SCREEN_HEIGHT_DEG/2], color=(0.,0.,0.,.1))
            axrow.plot([K.SCREEN_WIDTH_DEG-(K.SCREEN_WIDTH_DEG/2),-K.SCREEN_WIDTH_DEG/2],[0,0], color=(0.,0.,0.,.1))

            axrow.plot(X,Y,'k.', markersize=0.5)
            # axrow.yaxis.set_ticks([])
            # axrow.xaxis.set_ticks([])
            axrow.set_ylim(ymax=(K.SCREEN_HEIGHT_DEG-(K.SCREEN_HEIGHT_DEG/2)), ymin=-(K.SCREEN_HEIGHT_DEG/2))
            axrow.set_xlim(xmax=(K.SCREEN_WIDTH_DEG-(K.SCREEN_WIDTH_DEG/2)), xmin=-(K.SCREEN_WIDTH_DEG/2))
            axrow.text(0.95, .9,letter, ha='center', va='center', transform=axrow.transAxes)
            axrow.text(0.85, .6,'~%s°'%(round(sd,2)), ha='center', va='center', transform=axrow.transAxes)
            # if i == 0 and j == 0:
            #     axrow.text(0.09, .9,'%s°'%round(K.SCREEN_HEIGHT_DEG-(K.SCREEN_HEIGHT_DEG/2),1), ha='center', va='center', transform=axrow.transAxes)
            #     axrow.text(0.09, .1,'%s°'%round(-K.SCREEN_HEIGHT_DEG/2,1), ha='center', va='center', transform=axrow.transAxes)
                
            #     axrow.text(1., -.08,'%s°'%round(K.SCREEN_WIDTH_DEG-(K.SCREEN_WIDTH_DEG/2),1), ha='center', va='center', transform=axrow.transAxes)
            #     axrow.text(0.09, -.08,'%s°'%round(-K.SCREEN_WIDTH_DEG/2,1), ha='center', va='center', transform=axrow.transAxes)
               
def main():  
    figsize = (8, 10)
    figure, axes = plt.subplots(6, 3, sharey=True, sharex=True, figsize=figsize)

    data = get_data(axes, paths)
    custom_subplots(axes, data)
    figure.subplots_adjust(left=0.09, bottom=0.065, right=0.97,top=0.97,wspace=0.08, hspace=0.12)


    figure, axes = plt.subplots(6, 3, sharey=True, sharex=True, figsize=figsize)
    custom_subplots_points(axes, data)
    figure.subplots_adjust(left=0.04, bottom=0.04, right=0.97,top=0.97,wspace=0.03, hspace=0.07)


    plt.show()

if __name__ == '__main__':
    main()

# show_points(g1, '0.5 s')
# show_points(g2, '1.0 s')
# show_points(g3, '2.0 s')