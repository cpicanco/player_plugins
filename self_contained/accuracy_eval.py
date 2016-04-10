# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
# precision is defined as the Standard Deviation of successive observations

# seccessive observations = every point inside the central 1 second temporal gap of 2 seconds recorded = cluster
# Standard Deviation assumes that the mean of each cluster is at zero.  
import scipy.spatial.distance as distance

import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys  

import cv2
from glob import glob

import constants as K
from methods import normalized_to_pixel

reload(sys)  
sys.setdefaultencoding('utf8')

def load_scapp_report(scapp_report_path):
    """
       Source Header Names for Eye Orientation Study (eos) trials:

       ______X1 : left 1
       ______Y1 : top 1
       ______X2 : left 2
       ______Y2 : top 2

       ExpcResp : Expected response
            0 = no gap/false
            1 = gap/true

            angle0  = 0 = right
            angle0  = 1 = left
            
            angle45  = 0 = bottomright
            angle45  = 1 = topleft
            
            angle90  = 0 = bottom
            angle90  = 1 = top
            
            angle135 = 0 = bottomleft
            angle135 = 1 = topright

    """
    if os.path.isfile(scapp_report_path):
        try:
            scapp_report = np.genfromtxt(scapp_report_path,
                delimiter="\t", missing_values=["NA"], skip_header=6, skip_footer=1,
                filling_values=None, names=True, deletechars='_', autostrip=True,
                dtype=None)
            return scapp_report
        except ValueError, e:
            logger.warning("genfromtxt error")
    else:
        print "File not found: "+ scapp_report_path

# here we expect 24 clusters (2 responses- 4 angles - 3 distance)
# path = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/precision_report/data_ordered_by_metatag.npy'

# here we expect 96 clusters (2 responses- 4 angles - 3 distance - 4 - trials)
source = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/'
scapp_report = load_scapp_report(os.path.join(source,'scapp_report.data'))
paths = sorted(glob(os.path.join(source,'precision_report/data_ordered_by_trial*')))
# transform gaze positions from normalized to pixels than to degree

def load_data(path):
    def get_realXY_from_trial(trial):
        if scapp_report[trial]['ExpcResp'] == 0:
            realX = scapp_report[trial]['X2']+51
            realY = scapp_report[trial]['Y2']+55
        elif scapp_report[trial]['ExpcResp'] == 1:
            realX = scapp_report[trial]['X1']+51
            realY = scapp_report[trial]['Y1']+55
        else:
            raise "Missing Value on Report"
        return (K.SCREEN_WIDTH_PX-realX, realY)    
    
    realXY = [get_realXY_from_trial(i) for i in range(len(scapp_report))]

    by_trial = []
    all_gaze = []
    all_real = []
    mus = []
    clusters = np.load(path)
    for cluster in clusters:
        gaze_by_trial = []
        real_by_trial = []
        for gaze in cluster:
            g = gaze['norm_pos']
            r = get_realXY_from_trial(gaze['trial'])
            gaze_by_trial.append(g)
            real_by_trial.append(r)
            all_gaze.append(g)
            all_real.append(r)
        
        real_by_trial = np.vstack(real_by_trial)

        gaze_by_trial = np.vstack(gaze_by_trial)
        gaze_by_trial = normalized_to_pixel(gaze_by_trial)

        gmeanX = np.mean(gaze_by_trial[:,0])
        gmeanY = np.mean(gaze_by_trial[:,1])

        rmeanX = np.mean(real_by_trial[:,0])
        rmeanY = np.mean(real_by_trial[:,1])

        # intermediate accuracy defined as the mean 
        # difference of the real positions and the reported positions
        by_trial.append({'g':gaze_by_trial,'r':real_by_trial}) 
        mus.append(np.array([gmeanX,gmeanY,rmeanX,rmeanY]))

    # vertical stack
    realXY = np.vstack(set(realXY))
    all_gaze = np.vstack(all_gaze)
    all_real = np.vstack(all_real)

    all_gaze = normalized_to_pixel(all_gaze)

    meanX = np.mean(all_gaze[:,0])
    meanY = np.mean(all_gaze[:,1])

    MU = [meanX, meanY]
    return MU, mus, all_gaze, all_real, realXY, by_trial 

def get_data(axes, paths):
    data = [[[] for cols in range(3)] for rows in range(6)]
    letters = string.uppercase[:18]
    index = 0
    for i, axrow in enumerate(axes):
        for j, axcol in enumerate(axrow):   
            #print paths[index]
            MU, mus, all_gaze, all_real, realXY, gaze_by_trial = load_data(paths[index])
            data[i][j] = {'MU':MU,
                          'mus':mus,
                          'g':all_gaze,
                          'r':all_real,
                          'by_trial':gaze_by_trial,
                          'r24':realXY,
                          'l':letters[index]}
            index+=1        
            print data[i][j]['l'], j, i
    return data

def show_points(X,Y):
    figure = plt.figure()
    axes = figure.add_axes([.12, .10, .85, .85], axisbg=(.2, .2, .2, .3))
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['right'].set_visible(False)
    # axes.xaxis.set_ticks_position('none')
    # axes.yaxis.set_ticks_position('none')

    axes.plot(X, Y, '.')
    #plt.legend()

def custom_subplots(axes, data):
    container = []
    for i, axrow in enumerate(axes):
        for j, axcol in enumerate(axrow):
            axcol.spines['top'].set_visible(False)
            axcol.spines['bottom'].set_visible(False)
            #axcol.spines['left'].set_visible(False)
            axcol.spines['right'].set_visible(False)
            axcol.xaxis.set_ticks_position('none')
            axcol.yaxis.set_ticks_position('none')

            letter = data[i][j]['l']
            by_trial = data[i][j]['by_trial']
            

            # realXY = data[i][j]['r24']
            # container = []
            # for r in realXY:
            #     container.append({'real':tuple(r),'error':[]})

            # error_by_stimulus = []
            # for dc in container: # ofuscates
            #     for trial in by_trial:
            #         if dc['real'] == (trial['r'][0][0],trial['r'][0][1]):
            #             error = distance.cdist(np.vstack(trial['g']),np.vstack(trial['r'])).diagonal().copy()
            #             dc['error'].append(error)
            #     error_by_stimulus.append(np.mean(np.hstack(dc['error']))/K.PIXELS_PER_DEGREE)
            sd = []
            error_by_trial = []
            for trial in by_trial:
                error = distance.cdist(np.vstack(trial['g']),np.vstack(trial['r'])).diagonal().copy()
                error_by_trial.append(np.mean(error)/K.PIXELS_PER_DEGREE)
                sd.append(np.std(error)/K.PIXELS_PER_DEGREE)
            # for trial in by_trial: # pupil's implementation is faster
            #     error = []
            #     for g, r in zip(trial['g'],trial['r']):
            #         error.append(distance.euclidean(g, r))
            #     error_by_trial.append(np.mean(error)/K.PIXELS_PER_DEGREE) 

            error_global = []
            for trial in by_trial:
                error = distance.cdist(np.vstack(trial['g']),np.vstack(trial['r'])).diagonal().copy()
                error_global.append(error) 
            # for trial in by_trial: # pupil's implementation is faster
            #     for g, r in zip(trial['g'],trial['r']):
            #         error_global.append(distance.euclidean(g, r))

            error_global = np.hstack(error_global)
            error_global = np.mean(error_global)/K.PIXELS_PER_DEGREE

            # axcol.plot(error_by_stimulus, color=(.0,.0,.0,1.)) 
            axcol.errorbar(range(1,97),error_by_trial, sd,color=(.0,.0,.0,1.),linestyle='None', marker='o', markersize=1,capthick=0,ecolor=(.0,.0,.0,.4))                     
            axcol.text(0.05, .9,letter, ha='center', va='center', transform=axcol.transAxes)
            axcol.text(0.85, .9,'~%s°'%(round(error_global,2)), ha='center', va='center', transform=axcol.transAxes)

            #show_points(ex_by_trial/K.PIXELS_PER_DEGREE,ey_by_trial/K.PIXELS_PER_DEGREE)
    ax = axes[3][0]
    ax.set_ylabel('Distância euclidiana média (graus)')
    ax.yaxis.set_label_coords(-0.15, 1.)
    #ax.yaxis.set_ticks([0,round(yMAX, 2)])

    ax = axes[5][1]
    ax.set_xlabel('Tentativas')
    ax.xaxis.set_ticks([0,48,96])
    # ax.xaxis.set_label_coords(0.0, 0.0)

def custom_subplots_lines(axes, data):
    container = []
    for i, axrow in enumerate(axes):
        for j, axcol in enumerate(axrow):
            axcol.spines['top'].set_visible(False)
            #axcol.spines['bottom'].set_visible(False)
            #axcol.spines['left'].set_visible(False)
            axcol.spines['right'].set_visible(False)
            # axcol.xaxis.set_ticks_position('none')
            # axcol.yaxis.set_ticks_position('none')

            letter = data[i][j]['l']
            realXY = data[i][j]['r24']
            by_trial = data[i][j]['by_trial']

            container = []
            for r in realXY:
                container.append({'real':tuple(r),'gaze':[]})

            linesX = []
            linesY = []
            for c_r in container:
                for trial in by_trial:
                    if c_r['real'] == (trial['r'][0][0],trial['r'][0][1]):
                        c_r['gaze'].append(trial['g'])

                c_r['gaze'] = np.vstack(c_r['gaze'])

                linesX.append([c_r['real'][0],np.mean(c_r['gaze'][:,0]), np.nan])
                linesY.append([c_r['real'][1],np.mean(c_r['gaze'][:,1]), np.nan]) 
      
            linesX = np.hstack(linesX/K.PIXELS_PER_DEGREE)
            linesY = np.hstack(linesY/K.PIXELS_PER_DEGREE)

            # error_x = []
            # error_y = []
            # for g,r in zip(all_gaze,all_real):
            #     error_x.append([g[0],r[0],np.nan])
            #     error_y.append([g[1],r[1],np.nan])


            # error_y = np.hstack(error_y/K.PIXELS_PER_DEGREE)
            # error_x = np.hstack(error_x/K.PIXELS_PER_DEGREE)
            
            # axcol.plot(error_x, error_y, linewidth=0.5, color=(.0,.0,.0,.2))
            axcol.plot(realXY[:,0]/K.PIXELS_PER_DEGREE,realXY[:,1]/K.PIXELS_PER_DEGREE, '.',color=(.0,.0,.0,1.), markersize=1.8)           
            axcol.plot(linesX, linesY, color=(0.,.0,.0,.3))
            axcol.text(0.05, .9,letter, ha='center', va='center', transform=axcol.transAxes)
            #axcol.text(0.85, .9,'~%s°'%(round(sd,2)), ha='center', va='center', transform=axrow.transAxes)

            # axcol.plot([MU[0]/K.PIXELS_PER_DEGREE,((K.SCREEN_WIDTH_PX/2)+4)/K.PIXELS_PER_DEGREE], [MU[1]/K.PIXELS_PER_DEGREE,(K.SCREEN_HEIGHT_PX/2)/K.PIXELS_PER_DEGREE], color=(0.,.0,.0,1.))

            plt.ylim(ymax=K.SCREEN_HEIGHT_PX/K.PIXELS_PER_DEGREE, ymin=0)
            plt.xlim(xmax=K.SCREEN_WIDTH_PX/K.PIXELS_PER_DEGREE, xmin=0)    

# def stuff(realXY):
#     # load image file as numpy array
#     folders = ['distance_0-640-329',
#                'distance_0-695-329',
#                'distance_0-751-329',
#                'distance_45-624-368',
#                'distance_45-663-407',
#                'distance_45-703-447',
#                'distance_90-585-384',
#                'distance_90-585-439',
#                'distance_90-585-495',
#                'distance_135-467-447',
#                'distance_135-507-407',
#                'distance_135-546-368']

#     figure, axes = plt.subplots()
#     for folder in folders:
#       path = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/'
#       path = os.path.join(path,folder)
#       # print path
#       img_surface = cv2.imread(glob(os.path.join(path,'surface*'))[0],0)

#       # add alpha channel
#       _, img_surface = cv2.threshold(img_surface, 177, 255, cv2.THRESH_BINARY)

#       alpha = img_surface.copy()
#       alpha[alpha == 0] = 150
#       alpha[alpha == 255] = 0
#       img_surface = cv2.cvtColor(img_surface, cv2.COLOR_GRAY2BGRA)
#       img_surface[:,:,3] = alpha

#       surface = plt.imshow(img_surface)

#     print img_surface.shape
#     # y
#     axes.plot([0,img_surface.shape[1]], [img_surface.shape[0]/2,img_surface.shape[0]/2],color=(.5,.5,.5,.5))

#     # x
#     axes.plot([img_surface.shape[1]/2,img_surface.shape[1]/2],[0,img_surface.shape[0]/2],color=(.5,.5,.5,.5))

#     # visual feedback
#     axes.plot(realXY[:,0],realXY[:,1], 'r.')
#     #axes.xaxis.set_ticks_position('none')
#     axes.yaxis.set_ticks([0, 764])
#     axes.yaxis.set_ticks_position('none')
#     axes.xaxis.set_ticks([0, 1280]) 
#     axes.spines['top'].set_visible(False)
#     axes.spines['bottom'].set_visible(False)
#     axes.spines['left'].set_visible(False)
#     axes.spines['right'].set_visible(False)

#     figure.subplots_adjust(wspace=0.1,left=0.06, right=.95,bottom=0.0,top=0.99)

#     plt.ylim(ymax = img_surface.shape[0], ymin = 0)
#     plt.xlim(xmax = img_surface.shape[1], xmin = 0)

def main():  
    figsize = (8, 10)
    figure, axes = plt.subplots(6, 3, sharey=True, sharex=True, figsize=figsize)

    data = get_data(axes, paths)
    custom_subplots(axes, data)
    figure.subplots_adjust(wspace=0.07, hspace=0.14,left=0.08, bottom=0.07, right=0.96,top=0.97)


    figure, axes = plt.subplots(6, 3, sharey=True, sharex=True, figsize=figsize)
    custom_subplots_lines(axes, data)
    # figure.tight_layout()
    figure.subplots_adjust(wspace=0.015, hspace=0.03,left=0.04, bottom=0.04, right=0.98,top=0.98)

    #stuff(realXY)
    plt.show()


if __name__ == '__main__':
    main()

    