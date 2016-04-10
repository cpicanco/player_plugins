# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

from glob import glob

import constants as K

# load image file as numpy array
folders = ['distance_0-640-329',
           'distance_0-695-329',
           'distance_0-751-329',
           'distance_45-624-368',
           'distance_45-663-407',
           'distance_45-703-447',
           'distance_90-585-384',
           'distance_90-585-439',
           'distance_90-585-495',
           'distance_135-467-447',
           'distance_135-507-407',
           'distance_135-546-368']

print K.PIXELS_PER_DEGREE * 1.3

figure, axes = plt.subplots()
for folder in folders:
  path = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/'
  path = os.path.join(path,folder)
  # print path
  img_surface = cv2.imread(glob(os.path.join(path,'surface*'))[0],0)

  # add alpha channel
  _, img_surface = cv2.threshold(img_surface, 177, 255, cv2.THRESH_BINARY)

  alpha = img_surface.copy()
  alpha[alpha == 0] = 150
  alpha[alpha == 255] = 0
  img_surface = cv2.cvtColor(img_surface, cv2.COLOR_GRAY2BGRA)
  img_surface[:,:,3] = alpha

  surface = plt.imshow(img_surface)

print img_surface.shape
# y
axes.plot([0,img_surface.shape[1]], [img_surface.shape[0]/2,img_surface.shape[0]/2],color=(.5,.5,.5,.5))

# x
axes.plot([img_surface.shape[1]/2,img_surface.shape[1]/2],[0,img_surface.shape[0]],color=(.5,.5,.5,.5))
axes.text(.55, .95,"~9.2", ha='center', va='center', transform=axes.transAxes)
axes.text(.95, .55,"~15.2", ha='center', va='center', transform=axes.transAxes)

#axes.xaxis.set_ticks_position('none')
axes.yaxis.set_ticks([(K.SCREEN_HEIGHT_PX/2)+(K.PIXELS_PER_DEGREE*1.3*x) for x in range(-2,3)])
axes.set_yticklabels([round(1.3*x,2) for x in range(0,5)])
axes.yaxis.set_ticks_position('none')
axes.set_ylabel("Altura(graus)")

axes.xaxis.set_ticks([(K.SCREEN_WIDTH_PX/2)+(K.PIXELS_PER_DEGREE*1.3*x) for x in range(-2,3)])
axes.set_xticklabels([round(1.3*x,2) for x in range(0,5)])
axes.xaxis.set_ticks_position('none')
axes.set_xlabel("Comprimento(graus)")

axes.spines['top'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.spines['left'].set_visible(False)
axes.spines['right'].set_visible(False)

# axes2 = axes.twinx()
# axes2.spines['top'].set_visible(False)
# axes2.spines['bottom'].set_visible(False)
# axes2.spines['left'].set_visible(False)
# axes2.spines['right'].set_visible(False)
# axes2.set_ylim(ymax = img_surface.shape[0], ymin = 0)
# axes2.set_xlim(xmax = img_surface.shape[1], xmin = 0)

figure.subplots_adjust(left=0.09, bottom=0.0, right=.97,top=1.)

plt.ylim(ymax = img_surface.shape[0], ymin = 0)
plt.xlim(xmax = img_surface.shape[1], xmin = 0)
plt.show()