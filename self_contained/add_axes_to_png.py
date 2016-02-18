# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
# http://matplotlib.org/users/image_tutorial.html
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
from colormaps import viridis
from glob import glob

def get_visual_angle(xaxis=True):
  # http://en.wikipedia.org/wiki/Visual_angle
  # distances in cm
  # object size
  S = 70.0
  # how far distance
  D = 260.0
  # visual angle
  V = 2 * math.atan( S/(D*2) )
  # print 'Radians:', V
  degrees = math.degrees(V)
  if xaxis:
    return degrees
  else:
    return (764*degrees)/1280
    

# load image file as numpy array
path = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/'
# single 1.3
folder = 'distance_0-640-329'  #2.6
#folder = 'distance_0-695-329' #3.8
#folder = 'distance_0-751-329' #5.1

path = os.path.join(path,folder)
img_heatmap = mpimg.imread(glob(os.path.join(path,'heatmap*'))[0])
img_gaze_cloud = mpimg.imread(glob(os.path.join(path,'gaze_correction*'))[0])
img_surface = cv2.imread(glob(os.path.join(path,'surface*'))[0],0)

# add alpha channel
img_surface = cv2.cvtColor(img_surface, cv2.COLOR_GRAY2BGRA)
img_surface[:,:,3] = 170

# drawing
figure, axes = plt.subplots()
#figure.canvas.draw()

surface = plt.imshow(img_surface,cmap = viridis)
surface.set_alpha = 1.0
#heatmap = plt.imshow(img_heatmap)
gzcloud = plt.imshow(img_gaze_cloud)

# explicit assign the xy axis 
plt.xticks(np.arange(0, img_surface.shape[1]+1, (img_surface.shape[1]+1)/12))
plt.yticks(np.arange(0, img_surface.shape[0]+1, (img_surface.shape[0]+1)/7))

# x axis
labels = [item.get_text() for item in axes.get_xticklabels()]
angle = get_visual_angle()
print 'x:',angle
factor = (angle+1.5)/len(labels)
tlabels = np.arange(0,angle+factor,(angle+factor)/len(labels))
for i in range(0, len(labels)):
  print tlabels[i]
  labels[i] = round(tlabels[i], 1)
axes.set_xticklabels(labels)

# y axis
labels = [item.get_text() for item in axes.get_yticklabels()]
angle = get_visual_angle(False)
print 'y:',angle
factor = (angle+1.5)/len(labels)
tlabels = np.arange(0,angle+factor,(angle+factor)/len(labels))
tlabels = list(reversed(tlabels))
for i in range(0, len(labels)):
  print tlabels[i]
  labels[i] = round(tlabels[i], 1)
axes.set_yticklabels(labels)

x_label = 'x (graus)'
y_label = 'y (graus)'
#title = 'Pontos de gaza sobre a superficie da Etapa B'
axes.set_xlabel(x_label)
axes.set_ylabel(y_label)
#axes.set_title(title);
axes.xaxis.set_ticks_position('none')
axes.yaxis.set_ticks_position('none') 
plt.show() 