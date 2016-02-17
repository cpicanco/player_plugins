import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math

from glob import glob

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
axes.plot([img_surface.shape[1]/2,img_surface.shape[1]/2],[0,img_surface.shape[0]/2],color=(.5,.5,.5,.5))

axes.xaxis.set_ticks_position('none')
axes.yaxis.set_ticks_position('none') 
axes.spines['top'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.spines['left'].set_visible(False)
axes.spines['right'].set_visible(False)

plt.ylim(ymax = img_surface.shape[0], ymin = 0)
plt.xlim(xmax = img_surface.shape[1], xmin = 0)
plt.show()