# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
# hacked from square_marker_detector
from math import sqrt

import cv2
import numpy as np

from square_marker_detect import get_close_markers
from quad_segmentation import sortCorners

# hacked detect_markers
# now it detects a single outmost contour/screen instead
# "screens" is just for compatibility with the marker detector
def detect_screens(gray_img,grid_size):
    _ , edges = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(edges,
                                    mode=cv2.RETR_TREE,
                                    method=cv2.CHAIN_APPROX_SIMPLE,offset=(0,0)) #TC89_KCOS
    contours = np.array(contours)
    screens = []
    if contours != None: 
        # keep only > thresh_area   
        contours = [c for c in contours if cv2.contourArea(c) > (20 * 2500)]
        
        if len(contours) > 0: 
            # epsilon is a precision parameter, here we use 10% of the arc
            epsilon = cv2.arcLength(contours[0], True)*0.1

            # find the volatile vertices of the contour
            aprox_contours = [cv2.approxPolyDP(contours[0], epsilon, True)];
            
            rect_cand = [r for r in aprox_contours if r.shape[0]==4]

            # screens 
            size = 10*grid_size
            #top left,bottom left, bottom right, top right in image
            mapped_space = np.array( ((0,0),(size,0),(size,size),(0,size)) ,dtype=np.float32).reshape(4,1,2)
            for r in rect_cand:
                r = np.float32(r)

                screen = 0, 1
                if screen is not None:
                    angle,msg = screen

                    # define the criteria to stop and refine the screen verts
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                    cv2.cornerSubPix(gray_img,r,(3,3),(-1,-1),criteria)

                    centroid = r.sum(axis=0)/4.
                    centroid.shape = (2)

                    corners = np.array([r[0][0], r[1][0], r[2][0], r[3][0]])

                    center = [0, 0]
                    for i in corners:
                        center += i
                    center *= (1. / len(corners))

                    corners = sortCorners(corners, center)
                    r[0][0], r[1][0], r[2][0], r[3][0] = corners[0], corners[1], corners[2], corners[3]

                    r_norm = r/np.float32((gray_img.shape[1],gray_img.shape[0]))
                    r_norm[:,:,1] = 1-r_norm[:,:,1]
                    screen = {'id':msg,'verts':r,'verts_norm':r_norm,'centroid':centroid,"frames_since_true_detection":0}

                    screens.append(screen)

    return screens

#persistent vars for detect_markers_robust
lk_params = dict( winSize  = (45, 45),
                  maxLevel = 1,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
prev_img = None
tick = 0

# hacked detect_markers_robust
# modified L97, everything else keeped for compatibility issues with existing code
def detect_markers_robust(gray_img,grid_size,prev_markers,min_marker_perimeter=40,aperture=11,visualize=False,true_detect_every_frame = 1):
    global prev_img
    global tick
    if not tick:
        tick = true_detect_every_frame
        new_markers = detect_screens(gray_img,grid_size) # ,min_marker_perimeter,aperture,visualize)
    else:
        new_markers = []
    tick -=1


    if prev_img is not None and prev_markers:

        new_ids = [m['id'] for m in new_markers]

        #any old markers not found in the new list?
        not_found = [m for m in prev_markers if m['id'] not in new_ids and m['id'] >=0]
        if not_found:
            prev_pts = np.array([m['centroid'] for m in not_found])
            # new_pts, flow_found, err = cv2.calcOpticalFlowPyrLK(prev_img, gray_img,prev_pts,winSize=(100,100))
            new_pts, flow_found, err = cv2.calcOpticalFlowPyrLK(prev_img, gray_img,prev_pts,minEigThreshold=0.01,**lk_params)
            for pt,s,e,m in zip(new_pts,flow_found,err,not_found):
                if s: #ho do we ensure that this is a good move?
                    m['verts'] += pt-m['centroid'] #uniformly translate verts by optlical flow offset
                    r_norm = m['verts']/np.float32((gray_img.shape[1],gray_img.shape[0]))
                    r_norm[:,:,1] = 1-r_norm[:,:,1]
                    m['verts_norm'] = r_norm
                    m["frames_since_true_detection"] +=1
                else:
                    m["frames_since_true_detection"] =100


        #cocatenating like this will favour older markers in the doublication deletion process
        markers = [m for m in not_found if m["frames_since_true_detection"] < 10 ]+new_markers
        if 1: #del double detected markers
            min_distace = min_marker_perimeter/4.
            if len(markers)>1:
                remove = set()
                close_markers = get_close_markers(markers,min_distance=min_distace)
                for f,s in close_markers.T:
                    #remove the markers further down in the list
                    remove.add(s)
                remove = list(remove)
                remove.sort(reverse=True)
                for i in remove:
                    del markers[i]
    else:
        markers = new_markers


    prev_img = gray_img.copy()
    return markers