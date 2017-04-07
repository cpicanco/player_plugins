# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016-2017 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
# hacked from square_marker_detector
from math import sqrt

import cv2
import numpy as np

from quad_segmentation import sortCorners

# hacked detect_markers
# now it detects a single outmost contour/screen instead
# "screens" container is just for compatibility with the marker detector
def detect_screens(gray_img, draw_contours=False):
    edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, -2)

    _, contours, hierarchy = cv2.findContours(edges,
                                    mode=cv2.RETR_TREE,
                                    method=cv2.CHAIN_APPROX_SIMPLE,offset=(0,0)) #TC89_KCOS
    
    if draw_contours:
        cv2.drawContours(gray_img, contours,-1, (0,0,0))
    
    # remove extra encapsulation
    hierarchy = hierarchy[0]
    contours = np.array(contours)

    # keep only contours                        with parents     and      children
    contours = contours[np.logical_and(hierarchy[:,3]>=0, hierarchy[:,2]>=0)]

    contours = np.array(contours)
    screens = {}
    if contours is not None: 
        # keep only > thresh_area   
        contours = [c for c in contours if cv2.contourArea(c) > (20 * 2500)]
        
        if len(contours) > 0: 
            # epsilon is a precision parameter, here we use 10% of the arc
            epsilon = cv2.arcLength(contours[0], True)*0.1

            # find the volatile vertices of the contour
            aprox_contours = [cv2.approxPolyDP(contours[0], epsilon, True)]

            # we want all contours to be counter clockwise oriented, we use convex hull for this:
            # aprox_contours = [cv2.convexHull(c,clockwise=True) for c in aprox_contours if c.shape[0]==4]

            # a convex quadrangle what we are looking for.
            rect_cand = [r for r in aprox_contours if r.shape[0]==4]

            # if draw_contours:
            #     cv2.drawContours(gray_img, rect_cand,-1, (0,0,0))

            # screens
            for r in rect_cand:
                r = np.float32(r)

                msg = 1

                # define the criteria to stop and refine the screen verts
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                cv2.cornerSubPix(gray_img,r,(3,3),(-1,-1),criteria)

                corners = np.array([r[0][0], r[1][0], r[2][0], r[3][0]])

                # we need the centroid of the screen
                # M = cv2.moments(corners.reshape(-1,1,2))
                # centroid = np.array([M['m10']/M['m00'], M['m01']/M['m00']])
                # print 'a', centroid

                centroid = corners.sum(axis=0, dtype='float64')*0.25
                centroid.shape = (2)
                # print 'b', centroid

                # do not force dtype, use system default instead
                # centroid = [0, 0]
                # for i in corners:
                #     centroid += i
                # centroid *= (1. / len(corners))
                # print 'c', centroid

                corners = sortCorners(corners, centroid)
                r[0][0], r[1][0], r[2][0], r[3][0] = corners[0], corners[1], corners[2], corners[3]

                r_norm = r/np.float32((gray_img.shape[1],gray_img.shape[0]))
                r_norm[:,:,1] = 1-r_norm[:,:,1]
                screen = {'id':msg,'verts':r,'perimeter':cv2.arcLength(r,closed=True),'centroid':centroid,"frames_since_true_detection":0,"id_confidence":1.}
                
                if screen['id'] in screens:
                    if screens[screen['id']]['perimeter'] > screen['perimeter']:
                        pass
                else:
                    screens[screen['id']] = screen

    return screens.values()