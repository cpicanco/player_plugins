# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import cv2
import numpy as np

_RETR_TREE = 0

# Constants for the hierarchy[_RETR_TREE][contour][{next,back,child,parent}]
_ID_NEXT = 0
_ID_BACK = 1
_ID_CHILD = 2
_ID_PARENT = 3

# Channel constants
_CH_B = 0  
_CH_G = 1
_CH_R = 2
_CH_0 = 3

def find_edges(img, threshold, cv2_thresh_mode):
    blur = cv2.GaussianBlur(img,(5,5),0)
    #gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    edges = []
    # channels = (blur[:,:,_CH_B], blur[:,:,_CH_G], blur[:,:,_CH_R])
    # channels =  cv2.split(blur)
    for gray in (blur[:,:,_CH_B], blur[:,:,_CH_G], blur[:,:,_CH_R]):
        if threshold == 0:
            edg = cv2.Canny(gray, 0, 50, apertureSize = 5)
            edg = cv2.dilate(edg, None)
            edges.append(edg)
        else:
            retval, edg = cv2.threshold(gray, threshold, 255, cv2_thresh_mode)
            edges.append(edg)
    return edges

def ellipses_from_findContours(img, cv2_thresh_mode, delta_area_threshold, threshold, mode=True):
    candidate_ellipses = []
    debug_contours_output = []
    merge = []
    if mode:
        height = img.shape[0] 
        width = img.shape[1]
        edges = find_edges(img, threshold, cv2.THRESH_TOZERO)
        edges.append(np.zeros((height, width, 1), np.uint8))
        edges_edt = cv2.max(edges[_CH_B], edges[_CH_G])
        edges_edt = cv2.max(edges_edt, edges[_CH_R])
        edges = cv2.merge([edges_edt, edges_edt, edges_edt])
        merge = [edges_edt, edges_edt, edges_edt]
        edges = cv2.cvtColor(edges,cv2.COLOR_BGR2GRAY)
    else:
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        edges = cv2.adaptiveThreshold(gray_img, 255,
                                    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    thresholdType = cv2_thresh_mode,
                                    blockSize = 5,
                                    C = -1)


    #f = open('/home/rafael/Downloads/pupil-3x/pupil_src/player/data2.txt', 'w')
    #f.write(str(edges))
    #f.close
    #raise

    # cv2.findContours supports only black and white images (8uC1 and 32sC1 image)
    contours, hierarchy = cv2.findContours(edges,mode = cv2.RETR_TREE,method = cv2.CHAIN_APPROX_NONE,offset = (0,0)) #TC89_KCOS

    # remove extra encapsulation
    if hierarchy != None:
        if mode:
            hierarchy = hierarchy[_RETR_TREE]

            # turn outmost list into array
            contours =  np.array(contours)
            # keep only contours                        with parents     and      children
            contained_contours = contours[np.logical_or(hierarchy[:, _ID_PARENT] >= 0, hierarchy[:,2] >= _ID_CHILD)]

            debug_contours_output = contained_contours

            if contained_contours != None:
                contained_contours = [c for c in contained_contours if len(c) >= 5]

            if contained_contours != None:    
                contained_contours = [c for c in contained_contours if cv2.contourArea(c) > 1000]

            if contained_contours != None:    
                ellipses = [cv2.fitEllipse(c) for c in contained_contours]
            
            if ellipses != None:
                # filter for ellipses that have similar area as the source contour
                for e,c in zip(ellipses, contained_contours):
                    a,b = e[1][0] / 2., e[1][1] / 2.
                    if abs(cv2.contourArea(c) - np.pi * a * b) < delta_area_threshold:
                        candidate_ellipses.append(e)

        else:
            hierarchy = hierarchy[_RETR_TREE]

            # turn outmost list into array
            contours =  np.array(contours)
            # keep only contours                        with parents     and      children
            contained_contours = contours[np.logical_and(hierarchy[:, _ID_PARENT] >= 0, hierarchy[:,2] >= _ID_CHILD)]

            debug_contours_output = contained_contours
            #debug_contours_output = contained_contours
            # need at least 5 points to fit ellipse
            contained_contours =  [c for c in contained_contours if len(c) >= 5]

            if contained_contours != None:    
                contained_contours = [c for c in contained_contours if cv2.contourArea(c) > 1000]

            ellipses = [cv2.fitEllipse(c) for c in contained_contours]
            candidate_ellipses = []
            # filter for ellipses that have similar area as the source contour
            for e,c in zip(ellipses, contained_contours):
                a,b = e[1][0] / 2., e[1][1] / 2.
                if abs(cv2.contourArea(c) - np.pi * a * b) < delta_area_threshold:
                    candidate_ellipses.append(e)

    return candidate_ellipses, merge, debug_contours_output


img_path = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13/export_images/frame_1758.png'
img = cv2.imread(img_path)

ellipses = []
merge = []
contained_contours = []
delta_area_threshold = 20
threshold = 180
ellipses, merge, contained_contours = ellipses_from_findContours(img, cv2_thresh_mode=cv2.THRESH_BINARY,
                                                                      delta_area_threshold=delta_area_threshold,
                                                                      threshold=threshold,
                                                                      mode=False)

alfa = 2

#img = cv2.merge(merge)
#cv2.drawContours(img, contained_contours,-1, (0,0,255))
if ellipses:
    for ellipse in ellipses:
                center = ( int(round( ellipse[0][0] )), int( round( ellipse[0][1] ))) 
                axes = ( int( round( ellipse[1][0]/alfa )), int( round( ellipse[1][1]/alfa )))
                angle = int( round(ellipse[2] ))
                cv2.ellipse(img, center, axes, angle, startAngle=0, endAngle=359, color=(255, 0, 0), thickness=1, lineType=8, shift= 0)

#cv2.namedWindow("output", cv2.CV_WINDOW_AUTOSIZE)
while True:
    cv2.imshow("input", img)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
cv2.destroyAllWindows()