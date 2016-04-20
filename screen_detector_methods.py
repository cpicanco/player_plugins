# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
# hacked from square_marker_detector
from math import sqrt

import cv2
import numpy as np

from quad_segmentation import sortCorners

# def get_close_markers(markers,centroids=None, min_distance=20):
#     if centroids is None:
#         centroids = [m['centroid']for m in markers]
#     centroids = np.array(centroids)

#     ti = np.triu_indices(centroids.shape[0], 1)
#     def full_idx(i):
#         #get the pair from condensed matrix index
#         #defindend inline because ti changes every time
#         return np.array([ti[0][i], ti[1][i]])

#     #calculate pairwise distance, return dense distace matrix (upper triangle)
#     distances =  pdist(centroids,'euclidean')

#     close_pairs = np.where(distances<min_distance)
#     return full_idx(close_pairs)


# hacked detect_markers
# now it detects a single outmost contour/screen instead
# "screens" container is just for compatibility with the marker detector
def detect_screens(gray_img, draw_contours=False):
    edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, -2)

    contours, hierarchy = cv2.findContours(edges,
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
    if contours != None: 
        # keep only > thresh_area   
        contours = [c for c in contours if cv2.contourArea(c) > (20 * 2500)]
        
        if len(contours) > 0: 
            # epsilon is a precision parameter, here we use 10% of the arc
            epsilon = cv2.arcLength(contours[0], True)*0.1

            # find the volatile vertices of the contour
            aprox_contours = [cv2.approxPolyDP(contours[0], epsilon, True)]

            # we want all contours to be counter clockwise oriented, we use convex hull for this:
            # aprox_contours = [cv2.convexHull(c,clockwise=True) for c in aprox_contours if c.shape[0]==4]

            # a non convex quadrangle is not what we are looking for.
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
                screen = {'id':msg,'verts':r,'perimeter':cv2.arcLength(r,closed=True),'centroid':centroid,"frames_since_true_detection":0}
             
                if screens.has_key(screen['id']) and screens[screen['id']]['perimeter'] > screen['perimeter']:
                    pass
                else:
                    screens[screen['id']] = screen

    return screens.values()

#persistent vars for detect_markers_robust
# lk_params = dict( winSize  = (45, 45),
#                   maxLevel = 1,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# prev_img = None
# tick = 0

# def detect_markers_robust(gray_img,grid_size,prev_markers,min_marker_perimeter=40,aperture=11,visualize=False,true_detect_every_frame = 1):
#     #global prev_img

#     global tick
#     if not tick:
#         tick = true_detect_every_frame
#         new_markers = detect_screens(gray_img)
#     else:
#         new_markers = []
#     tick -=1


#     # if prev_img is not None and prev_markers:

#     #     new_ids = [m['id'] for m in new_markers]

#     #     #any old markers not found in the new list?
#     #     not_found = [m for m in prev_markers if m['id'] not in new_ids and m['id'] >=0]
#     #     if not_found:
#     #         prev_pts = np.array([m['centroid'] for m in not_found])
#     #         # new_pts, flow_found, err = cv2.calcOpticalFlowPyrLK(prev_img, gray_img,prev_pts,winSize=(100,100))
#     #         new_pts, flow_found, err = cv2.calcOpticalFlowPyrLK(prev_img, gray_img,prev_pts,minEigThreshold=0.01,**lk_params)
#     #         for pt,s,e,m in zip(new_pts,flow_found,err,not_found):
#     #             if s: #ho do we ensure that this is a good move?
#     #                 m['verts'] += pt-m['centroid'] #uniformly translate verts by optlical flow offset
#     #                 r_norm = m['verts']/np.float32((gray_img.shape[1],gray_img.shape[0]))
#     #                 r_norm[:,:,1] = 1-r_norm[:,:,1]
#     #                 m['verts_norm'] = r_norm
#     #                 m["frames_since_true_detection"] +=1
#     #             else:
#     #                 m["frames_since_true_detection"] =100


#     #     #cocatenating like this will favour older markers in the doublication deletion process
#     #     markers = [m for m in not_found if m["frames_since_true_detection"] < 10 ]+new_markers
#     #     if 1: #del double detected markers
#     #         min_distace = min_marker_perimeter/4.
#     #         if len(markers)>1:
#     #             remove = set()
#     #             close_markers = get_close_markers(markers,min_distance=min_distace)
#     #             for f,s in close_markers.T:
#     #                 #remove the markers further down in the list
#     #                 remove.add(s)
#     #             remove = list(remove)
#     #             remove.sort(reverse=True)
#     #             for i in remove:
#     #                 del markers[i]
#     # else:
#     markers = new_markers


#     #prev_img = gray_img.copy()
#     return markers

# def bench():
#     cap = cv2.VideoCapture('/home/rafael/greserved/pupil-o/recordings/2015_05_13/natan/world.mkv')
#     status,img = cap.read()
#     markers = []
#     while status:
#         markers = detect_markers_robust(img,5,markers,true_detect_every_frame=1)
#         status,img = cap.read()
#         if markers:
#             return

# if __name__ == '__main__':
#     import cProfile,subprocess,os
#     cProfile.runctx("bench()",{},locals(),"world.pstats")
#     loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
#     gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
#     subprocess.call("python "+gprof2dot_loc+" -f pstats world.pstats | dot -Tpng -o world_cpu_time.png", shell=True)
#     print "created  time graph for  process. Please check out the png next to this file"