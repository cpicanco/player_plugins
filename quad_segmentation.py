# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
# Translated from:
# github.com/bsdnoobz/opencv-code/blob/master/quad-segmentation.cpp

import numpy as np

def computeIntersect(a, b):
    """
    a, b: HoughLinesP line [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = a[0], a[1], a[2], a[3] 
    x3, y3, x4, y4 = b[0], b[1], b[2], b[3]

    d = float(((x1 - x3) * (y3 - y4)) - ((y1 - y2) * (x3 -x4)))

    if  d != 0.0:
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        return np.array([x, y])
    else:
        return (-1, -1)

def sortCorners(corners, center):
    """
    corners : list of points 
    center : point
    """
    top = []
    bot = []

    for corner in corners:
        if corner[1] < center[1]:
            top.append(corner)
        else:
            bot.append(corner)

    corners = np.zeros(shape=(4,2))

    if (len(top) == 2) and (len(bot) == 2):
        # top left
        if top[0][0] > top[1][0]:
            tl = top[1]
        else:
            tl = top[0]

        # top right
        if top[0][0] > top[1][0]:
            tr = top[0]
        else:
            tr = top[1]

        # botton left
        if bot[0][0] > bot[1][0]:
            bl = bot[1]
        else:
            bl = bot[0]

        # botton right
        if bot[0][0] > bot[1][0]:
            br = bot[0]
        else:
            br = bot[1]

    # if (len(top) == 1) and (len(bot) == 3):
    #     # top left
    #     if top[0][0] > top[1][0]:
    #         tl = top[1]
    #     else:
    #         tl = top[0]

    #     # top right
    #     if top[0][0] > top[1][0]:
    #         tr = top[0]
    #     else:
    #         tr = top[1]

    #     # botton left
    #     if bot[0][0] > bot[1][0]:
    #         bl = bot[1]
    #     else:
    #         bl = bot[0]

    #     # botton right
    #     if bot[0][0] > bot[1][0]:
    #         br = bot[0]
    #     else:
    #         br = bot[1]        

    try:
        corners[0] = np.array(tl)
        corners[1] = np.array(tr)
        corners[2] = np.array(br)
        corners[3] = np.array(bl)
    except Exception as e:
        print(center,'\n')
        print(top,'\n')
        print(bot,'\n')

    return corners

if __name__ == "__main__":
    import cv2
    import numpy as np

    pts =[(676.5168457,  115.4397049),
          (913.91461182, 395.74905396),
          (585.34643555, 669.7074585),
          (352.597229,   395.49456787),
          (632.09378052,  394.0976963)]


    image = np.zeros((720,1280,3), np.uint8)
    def circles(points):
        for point in points:
            if points.index(point) == 0:
                color = (0,0,255)   
            else:
                color = (255,255,255) 
            cv2.circle(image,(int(point[0]),int(point[1])),5,color,-1)
    
    circles(pts)
    # [ 471.10463715  481.35826111] 

    # [array([ 419.14718628,  271.10623169], dtype=float32)] 

    # [array([ 683.03338623,  485.45056152], dtype=float32), 
    # array([ 530.01513672,  687.05786133], dtype=float32),
    # array([ 252.22283936,  481.81838989], dtype=float32)] 

    while True:
        cv2.imshow("image", image)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()