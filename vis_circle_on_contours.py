# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# Hack from Pupil (v0.3.7.4 .. v0.4x):
# - plugin.py
# - vis_circle.py
# - circle_detector.py

from player_methods import transparent_circle
from plugin import Plugin
import numpy as np
import cv2

from vcc_methods import ellipses_from_findContours, get_cluster_hierarchy, ellipse_to_contour
from vcc_methods import PolygonTestRC, get_codes, get_512_colors
#from vcc_methods import find_edges, draw_contours

from pyglui import ui

from methods import denormalize

# pt_codes references
_POINT = 0
_CODE = 1

# channel constants
_CH_B = 0  
_CH_G = 1
_CH_R = 2
_CH_0 = 3

class Vis_Circle_On_Contours(Plugin):
    """
        if gaze_point  is outside all contours:
            draw "outside"  
        elif gaze_point  is inside contour_a only:
            draw circle a
        elif gaze_point  is inside contour_b only:
            draw circle b
        elif gaze_point  is (inside contour_a) and (inside contour_b):
            draw circle c
        elif ...:
            draw circle ...

    """
    def __init__(self, g_pool,radius=20,color=(0.0, 0.0, 1.0, 0.8),thickness=2,fill=True, epsilon = 0.007, delta_area_threshold=20, dist_threshold=20,threshold=255, menu_conf={'pos':(300,300),'size':(300,300),'collapsed':False}):
        super(Vis_Circle_On_Contours, self).__init__(g_pool)
        self.order = .8
        self.uniqueness = "unique"

        # initialize empty menu
        # and load menu configuration of last session
        self.menu = None
        self.menu_conf = menu_conf

        # provided color, default red
        self.r = color[0]
        self.g = color[1]
        self.b = color[2]
        self.a = color[3]

        # shared configs
        self.radius = radius
        self.thickness = thickness
        self.fill = fill

        # detector
        self.candraw = False
        self.expected_contours = 9
        self.ellipse_size = 2.0
        self.epsilon = epsilon
        self.show_edges = True
        self.dist_threshold = dist_threshold
        self.delta_area_threshold = delta_area_threshold
        self.threshold = threshold

        # hardcoded colors, but a gradient could be assigned somehow based on provided one
        # self.colors = [ (self.b, self.g, self.r, self.a),  # red 
        #            (1, 0, 0, self.a),       # blue
        #            (0, 1, 0, self.a),       # green
        #            (1, 1, 0, self.a),       # blue marine
        #            (0, 1, 1, self.a),       # yellow
        #            (1, 0, 1, self.a),       # purple
        #            (0, 0, 1, self.a),       # red
        #            (1, 1, 1, self.a),       # white
        #            (0.5, 0.5, 0.5, self.a)] # gray
        #
        # for 9 expected contours we need 512 color        ]
        self.colors = get_512_colors()


        self.codes = list(get_codes('-+', self.expected_contours))

        self.ColorDictionary = dict(zip(self.codes, self.colors))

        # overide some target colors
        self.ColorDictionary['+1-2-3-4-5-6-7-8-9'] = (1, 0, 0, self.a) # blue
        self.ColorDictionary['-1+2-3-4-5-6-7-8-9'] = (0, 1, 0, self.a) # green
        self.ColorDictionary['-1-2+3-4-5-6-7-8-9'] = (1, 1, 0, self.a) # blue marine
        self.ColorDictionary['-1-2-3+4-5-6-7-8-9'] = (0, 1, 1, self.a) # yellow
        self.ColorDictionary['-1-2-3-4+5-6-7-8-9'] = (1, 0, 1, self.a) # purple
        self.ColorDictionary['-1-2-3-4-5+6-7-8-9'] = (0, 0, 1, self.a) # red
        self.ColorDictionary['-1-2-3-4-5-6+7-8-9'] = (1, 1, 1, self.a) # white
        self.ColorDictionary['-1-2-3-4-5-6-7+8-9'] = (0.5, 0.5, 0.5, self.a)  # gray
        self.ColorDictionary['-1-2-3-4-5-6-7-8+9'] = (0.75, 0.2, 0.5, self.a)  # ?
        #self.ColorDictionary['+1'] = (230, 50, 230, 150)
        #self.ColorDictionary['-1'] = (0, 0, 0, 255) 

    def update(self,frame,events):
        # get image from frame       
        img = frame.img

        # set color1    
        color1 = map(lambda x: int(x * 255),(self.b, self.g, self.r, self.a))

        # cv2.THRESH_BINARY
        # cv2.THRESH_BINARY_INV
        # cv2.THRESH_TRUNC

        # find raw ellipses from cv2.findContours

        # the less the difference between ellipse area and source contour area are,
        # the better a fit between ellipse and source contour will be
        # delta_area_threshold gives the maximum allowed difference
        ellipses = []
        merge = []
        contained_contours = []

        ellipses, merge, contained_contours = ellipses_from_findContours(img,cv2_thresh_mode=cv2.THRESH_BINARY,delta_area_threshold=self.delta_area_threshold,threshold=self.threshold)
        
        alfa = self.ellipse_size

        if self.show_edges:
            #frame.img = cv2.merge(merge)
            #cv2.drawContours(frame.img, contained_contours,-1, (0,0,255))
            if ellipses:
                for ellipse in ellipses:
                            center = ( int(round( ellipse[0][0] )), int( round( ellipse[0][1] ))) 
                            axes = ( int( round( ellipse[1][0]/alfa )), int( round( ellipse[1][1]/alfa )))
                            angle = int( round(ellipse[2] ))
                            cv2.ellipse(img, center, axes, angle, startAngle=0, endAngle=359, color=color1, thickness=1, lineType=8, shift= 0)


        # we need denormalized points for point polygon tests    
        pts = [denormalize(pt['norm_pos'],frame.img.shape[:-1][::-1],flip_y=True) for pt in events.get('gaze_positions',[])]
           
        if ellipses:
            # get area of all ellipses
            ellipses_temp = [e[1][0]/2. * e[1][1]/2. * np.pi for e in ellipses]
            ellipses_temp.sort()

            # take the highest area as reference
            area_threshold = ellipses_temp[-1]
        
            # filtering by proportional area
            ellipses_temp = []
            for e in ellipses:
                a,b = e[1][0] / 2., e[1][1] / 2.
                ellipse_area = np.pi * a * b
                if (ellipse_area/area_threshold) < .10:
                    pass  
                else:
                    ellipses_temp.append(e)

            # cluster_hierarchy is ordenated by appearence order, from top left screen
            # it is a list of clustered ellipses
            cluster_hierarchy = []
            cluster_hierarchy = get_cluster_hierarchy(
                                    ellipses=ellipses_temp,
                                    dist_threshold=self.dist_threshold)
            # total_stm is expected to be the number of stimuli on screen
            # total_stm = len(cluster_hierarchy)

            # we need contours for point polygon tests, not ellipses
            stm_contours = []

            # cluster_set is the ellipse set associated with each stimulus on screen
            

            temp = list(cluster_hierarchy)
            for cluster_set in temp:
                #print len(cluster_set)
                if len(cluster_set) > 2:
                    cluster_hierarchy.append(cluster_hierarchy.pop(cluster_hierarchy.index(cluster_set)))

            for cluster_set in cluster_hierarchy:
                if len(cluster_set) > 0:
                    if True:
                        for ellipse in cluster_set:
                            center = ( int(round( ellipse[0][0] )), int( round( ellipse[0][1] ))) 
                            axes = ( int( round( ellipse[1][0]/alfa )), int( round( ellipse[1][1]/alfa )))
                            angle = int( round(ellipse[2] ))
                            cv2.ellipse(img, center, axes, angle, startAngle=0, endAngle=359, color=color1, thickness=1, lineType=8, shift= 0)

                    # use only the biggest (last) ellipse for reference
                    stm_contours.append(ellipse_to_contour(cluster_set[-1], alfa))

            #print stm_contours
            # pt_codes is a list tuples:
            # tuple((denormalized point as a float x, y coordenate), 'string code given by the PointPolygonTextEx function')
            # ex.: tuple([x, y], '+1-2')
            contour_count = 0
            pt_codes = []
            for pt in pts:
                contour_count = 0
                counter_code = ''
                for contour in stm_contours:
                    contour_count, counter_code = PolygonTestRC(contour, pt, contour_count, counter_code)
                # a single code for a single point
                pt_codes.append((pt, counter_code))
            #print pt_codes
        else:
            #print 'else'
            contour_count = 0
           
        # transparent circle parameters
        radius = self.radius
        if self.fill:
            thickness= -1
        else:
            thickness = self.thickness

        # each code specifies the color of each point
        # in accordance with the self.ColorDictionary
        if contour_count > 0:
            for x in xrange(len(pt_codes)):
                try:
                    #print pt_codes[x]
                    color = self.ColorDictionary[pt_codes[x][_CODE]]
                except KeyError, e:
                    #print e
                    color = map(lambda x: int(x * 255),(0, 0, 0, self.a))

                transparent_circle(
                            img,
                            pt_codes[x][_POINT],
                            radius = int(radius/2),
                            color = color,
                            thickness = thickness    )
        # do not find any contour        
        else:
            for pt in pts:
                transparent_circle(
                    frame.img,
                    pt,
                    radius = radius,
                    color = map(lambda x: int(x * 255),self.colors[-1]),
                    thickness = thickness    )
                cv2.putText(img, '?', (int(pt[0] -10),int(pt[1]) +10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, lineType = cv2.CV_AA )
    
    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Gaze Circles on Contours')
        # load the configuration of last session
        self.menu.configuration = self.menu_conf
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('Close', self.unset_alive))
        self.menu.append(ui.Info_Text('Circle Properties'))
        self.menu.append(ui.Slider('radius',self,min=1,step=1,max=100,label='Radius'))
        self.menu.append(ui.Slider('thickness',self,min=1,step=1,max=15,label='Stroke width'))
        self.menu.append(ui.Switch('fill',self,label='Fill'))

        self.menu.append(ui.Info_Text('Detector Properties'))
        self.menu.append(ui.Slider('ellipse_size',self,min=0,step=0.001,max=4,label='Ellipse Size'))
        self.menu.append(ui.Slider('epsilon',self,min=0,step=1,max=1000.,label='Epsilon'))
        self.menu.append(ui.Slider('dist_threshold',self,min=1,step=1,max=20000,label='Distance threshold'))
        self.menu.append(ui.Slider('delta_area_threshold',self,min=0,step=0.1,max=20,label='Area threshold'))
        self.menu.append(ui.Slider('threshold',self,min=0,step=1,max=255,label='Threshold'))
        self.menu.append(ui.Slider('expected_contours',self,min=1, step=1, max=32, label='Expected contours (not working yet)'))       
        self.menu.append(ui.Switch('show_edges',self,label='Show edges'))

        color_menu = ui.Growing_Menu('Colors')
        color_menu.collapsed = True
        
        color_menu.append(ui.Info_Text('Outside Color'))
        color_menu.append(ui.Slider('r',self,min=0.0,step=0.05,max=1.0,label='Red'))
        color_menu.append(ui.Slider('g',self,min=0.0,step=0.05,max=1.0,label='Green'))
        color_menu.append(ui.Slider('b',self,min=0.0,step=0.05,max=1.0,label='Blue'))
        color_menu.append(ui.Slider('a',self,min=0.0,step=0.05,max=1.0,label='Alpha'))

        self.menu.append(color_menu)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def gl_display(self):
        pass

    def get_init_dict(self):
        return {'radius':self.radius,
                'color':(self.r, self.g, self.b, self.a),
                'epsilon':self.epsilon,
                'dist_threshold':self.dist_threshold,
                'delta_area_threshold':self.delta_area_threshold,
                'threshold':self.threshold,
                'thickness':self.thickness,
                'fill':self.fill,
                'menu_conf':self.menu.configuration}

    def clone(self):
        return Vis_Circle_On_Contours(**self.get_init_dict())

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()


