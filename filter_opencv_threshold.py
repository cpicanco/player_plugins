'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from gl_utils import draw_gl_points_norm
from player_methods import transparent_circle
from plugin import Plugin
import numpy as np
import cv2

from pyglui import ui

from methods import denormalize

class Filter_Opencv_Threshold(Plugin):
    """
        Apply cv2.threshold in each channel
    """
    def __init__(self, g_pool, threshold=177, thresh_mode="BINARY", menu_conf={'pos':(300,300),'size':(300,300),'collapsed':False}):
        super(Filter_Opencv_Threshold, self).__init__(g_pool)
        self.order = .1
        self.uniqueness = "not_unique"

        # initialize empty menu
        # and load menu configuration of last session
        self.menu = None
        self.menu_conf = menu_conf

        # filter properties
        self.threshold = threshold
        self.thresh_mode = thresh_mode

    def update(self,frame,events):
        img = frame.img
        height = img.shape[0] 
        width = img.shape[1] 

        #blur = cv2.GaussianBlur(img,(5,5),0)
        blur = frame.img
        edges = []
        blue, green, red = 0, 1, 2
        
        # thresh_mode
        if self.thresh_mode == "BINARY":
            cv2_thresh_mode = cv2.THRESH_BINARY

        if self.thresh_mode == "BINARY_INV":
            cv2_thresh_mode = cv2.THRESH_BINARY_INV

        if self.thresh_mode == "TRUNC":
            cv2_thresh_mode = cv2.THRESH_TRUNC

        if self.thresh_mode == "TOZERO":
            cv2_thresh_mode = cv2.THRESH_TOZERO

        # apply the threshold to each channel 
        for channel in (blur[:,:,blue], blur[:,:,green], blur[:,:,red]):
          retval, edg = cv2.threshold(channel, self.threshold, 255, cv2_thresh_mode)
          edges.append(edg)

        # lets merge the channels again
        edges.append(np.zeros((height, width, 1), np.uint8))
        edges_edt = cv2.max(edges[blue], edges[green])
        edges_edt = cv2.max(edges_edt, edges[red])
        merge = [edges_edt, edges_edt, edges_edt]

        # lets check the result
        frame.img = cv2.merge(merge)
        

       # if self.fill:
       #     thickness = -1
       # else:
       #     thickness = self.thickness

       # pts = [denormalize(pt['norm_gaze'],frame.img.shape[:-1][::-1],flip_y=True) for pt in events['pupil_positions'] if pt['norm_gaze'] is not None]
       # for pt in pts:
       #     transparent_circle(frame.img, pt, radius=self.radius, color=(self.b, self.g, self.r, self.a), thickness=thickness)

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Threshold')
        # load the configuration of last session
        self.menu.configuration = self.menu_conf
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('remove',self.unset_alive))
        self.menu.append(ui.Info_Text('Filter Properties'))

        self.menu.append(ui.Info_Text('Detector Properties'))
        self.menu.append(ui.Selector('thresh_mode',self,label='Thresh Mode',selection=["BINARY","BINARY_INV", "TRUNC","TOZERO"] ))
        self.menu.append(ui.Slider('threshold',self,min=0,step=1,max=255,label='Threshold'))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def gl_display(self):
        pass

    def get_init_dict(self):
        return {'threshold':self.threshold, 'thresh_mode':self.thresh_mode, 'menu_conf':self.menu.configuration}

    def clone(self):
        return Filter_Opencv_Threshold(**self.get_init_dict())

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
