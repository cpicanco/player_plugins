# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import cv2
from pyglui import ui

from plugin import Plugin

class Filter_Opencv_Threshold(Plugin):
    """
        Apply cv2.threshold in each channel of the (world) frame.img
    """
    uniqueness = "not_unique"
    def __init__(self, g_pool, threshold=177, thresh_mode="BINARY"):
        super(Filter_Opencv_Threshold, self).__init__(g_pool)
        # run before all plugins
        self.order = .1

        # initialize empty menu
        self.menu = None

        # filter properties
        self.threshold = threshold
        self.thresh_mode = thresh_mode

    def update(self,frame,events):
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
        for i, channel in enumerate((frame.img[:,:,blue], frame.img[:,:,green], frame.img[:,:,red])):
          retval, edg = cv2.threshold(channel, self.threshold, 255, cv2_thresh_mode)
          frame.img[:,:,i] = edg

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Threshold')

        # add menu to the window
        self.g_pool.gui.append(self.menu)

        # append elements to the menu
        self.menu.append(ui.Button('remove',self.unset_alive))
        self.menu.append(ui.Info_Text('Filter Properties'))
        self.menu.append(ui.Selector('thresh_mode',self,label='Thresh Mode',selection=["BINARY","BINARY_INV", "TRUNC","TOZERO"] ))
        self.menu.append(ui.Slider('threshold',self,min=0,step=1,max=255,label='Threshold'))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        # persistent properties throughout sessions
        return {'threshold':self.threshold, 'thresh_mode':self.thresh_mode}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()