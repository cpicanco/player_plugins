# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import cv2
from copy import deepcopy
from plugin import Plugin
import numpy as np
from methods import normalize
from pyglui import ui
import logging
logger = logging.getLogger(__name__)

class KMeans_Gaze_Correction(Plugin):
    """
        Correct gaze by:
        -1) assuming equaly distributed k stimuli at a detected screen (in the world frame);
        0) screen center and gaze points is given from homographic trasformation 
        1) grouping gaze in blocks with arbitrary size;
        2) infering bias from the difference between the mean k-means clustering and the screen center
        3) corrections is the screen center subtracted from bias
    """

    def __init__(self,g_pool, k=2):
        super(Manual_Gaze_Correction, self).__init__(g_pool)
        #let the plugin work before most other plugins.
        self.order = .3
        self.menu = None

        self.untouched_gaze_positions_by_frame = deepcopy(self.g_pool.gaze_positions_by_frame)
        self._k = k
        self._set_offset()

    def _set_k(self, kvalue):
        self._k = kvalue
        self._set_offset()

    def _set_offset(self):
        # k = self._k
        # xmax = 1280
        # ymax = 768
        # xmin = 0
        # ymin = 0
        # screen_center = 
        pass

    def init_gui(self):
        logger.error('Not implemented yet.')
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Gaze Correction (k-means)')
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('Close',self.unset_alive))
        self.menu.append(ui.Info_Text('Screen width and height are one unit respectively.'))
        self.menu.append(ui.Slider('k',self,min=1,step=1,max=20,setter=self._set_k))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'x_offset':self.x_offset,'y_offset':self.y_offset}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.g_pool.gaze_positions_by_frame = self.untouched_gaze_positions_by_frame
        self.notify_all({'subject':'gaze_positions_changed'})
        self.deinit_gui()