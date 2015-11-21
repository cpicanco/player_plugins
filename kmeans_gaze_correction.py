# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# modified from manual_gaze_correction

import cv2
from copy import deepcopy
from plugin import Plugin
import numpy as np
from methods import normalize
from pyglui import ui
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# todo: test https://en.wikipedia.org/wiki/DBSCAN, or alternative clustering algorithms.

class KMeans_Gaze_Correction(Plugin):
    """
        Correct gaze by:
        -1) assuming equaly distributed k stimuli at a detected screen (in the world frame);
        0) screen center and gaze points are given from homographic trasformation 
        1) grouping gaze in blocks with arbitrary size;
        2) infering bias from the difference between the mean k-means clustering and the screen center
        3) correction is the screen center subtracted from bias
    """

    def __init__(self,g_pool):
        super(KMeans_Gaze_Correction, self).__init__(g_pool)
        self.g_pool = g_pool
        self.order = .3
        self.menu = None
        self.untouched_gaze_positions_by_frame = None
        self.bias_by_frame = None
        self.screen_detector = None
        for p in self.g_pool.plugins:
            if p.class_name == 'Offline_Screen_Detector':
                if p.alive:
                    self.screen_detector = p
                    break
        
        if not self.screen_detector:
            self.alive = False
            logger.error('Open the Offline Screen Detector.')
        else:
            self.gaze_correction()

    def _load_untouched_gaze(self):
        if self.untouched_gaze_positions_by_frame:
            self.g_pool.gaze_positions_by_frame = self.untouched_gaze_positions_by_frame
            self.notify_all({'subject':'gaze_positions_changed'})

    def _get_bias_by_frame(self):
        # load dependencies
        bias_along_blocks = None   
        if self.screen_detector.surfaces:
            s = self.screen_detector.surfaces[0]
            if s.output_data:
                try:
                    bias_along_blocks = self.screen_detector.surfaces[0].output_data['bias_along_blocks']
                except Exception, e:
                    logger.error('Bias data not found. Error:%s'%e) # this should never happen
            else:
                logger.error('Correction data not found (Did you press recalculate?).')    
        else:
            logger.error('Screen not found. Define a screen/surface first.')

        if not bias_along_blocks:
            self.alive = False
            return False
  
        # create an empty array with the same structure as gaze positions by frame
        bias_by_frame = [[] for frame in self.g_pool.gaze_positions_by_frame]
        for f in range(0,len(bias_by_frame)):
            bias_by_frame[f] = [[] for gaze in self.g_pool.gaze_positions_by_frame[f]]

        unbiased_gaze = s.output_data['unbiased_gaze']
        indexed_bias = [[] for gaze in unbiased_gaze]

        # fill with data
        for b in bias_along_blocks:
            for i,_ in enumerate(indexed_bias[b['block'][0]:b['block'][1]]):
                indexed_bias[i] = b['bias']

        for index, d in enumerate(unbiased_gaze):
            bias = indexed_bias[index]
            f = d['frame']
            i = d['i']
            bias_by_frame[f][i] = bias

        # we must have at least the first gaze to continue no matter what happend before this line
        if len(bias_by_frame[0][0]) < 2:
            bias_by_frame[0][0] = bias_along_blocks[0]['bias']

        # normalize and reverse homographic transformation
        x_size, y_size = s.real_world_size['x'], s.real_world_size['y']
        for f in range(len(self.g_pool.gaze_positions_by_frame)):
            for i in range(len(self.g_pool.gaze_positions_by_frame[f])):
                if len(bias_by_frame[f][i]) == 2:
                    bias_by_frame[f][i] = normalize(bias_by_frame[f][i], (x_size,y_size))
                    bias_by_frame[f][i] = s.ref_surface_to_img(np.array(bias_by_frame[f][i]))
                    last_bias = bias_by_frame[f][i]
                else:
                    # finally, we apply the bias correction to all gaze, not only to the filtered ones
                    bias_by_frame[f][i] = last_bias
        self.bias_by_frame = bias_by_frame
        print self.bias_by_frame[0:10]
        return True

    def _set_gaze_correction(self):
        def apply_correction(f, i):
            x_bias, y_bias = self.bias_by_frame[f][i]
            gaze_pos = self.untouched_gaze_positions_by_frame[f][i]['norm_pos']
            gaze_pos = gaze_pos[0]+(x_bias), gaze_pos[1]+(y_bias)
            self.g_pool.gaze_positions_by_frame[f][i]['norm_pos'] = gaze_pos

        not_corrected = 0
        for f in range(len(self.g_pool.gaze_positions_by_frame)):
            for i in range(len(self.g_pool.gaze_positions_by_frame[f])):
                if len(self.bias_by_frame[f][i]) == 2:
                    apply_correction(f, i)
                else:
                    not_corrected += 1

        if not_corrected > 0:
            logger.error('{0} gaze points were not corrected. This is bad.'.format(not_corrected))
        self.notify_all_delayed({'subject':'gaze_positions_changed'})

    def gaze_correction(self):
        self.untouched_gaze_positions_by_frame = deepcopy(self.g_pool.gaze_positions_by_frame)
        if self._get_bias_by_frame():
            self._set_gaze_correction()

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Gaze Correction (k-means)')
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('Close',self.unset_alive))
        self.menu.append(ui.Info_Text('This plugin applies to all gaze the bias found by the offline screen detector plugin.'))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def load_untouched_gaze_no_notify(self):
        if self.untouched_gaze_positions_by_frame:
            self.g_pool.gaze_positions_by_frame = self.untouched_gaze_positions_by_frame

    def unset_alive(self):
        self.alive = False

    #def get_init_dict(self):
    #    return {'x_offset':self.x_offset,'y_offset':self.y_offset}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self._load_untouched_gaze()
        self.deinit_gui()