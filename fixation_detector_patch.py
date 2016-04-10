# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
from pyglui import ui
from fixation_detector import Fixation_Detector_Dispersion_Duration 

# logging
import logging
logger = logging.getLogger(__name__)

class Fixation_Detector_Dispersion_Duration_Extended(Fixation_Detector_Dispersion_Duration):
    '''
        Extend Fixation_Detector_Dispersion_Duration to export multi sections.
    '''
    def __init__(self,g_pool,max_dispersion = 1.0,min_duration = 0.15,h_fov=64, v_fov=36,show_fixations = True):
        super(Fixation_Detector_Dispersion_Duration_Extended, self).__init__(g_pool,h_fov=h_fov, v_fov=v_fov)


    def init_gui(self):
        self.menu = ui.Scrolling_Menu('Fixation Detector (Patch)')
        self.g_pool.gui.append(self.menu)

        def set_h_fov(new_fov):
            self.h_fov = new_fov
            self.pix_per_degree = float(self.g_pool.capture.frame_size[0])/new_fov
            self.notify_all_delayed({'subject':'fixations_should_recalculate'},delay=1.)

        def set_v_fov(new_fov):
            self.v_fov = new_fov
            self.pix_per_degree = float(self.g_pool.capture.frame_size[1])/new_fov
            self.notify_all_delayed({'subject':'fixations_should_recalculate'},delay=1.)

        def set_duration(new_value):
            self.min_duration = new_value
            self.notify_all_delayed({'subject':'fixations_should_recalculate'},delay=1.)

        def set_dispersion(new_value):
            self.max_dispersion = new_value
            self.notify_all_delayed({'subject':'fixations_should_recalculate'},delay=1.)

        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text('This plugin detects fixations based on a dispersion threshold in terms of degrees of visual angle. It also uses a min duration threshold.'))
        self.menu.append(ui.Info_Text("Press the export button or type 'e' to start the export."))
        self.menu.append(ui.Slider('min_duration',self,min=0.0,step=0.05,max=1.0,label='duration threshold',setter=set_duration))
        self.menu.append(ui.Slider('max_dispersion',self,min=0.0,step=0.05,max=3.0,label='dispersion threshold',setter=set_dispersion))
        self.menu.append(ui.Switch('show_fixations',self,label='Show fixations'))
        self.menu.append(ui.Slider('h_fov',self,min=5,step=1,max=180,label='horizontal FOV of scene camera',setter=set_h_fov))
        self.menu.append(ui.Slider('v_fov',self,min=5,step=1,max=180,label='vertical FOV of scene camera',setter=set_v_fov))

        self.menu.append(ui.Info_Text('Export fixations.'))
        self.menu.append(ui.Button('Export current section',self.export_fixations))
        self.menu.append(ui.Button('Export all sections',self.export_all_sections))

    def export_all_sections(self):
        for section in self.g_pool.trim_marks.sections:
            self.g_pool.trim_marks.focus = self.g_pool.trim_marks.sections.index(section)
            self.export_fixations()

del Fixation_Detector_Dispersion_Duration