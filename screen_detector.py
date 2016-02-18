# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
# modified version of marker_detector

import cv2
import numpy as np

from pyglui import ui
from methods import normalize
from glfw import glfwGetCurrentContext, glfwGetCursorPos, glfwGetWindowSize

#logging
import logging
logger = logging.getLogger(__name__)

from marker_detector import Marker_Detector
from square_marker_detect import draw_markers,m_marker_to_screen
from screen_detector_methods import detect_markers_robust, detect_screens
from reference_surface import Reference_Surface

class Screen_Detector(Marker_Detector):
    """docstring
    """
    def __init__(self,g_pool,mode="Show markers and frames",min_marker_perimeter = 40):
        super(Screen_Detector, self).__init__(g_pool)
        for p in g_pool.plugins:
            if p.class_name == 'Marker_Detector':
                p.alive = False
                break

    def init_gui(self):
        if self.g_pool.app == "player":
            self.alive = False
            logger.error('For capture only.')
            return
        self.menu = ui.Growing_Menu('Screen Detector')
        self.g_pool.sidebar.append(self.menu)

        self.button = ui.Thumb('running',self,label='Track',hotkey='t')
        self.button.on_color[:] = (.1,.2,1.,.8)
        self.g_pool.quickbar.append(self.button)
        self.add_button = ui.Thumb('add_surface',setter=self.add_surface,getter=lambda:False,label='Add surface',hotkey='a')
        self.g_pool.quickbar.append(self.add_button)
        self.update_gui_markers()

    def update_gui_markers(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text('This plugin detects the outmost computer, projection, tv screen visible in the scene. You can define a surface using 1 visible screen within the world view by clicking *add surface*. You can edit the defined surface by selecting *Surface edit mode*.'))
        self.menu.append(ui.Switch('robust_detection',self,label='Robust detection'))
        self.menu.append(ui.Slider('min_marker_perimeter',self,step=1,min=10,max=80))
        self.menu.append(ui.Switch('locate_3d',self,label='3D localization'))
        self.menu.append(ui.Selector('mode',self,label="Mode",selection=['Show markers and frames','Show marker IDs', 'Surface edit mode'] ))
        self.menu.append(ui.Button("Add surface", lambda:self.add_surface('_'),))

        # disable locate_3d if camera intrinsics don't exist
        if self.camera_intrinsics is None:
            self.menu.elements[4].read_only = True

        for s in self.surfaces:
            idx = self.surfaces.index(s)

            s_menu = ui.Growing_Menu("Surface %s"%idx)
            s_menu.collapsed=True
            s_menu.append(ui.Text_Input('name',s,label='Name'))
            #     self._bar.add_var("%s_markers"%i,create_string_buffer(512), getter=s.atb_marker_status,group=str(i),label='found/registered markers' )
            s_menu.append(ui.Text_Input('x',s.real_world_size,'x_scale'))
            s_menu.append(ui.Text_Input('y',s.real_world_size,'y_scale'))
            s_menu.append(ui.Button('Open debug window',s.open_close_window))
            #closure to encapsulate idx
            def make_remove_s(i):
                return lambda: self.remove_surface(i)
            remove_s = make_remove_s(idx)
            s_menu.append(ui.Button('Remove',remove_s))
            self.menu.append(s_menu)

    def update(self,frame,events):
        self.img_shape = frame.height,frame.width,3

        if self.running:
            gray = frame.gray

            if self.robust_detection:
                self.markers = detect_markers_robust(gray,
                                                    grid_size = 5,
                                                    prev_markers=self.markers,
                                                    min_marker_perimeter=self.min_marker_perimeter,
                                                    aperture=self.aperture,
                                                    visualize=0,
                                                    true_detect_every_frame=3)
            else:
                self.markers = detect_screens(gray)
                                                #grid_size = 5,
                                                #min_marker_perimeter=self.min_marker_perimeter,
                                                #aperture=self.aperture,
                                                #visualize=0)


            if self.mode == "Show marker IDs":
                draw_markers(frame.img,self.markers)


        # locate surfaces
        for s in self.surfaces:
            s.locate(self.markers, self.locate_3d,self.camera_intrinsics)
            # if s.detected:
                # events.append({'type':'marker_ref_surface','name':s.name,'uid':s.uid,'m_to_screen':s.m_to_screen,'m_from_screen':s.m_from_screen, 'timestamp':frame.timestamp})

        if self.running:
            self.button.status_text = '%s/%s'%(len([s for s in self.surfaces if s.detected]),len(self.surfaces))
        else:
            self.button.status_text = 'tracking paused'

        # edit surfaces by user
        if self.mode == "Surface edit mode":
            window = glfwGetCurrentContext()
            pos = glfwGetCursorPos(window)
            pos = normalize(pos,glfwGetWindowSize(window),flip_y=True)
            for s,v_idx in self.edit_surfaces:
                if s.detected:
                    new_pos =  s.img_to_ref_surface(np.array(pos))
                    s.move_vertex(v_idx,new_pos)

        #map recent gaze onto detected surfaces used for pupil server
        for s in self.surfaces:
            if s.detected:
                s.gaze_on_srf = []
                for p in events.get('gaze_positions',[]):
                    gp_on_s = tuple(s.img_to_ref_surface(np.array(p['norm_pos'])))
                    p['realtime gaze on ' + s.name] = gp_on_s
                    s.gaze_on_srf.append(gp_on_s)

del Marker_Detector