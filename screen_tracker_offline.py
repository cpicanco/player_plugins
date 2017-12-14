# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
# modified version of offline_marker_detector

import os, platform
import sys

# from memory_profiler import profile

from pathlib import Path

base_dir = Path(__file__).parents[2]
print(base_dir)
sys.path.append(os.path.join(str(base_dir),'pupil_plugins_shared'))

import sys, os
import platform
import cv2
import numpy as np
import csv
import multiprocessing as mp

from ctypes import c_bool


from itertools import chain
from OpenGL.GL import *
from methods import normalize, denormalize
from file_methods import Persistent_Dict, save_object
from cache_list import Cache_List
from glfw import *
from pyglui import ui
from pyglui.cygl.utils import *
from math import sqrt
from square_marker_detect import draw_markers,m_marker_to_screen

from offline_surface_tracker import Offline_Surface_Tracker
from screen_tracker import Screen_Tracker
from offline_reference_surface_patch import Offline_Reference_Surface_Extended

#logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# first will look into Offline_Surface_Tracker namespaces then Screen_Tracker and so on

class Screen_Tracker_Offline(Offline_Surface_Tracker,Screen_Tracker):
    """
    Special version of screen tracker for use with videofile source.
    It uses a seperate process to search all frames in the world.avi file for markers.
     - self.cache is a list containing marker positions for each frame.
     - self.surfaces[i].cache is a list containing surface positions for each frame
    Both caches are build up over time. The marker cache is also session persistent.
    See marker_tracker.py for more info on this marker tracker.
    """
    def __init__(self,*args, **kwargs):
        self.mode = "Show Markers and Surfaces"
        self.matrix = None
        for name, value in kwargs.items():
            if name == 'matrix':
                self.matrix = value
            if name == 'mode':
                self.mode = value

        self.heatmap_blur = True
        self.heatmap_blur_gradation = 0.12
        self.heatmap_colormap = "viridis"
        self.gaze_correction_block_size = '1000'
        self.gaze_correction_min_confidence = 0.98
        self.gaze_correction_k = 2
        self.heatmap_use_kdata = False
        super().__init__(*args,self.mode,100,False,True)
        
    def load_surface_definitions_from_file(self):
        self.surface_definitions = Persistent_Dict(os.path.join(self.g_pool.rec_dir,'surface_definitions'))
        if self.surface_definitions.get('offline_square_marker_surfaces',[]) != []:
            logger.debug("Found ref surfaces defined or copied in previous session.")
            self.surfaces = [Offline_Reference_Surface_Extended(self.g_pool,saved_definition=d) for d in self.surface_definitions.get('offline_square_marker_surfaces',[]) if isinstance(d,dict)]
        elif self.surface_definitions.get('realtime_square_marker_surfaces',[]) != []:
            logger.debug("Did not find ref surfaces def created or used by the user in player from earlier session. Loading surfaces defined during capture.")
            self.surfaces = [Offline_Reference_Surface_Extended(self.g_pool,saved_definition=d) for d in self.surface_definitions.get('realtime_square_marker_surfaces',[]) if isinstance(d,dict)]
        else:
            logger.debug("No surface defs found. Please define using GUI.")
            self.surfaces = []

    def init_ui(self):
        super().init_ui()
        self.menu.label = 'Offline Screen Tracker'
  
    def init_marker_cacher(self):
        pass

    def update_marker_cache(self):
        pass

    def close_marker_cacher(self):
        pass

    def seek_marker_cacher(self,idx):
        pass

    def update_cache_hack(self):
        from screen_detector_cacher import Global_Container
        from video_capture import File_Source, EndofVideoFileError, FileSeekError
        from screen_detector_methods import detect_screens
        
        def put_in_cache(frame_idx,detected_screen):
            print(frame_idx)
            visited_list[frame_idx] = True
            self.cache.update(frame_idx,detected_screen)
            for s in self.surfaces:
                s.update_cache(self.cache,
                    min_marker_perimeter=self.min_marker_perimeter,
                    min_id_confidence=self.min_id_confidence,
                    idx=frame_idx)
            
        def next_unvisited_idx(frame_idx):
            try:
                visited = visited_list[frame_idx]
            except IndexError:
                visited = True # trigger search

            if not visited:
                next_unvisited = frame_idx
            else:
                # find next unvisited site in the future
                try:
                    next_unvisited = visited_list.index(False,frame_idx)
                except ValueError:
                    # any thing in the past?
                    try:
                        next_unvisited = visited_list.index(False,0,frame_idx)
                    except ValueError:
                        #no unvisited sites left. Done!
                        #logger.debug("Caching completed.")
                        next_unvisited = None
            return next_unvisited

        def handle_frame(next_frame):
            if next_frame != cap.get_frame_index():
                #we need to seek:
                #logger.debug("Seeking to Frame %s" %next_frame)
                try:
                    cap.seek_to_frame(next_frame)
                except FileSeekError:
                    put_in_cache(next_frame,[]) # we cannot look at the frame, report no detection
                    return
                #seeking invalidates prev markers for the detector
                # markers[:] = []
            
            try:
                frame = cap.get_frame()
            except EndofVideoFileError:
                put_in_cache(next_frame,[]) # we cannot look at the frame, report no detection
                return

            put_in_cache(frame.index,detect_screens(frame.gray))

        self.cacher_seek_idx = 0
        visited_list = [False for x in self.cache]
        # markers = []
        cap = File_Source(Global_Container(),self.g_pool.capture.source_path)

        for _ in self.cache:
            next_frame = cap.get_frame_index()
            if next_frame is None or next_frame >=len(self.cache):
                #we are done here:
                break
            else:
                handle_frame(next_frame)

    def matrix_segmentation(self):
        if not self.mode == 'Show Markers and Surfaces':
            logger.error('Please, select the "Show Markers and Surfaces" option at the Mode Selector.')
            return 
        screen_width = 1280
        screen_height = -768

        def move_srf_to_stm(s,p):
            """
            ######### #########
            # 0 . 1 # # lt.rt #
            # .   . # # .   . #
            # 3 . 2 # # lb.rb #
            # uv #### #########

            #########
            # 3 . 2 #
            # .   . #
            # 0 . 1 #
            # sv #### 
            """
            sw = 150./screen_width
            sh = 150./screen_height

            before = s.markers.values()[0].uv_coords
            #before = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
            after = before.copy() 
            after[0] = p
            after[1] = p + np.array([sw,0])
            after[2] = p + np.array([sw,sh])
            after[3] = p + np.array([0,sh])

            transform = cv2.getPerspectiveTransform(after,before)
            for m in s.markers.values():
                m.uv_coords = cv2.perspectiveTransform(m.uv_coords,transform)

        n = 3
        namei = 0
        for i in xrange(0,n):
            for j in xrange(0,n):
                namei += 1
                sname = 'S'+str(namei)
                for s in self.surfaces:
                    if s.name == sname:
                        move_srf_to_stm(s, self.matrix[i][j])

        for s in self.surfaces:        
            s.invalidate()
        
        self.update_gui_markers()
             
    def add_matrix_surfaces(self):
        if not self.mode == 'Show Markers and Surfaces':
            logger.error('Please, select the "Show Markers and Surfaces" option at the Mode Selector.')
            return 

        screen_width = 1280
        screen_height = -768
        def midpoint(v1, v2):
            return np.array([(v1[0]+v2[0])/2,(v1[1]+v2[1])/2])

        def get_m(s, n=3):
            def get_coord(index,midxy, y=False):
                if y:
                  rws = screen_height # must flip
                else:
                  rws = screen_width

                return ((250./rws)*index)+midxy-(((250./rws)*n)/2)+((100./rws)/2)
            rwsx = screen_width
            rwsy = screen_height
            lt = s.left_top
            rt = s.right_top
            lb = s.left_bottom
            rb = s.right_bottom
            m = [[[] for _ in xrange(0,n)] for _ in xrange(0,n)]
            for j in xrange(0,n):
                xt = get_coord(j,midpoint(lt,rt)[0])
                # yt = get_coord(j,midpoint(lt,rt)[1])
                # xb = get_coord(j,midpoint(lb,rb)[0])
                # yb = get_coord(j,midpoint(lb,rb)[1])
                for i in xrange(0,n):
                    yt = get_coord(i,midpoint(lt,rb)[1],True) 
                    # yt = get_coord(i,midpoint([xt,yt],[xb,yb])[1],True) 
                    m[i][j] = np.array([xt, yt])
            return m
 
        def create_surface(name):
            self.surfaces.append(Offline_Reference_Surface_Extended(self.g_pool))
            self.surfaces[-1].name = name
            self.surfaces[-1].real_world_size['x'] = 150
            self.surfaces[-1].real_world_size['y'] = 150
            # self.surfaces[-1].markers = markers

        n = 3
        for s in self.surfaces:
            if s.name == 'Screen':
                self.matrix = get_m(s,n)
                # markers = s.markers
        
        for i in xrange(0,n*n):
            create_surface('S'+str(i+1))

        for s in self.surfaces:        
            s.invalidate()
        
        self.update_gui_markers()
                  
    def screen_segmentation(self):
        """
        no standards here, uv_coords ordering differing from the surface vertice one.

        0 . 1
        .   .
        3 . 2
        uv 

        3 . 2
        .   .
        0 . 1
        sv

        """
        if not self.mode == 'Show Markers and Surfaces':
            logger.error('Please, select the "Show Markers and Surfaces" option at the Mode Selector.')
            return

        correcly_named = [False, False]
        for s in self.surfaces:
            if s.name == 'Left':
                correcly_named[0] = (s.name == 'Left')
            if s.name == 'Right':
                correcly_named[1] = (s.name == 'Right')

        if not (correcly_named[0] and correcly_named[1]):
            logger.error('Please, create two identical surfaces and name them as "Left" and "Right".')
            return
            
        for s in self.surfaces:
            s.real_world_size['x'] = s.real_world_size['x']/2.
            lt = s.left_top
            rt = s.right_top
            lb = s.left_bottom
            rb = s.right_bottom

            midtop = np.array([(lt[0]+rt[0])/2,(lt[1]+rt[1])/2])
            midbottom = np.array([(lb[0]+rb[0])/2,(lb[1]+rb[1])/2])

            if s.name == 'Left':
                s.right_top = midtop
                s.right_bottom = midbottom

            if s.name == 'Right':
                s.left_top = midtop
                s.left_bottom = midbottom

        self.update_gui_markers()

    def update_gui_markers(self):

        def close():
            self.alive = False

        def set_min_marker_perimeter(val):
            self.min_marker_perimeter = val
            self.notify_all_delayed({'subject':'min_marker_perimeter_changed'},delay=1)

        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',close))
        self.menu.append(ui.Info_Text('The offline screen tracker will look for a screen for each frame of the video. By default it uses surfaces defined in capture. You can change and add more surfaces here.'))
        self.menu.append(ui.Selector('mode',self,setter=self.set_mode,label='Mode',selection=["Show Markers and Surfaces"] ))
        
        if self.mode == 'Show Markers and Surfaces':
            self.menu.append(ui.Info_Text('Before starting, you must update the screen detector cache:'))
            self.menu.append(ui.Button("Update Cache", self.update_cache_hack))            

            self.menu.append(ui.Info_Text('Then you can add a screen. Move to a frame where the screen was detected (in blue) then press the add screen surface button.'))
            self.menu.append(ui.Button("Add screen surface",lambda:self.add_surface('_')))

        self.menu.append(ui.Info_Text("Press the export button to export data from the current section."))

        if self.mode == 'Show Kmeans Correction':
            self.menu.append(ui.Info_Text('Gaze Correction requires a non segmented screen. It requires k equally distributed stimuli on the screen.'))
            self.menu.append(ui.Text_Input('gaze_correction_block_size',self,label='Block Size'))
            self.menu.append(ui.Slider('gaze_correction_min_confidence',self,min=0.0,step=0.01,max=1.0,label='Minimun gaze confidence'))
            self.menu.append(ui.Slider('gaze_correction_k',self,min=1,step=1,max=24,label='K clusters'))

        if self.mode == 'Show Gaze Cloud':
            self.menu.append(ui.Slider('gaze_correction_min_confidence',self,min=0.0,step=0.01,max=1.0,label='Minimun gaze confidence'))
            self.menu.append(ui.Slider('gaze_correction_k',self,min=1,step=1,max=24,label='K clusters'))

        if self.mode == 'Show Heatmaps':
            self.menu.append(ui.Info_Text('Heatmap Settings'))
            self.menu.append(ui.Switch('heatmap_blur',self,label='Blur'))
            self.menu.append(ui.Slider('heatmap_blur_gradation',self,min=0.01,step=0.01,max=1.0,label='Blur Gradation'))
            self.menu.append(ui.Selector('heatmap_colormap',self,label='Color Map',selection=['magma', 'inferno', 'plasma', 'viridis', 'jet']))
            self.menu.append(ui.Switch('heatmap_use_kdata',self,label='Use K Data'))

        for s in self.surfaces:
            idx = self.surfaces.index(s)
            s_menu = ui.Growing_Menu("Surface %s"%idx)
            s_menu.collapsed=True
            s_menu.append(ui.Text_Input('name',s))
            s_menu.append(ui.Text_Input('x',s.real_world_size,label='X size'))
            s_menu.append(ui.Text_Input('y',s.real_world_size,label='Y size'))
            s_menu.append(ui.Button('Open Debug Window',s.open_close_window))
            #closure to encapsulate idx
            def make_remove_s(i):
                return lambda: self.remove_surface(i)
            remove_s = make_remove_s(idx)
            s_menu.append(ui.Button('remove',remove_s))
            self.menu.append(s_menu)

    def set_mode(self, value):
        self.mode = value
        self.update_gui_markers()

    def add_surface(self,_):
        self.surfaces.append(Offline_Reference_Surface_Extended(self.g_pool))

        self.surfaces[0].name = 'Screen'
        self.surfaces[0].real_world_size['x'] = 1280
        self.surfaces[0].real_world_size['y'] = 768

        # self.surfaces[0].name = 'Left'
        # self.surfaces[0].real_world_size['x'] = 1280
        # self.surfaces[0].real_world_size['y'] = 768

        # self.surfaces[1].name = 'Right'
        # self.surfaces[1].real_world_size['x'] = 1280
        # self.surfaces[1].real_world_size['y'] = 768

        self.update_gui_markers()

    def gl_display(self):
        """
        Display marker and surface info inside world screen
        """
        super().gl_display()

        if self.mode == "Show Gaze Cloud":
            for s in self.surfaces:
                s.gl_display_gaze_cloud()

        if self.mode == "Show Kmeans Correction":
            for s in self.surfaces:
                s.gl_display_gaze_correction()

        if self.mode == "Show Heatmap Correction":
            for s in self.surfaces:
                s.gl_display_mean_correction()

        if self.mode == "Show Mean Correction":
            for s in self.surfaces:
                s.gl_display_mean_correction()

    def get_init_dict(self):
        return {'mode':self.mode,
                'matrix':self.matrix}

    def on_notify(self,notification):
        if notification['subject'] == 'gaze_positions_changed':
            logger.info('Gaze positions changed. Please, recalculate.')
            #self.recalculate()
        if notification['subject'] == 'gaze_positions_changed':
            logger.info('Gaze postions changed. Please, recalculate..')
            #self.recalculate()
        elif notification['subject'] == 'surfaces_changed':
            logger.info('Surfaces changed. Please, recalculate..')
            #self.recalculate()
        elif notification['subject'] == 'min_marker_perimeter_changed':
            logger.info('min_marker.. not implemented')
            #logger.info('Min marper perimeter adjusted. Re-detecting surfaces.')
            #self.invalidate_surface_caches()
        elif notification['subject'] == "should_export":
            self.save_surface_statsics_to_file(notification['range'],notification['export_dir'])
            #logger.info('Min marper perimeter adjusted. Re-detecting surfaces.')
            #self.save_surface_statsics_to_file(notification['range'],notification['export_dir'])


del Screen_Tracker
del Offline_Surface_Tracker