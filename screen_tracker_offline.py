# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
# modified version of offline_marker_detector

# from memory_profiler import profile
import sys, os
from pathlib import Path

base_dir = Path(__file__).parents[2]
print(base_dir)
sys.path.append(os.path.join(str(base_dir),'pupil_plugins_shared'))

import numpy as np

from file_methods import Persistent_Dict, save_object
from pyglui import ui
from pyglui.cygl.utils import *

from offline_surface_tracker import Offline_Surface_Tracker
from screen_tracker import Screen_Tracker
from offline_reference_surface import Offline_Reference_Surface

#logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class Global_Container(object):
    pass

# first will look into Offline_Surface_Tracker namespaces then Screen_Tracker and so on
class Screen_Tracker_Offline(Offline_Surface_Tracker,Screen_Tracker):
    """
    Special version of screen tracker for use with videofile source.
    It will search all frames in the world.avi file for screens.
     - self.cache is a list containing marker positions for each frame.
     - self.surfaces[i].cache is a list containing surface positions for each frame
    Both caches are build up explicitly by pressing buttons.
    The cache is also session persistent.
    See marker_tracker.py for more info on this marker tracker.
    """
    def __init__(self,*args, **kwargs):
        self.screen_x = 1280 
        self.screen_y = 764
        for name, value in kwargs.items():
            if name == 'screen_x':
                self.screen_x = value
            if name == 'screen_y':
                self.screen_y = value
        super().__init__(*args,"Show Markers and Surfaces",100,False,True)
        
    def load_surface_definitions_from_file(self):
        self.surface_definitions = Persistent_Dict(os.path.join(self.g_pool.rec_dir,'screen_definition'))
        if self.surface_definitions.get('offline_square_marker_surfaces',[]) != []:
            logger.debug("Found screen defined or copied in previous session.")
            self.surfaces = [Offline_Reference_Surface(self.g_pool,saved_definition=d) for d in self.surface_definitions.get('offline_square_marker_surfaces',[]) if isinstance(d,dict)]
        elif self.surface_definitions.get('realtime_square_marker_surfaces',[]) != []:
            logger.debug("Did not find any screen in player from earlier session. Loading surfaces defined during capture.")
            self.surfaces = [Offline_Reference_Surface(self.g_pool,saved_definition=d) for d in self.surface_definitions.get('realtime_square_marker_surfaces',[]) if isinstance(d,dict)]
        else:
            logger.debug("No screen found. Please define using GUI.")
            self.surfaces = []

    def init_ui(self):
        super().init_ui()
        self.menu.label = 'Screen Tracker (Offline)'
  
    def init_marker_cacher(self):
        pass

    def update_marker_cache(self):
        pass

    def close_marker_cacher(self):
        pass

    def seek_marker_cacher(self,idx):
        pass

    def update_cache_hack(self):
        from video_capture import File_Source, EndofVideoError, FileSeekError
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
            except EndofVideoError:
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

    def update_gui_markers(self):

        def close():
            self.alive = False

        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close', close))
        self.menu.append(ui.Info_Text('The offline screen tracker will look for a screen for each frame of the video. By default it uses surfaces defined in capture. You can change and add more surfaces here.'))

        self.menu.append(ui.Info_Text('Before starting, you must update the screen detector cache:'))
        self.menu.append(ui.Button("Update Cache", self.update_cache_hack))            

        self.menu.append(ui.Info_Text('Then you can add a screen. Move to a frame where the screen was detected (in blue) then press the add screen surface button.'))
        self.menu.append(ui.Button("Add screen surface",lambda:self.add_surface()))

        self.menu.append(ui.Info_Text("Press the export button to export data from the current section."))

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

    def remove_surface(self, i):
        remove_surface = self.surfaces[i]
        if remove_surface == self.marker_edit_surface:
            self.marker_edit_surface = None
        if remove_surface in self.edit_surfaces:
            self.edit_surfaces.remove(remove_surface)

        self.surfaces[i].cleanup()
        del self.surfaces[i]
        self.update_gui_markers()
        self.notify_all({'subject': 'surfaces_changed'})
        self.timeline.content_height -= self.timeline_line_height

    def add_surface(self):
        self.surfaces.append(Offline_Reference_Surface(self.g_pool))
        self.timeline.content_height += self.timeline_line_height

        self.surfaces[0].name = 'Screen'
        self.surfaces[0].real_world_size['x'] = self.screen_x
        self.surfaces[0].real_world_size['y'] = self.screen_y
        self.update_gui_markers()

    def get_init_dict(self):
        return {
            'screen_x':self.screen_x,
            'screen_y':self.screen_y
        }

    def on_notify(self,notification):
        if notification['subject'] == "should_export":
            self.recalculate()
            self.save_surface_statsics_to_file(notification['range'],notification['export_dir'])

del Screen_Tracker
del Offline_Surface_Tracker
