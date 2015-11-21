# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
# modified version of offline_marker_detector

import os, platform
import cv2
import numpy as np
import csv

if platform.system() == 'Darwin':
    from billiard import Process,Queue,forking_enable
    from billiard.sharedctypes import Value
else:
    from multiprocessing import Process, Queue
    forking_enable = lambda x: x #dummy fn
    from multiprocessing.sharedctypes import Value
from ctypes import c_bool

from OpenGL.GL import *
from methods import normalize #,denormalize
from glfw import glfwGetCurrentContext,glfwGetWindowSize,glfwGetCursorPos

from pyglui import ui
from pyglui.cygl.utils import *

from file_methods import Persistent_Dict
from screen_detector import Screen_Detector
from offline_marker_detector import Offline_Marker_Detector
from square_marker_detect import draw_markers,m_marker_to_screen
from offline_reference_surface_patch import Offline_Reference_Surface_Extended

#logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# first will look into Offline_Marker_Detector namespaces then Screen_Detector and so on
class Offline_Screen_Detector(Offline_Marker_Detector,Screen_Detector):
    """
    Special version of screen detector for use with videofile source.
    It uses a seperate process to search all frames in the world.avi file for markers.
     - self.cache is a list containing marker positions for each frame.
     - self.surfaces[i].cache is a list containing surface positions for each frame
    Both caches are build up over time. The marker cache is also session persistent.
    See marker_tracker.py for more info on this marker tracker.
    """

    def __init__(self,g_pool,mode="Show Screen"):
        super(Offline_Screen_Detector, self).__init__(g_pool)

        # we need to override self.surface inherited values
        self.init_surfaces()

        # heatmap
        self.heatmap_blur = True
        self.heatmap_blur_gradation = 0.2
        self.gaze_correction_block_size = 1000
        self.gaze_correction_min_confidence = 0.98

    def init_surfaces(self):
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

    def init_gui(self):
        self.menu = ui.Scrolling_Menu('Offline Screen Tracker')
        self.g_pool.gui.append(self.menu)

        self.add_button = ui.Thumb('add_surface',setter=self.add_surface,getter=lambda:False,label='Add Surface',hotkey='a')
        self.g_pool.quickbar.append(self.add_button)
        self.update_gui_markers()

        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))

    def init_marker_cacher(self):
        forking_enable(0) #for MacOs only
        from screen_detector_cacher import fill_cache
        visited_list = [False if x == False else True for x in self.cache]
        video_file_path =  self.g_pool.capture.src
        timestamps = self.g_pool.capture.timestamps
        self.cache_queue = Queue()
        self.cacher_seek_idx = Value('i',0)
        self.cacher_run = Value(c_bool,True)
        self.cacher = Process(target=fill_cache, args=(visited_list,video_file_path,timestamps,self.cache_queue,self.cacher_seek_idx,self.cacher_run,self.min_marker_perimeter))
        self.cacher.start()

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
        if not self.mode == 'Surface edit mode':
            logger.error('Please, select the "Surface edit mode" option at the Mode Selector.')
            return

        correcly_named = [False, False]
        for s in self.surfaces:
            if s.name == 'Left':
                correcly_named[0] = (s.name == 'Left')
            if s.name == 'Right':
                correcly_named[1] = (s.name == 'Right')

        if not (correcly_named[0] and correcly_named[1]):
            logger.error('Please, create two identical surfaces and name them as "Left" and "Right"')
            return
            
        surface_vertices_to_move = []
        for s in self.surfaces:
            s.real_world_size['x'] = s.real_world_size['x']/2.
            if s.name == 'Left':
                uv = s.markers.values()[0].uv_coords
                # original marker position
                surface_vertices_to_move.append((s, 3, uv[0]))
                surface_vertices_to_move.append((s, 2, uv[1]))
                surface_vertices_to_move.append((s, 1, uv[2]))  
                surface_vertices_to_move.append((s, 0, uv[3]))

                # take the midpoint of the segment to the new position
                new_pos = [(uv[0][0][0] + uv[1][0][0])/2., (uv[0][0][1] + uv[1][0][1])/2.]
                surface_vertices_to_move.append((s, 2, new_pos))

                new_pos = [(uv[2][0][0] + uv[3][0][0])/2., (uv[2][0][1] + uv[3][0][1])/2.]
                surface_vertices_to_move.append((s, 1, new_pos))

            if s.name == 'Right':
                uv = s.markers.values()[0].uv_coords
                # original marker position
                surface_vertices_to_move.append((s, 3, uv[0]))
                surface_vertices_to_move.append((s, 2, uv[1]))
                surface_vertices_to_move.append((s, 1, uv[2]))  
                surface_vertices_to_move.append((s, 0, uv[3]))

                # take the midpoint of the segment to the new position
                new_pos = [(uv[0][0][0] + uv[1][0][0])/2., (uv[0][0][1] + uv[1][0][1])/2.]
                surface_vertices_to_move.append((s, 3, new_pos))

                new_pos = [(uv[2][0][0] + uv[3][0][0])/2., (uv[2][0][1] + uv[3][0][1])/2.]
                surface_vertices_to_move.append((s, 0, new_pos))
                     
        for (s, v_idx, new_pos) in surface_vertices_to_move:
            if s.detected:
                s.move_vertex(v_idx,np.array(new_pos))
                s.cache = None
                self.heatmap = None
                self.gaze_cloud = None

        self.update_gui_markers()
      
    def update_gui_markers(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text('The offline screen tracker will look for a screen for each frame of the video. By default it uses surfaces defined in capture. You can change and add more surfaces here.'))
        self.menu.append(ui.Selector('mode',self,label='Mode',selection=["Show Markers and Frames","Show marker IDs", "Surface edit mode","Show Heatmaps","Show Gaze Cloud", "Show Gaze Correction","Show Metrics"] ))
        self.menu.append(ui.Info_Text('To see heatmap, surface metrics, gaze cloud or gaze correction visualizations, click (re)-calculate gaze distributions. Set "X size" and "Y size" for each surface to see heatmap visualizations.'))
        self.menu.append(ui.Button("(Re)-calculate gaze distributions", self.recalculate))
        self.menu.append(ui.Button("Add screen surface", lambda:self.add_surface('_')))

        self.menu.append(ui.Info_Text('1) add two surfaces; 2) name them as "Left" and "Right"; 3) press Left Right segmentation'))
        self.menu.append(ui.Button("Left Right segmentation", self.screen_segmentation))

        self.menu.append(ui.Info_Text('Heatmap Blur'))
        self.menu.append(ui.Switch('heatmap_blur', self, label='Blur'))
        self.menu.append(ui.Slider('heatmap_blur_gradation',self,min=0.01,step=0.01,max=1.0,label='Blur Gradation'))

        self.menu.append(ui.Info_Text('Gaze Correction Parameters requires a non segmented screen'))
        self.menu.append(ui.Text_Input('gaze_correction_block_size', self,label='Block Size'))
        self.menu.append(ui.Switch('gaze_correction_min_confidence', self,min=0.0,step=0.01,max=1.0,label='Minimun gaze confidence')))
        
        self.menu.append(ui.Info_Text('Export gaze metrics. We recalculate metrics for each section when exporting all sections. Press the recalculate button before export the current selected section.'))
        self.menu.append(ui.Button("Export current section", self.save_surface_statsics_to_file))
        self.menu.append(ui.Button("Export all sections", self.export_all_sections))

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

    def add_surface(self,_):
        self.surfaces.append(Offline_Reference_Surface_Extended(self.g_pool))
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

    def recalculate(self):
        super(Offline_Screen_Detector, self).recalculate()
        # calc heatmaps
        in_mark = self.g_pool.trim_marks.in_mark
        out_mark = self.g_pool.trim_marks.out_mark
        section = slice(in_mark,out_mark)
        
        for s in self.surfaces:
            if s.defined:
                s.heatmap_blur = self.heatmap_blur
                s.heatmap_blur_gradation = self.heatmap_blur_gradation
                s.generate_gaze_cloud(section)
                s.gaze_correction_block_size = self.gaze_correction_block_size
                s.gaze_correction_min_confidence = self.gaze_correction_min_confidence
                s.generate_gaze_correction(section)

 
    def gl_display(self):
        """
        Display marker and surface info inside world screen
        """
        super(Offline_Screen_Detector, self).gl_display()

        if self.mode == "Show Gaze Cloud":
            for s in self.surfaces:
                s.gl_display_gaze_cloud()

        if self.mode == "Show Gaze Correction":
            for s in self.surfaces:
                s.gl_display_gaze_correction()

    def export_all_sections(self):
        for section in self.g_pool.trim_marks.sections:
            self.g_pool.trim_marks.focus = self.g_pool.trim_marks.sections.index(section)
            self.recalculate()
            self.save_surface_statsics_to_file()
            for s in self.surfaces:
                if s.heatmap is not None:
                    logger.info("Saved Heatmap as .png file.")
                    cv2.imwrite(os.path.join(metrics_dir,'heatmap'+surface_name+'.png'),s.heatmap)

                if s.gaze_cloud is not None:
                    logger.info("Saved Gaze Cloud as .png file.")
                    cv2.imwrite(os.path.join(metrics_dir,'gaze_cloud'+surface_name+'.png'),s.gaze_cloud)

                # lets save out the current surface image found in video
                seek_pos = in_mark + ((out_mark - in_mark)/2)
                self.g_pool.capture.seek_to_frame(seek_pos)
                new_frame = self.g_pool.capture.get_frame()
                frame = new_frame.copy()
                self.update(frame, None)
                if s.detected and frame.img is not None:
                    #here we get the verts of the surface quad in norm_coords
                    mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32).reshape(-1,1,2)
                    screen_space = cv2.perspectiveTransform(mapped_space_one,s.m_to_screen).reshape(-1,2)
                    
                    #now we convert to image pixel coords
                    screen_space[:,1] = 1-screen_space[:,1]
                    screen_space[:,1] *= frame.img.shape[0]
                    screen_space[:,0] *= frame.img.shape[1]
                    s_0,s_1 = s.real_world_size['x'], s.real_world_size['y'] 
                    
                    #now we need to flip vertically again by setting the mapped_space verts accordingly.
                    mapped_space_scaled = np.array(((0,s_1),(s_0,s_1),(s_0,0),(0,0)),dtype=np.float32)
                    M = cv2.getPerspectiveTransform(screen_space,mapped_space_scaled)
                    
                    #here we do the actual perspactive transform of the image.
                    srf_in_video = cv2.warpPerspective(frame.img,M, (int(s.real_world_size['x']),int(s.real_world_size['y'])) )
                    cv2.imwrite(os.path.join(metrics_dir,'surface'+surface_name+'.png'),srf_in_video)
                    logger.info("Saved current image as .png file.")
                else:
                    logger.info("'%s' is not currently visible. Seek to appropriate frame and repeat this command."%s.name)

                # lets create alternative versions of the surfaces *.pngs
                src1 = cv2.imread(os.path.join(metrics_dir,'surface'+surface_name+'.png'))
                for g in s.output_data['gaze']:
                    cv2.circle(src1, (int(g[0]),int(g[1])), 3, (0, 0, 0), 0)

                for c in s.output_data['kmeans']:
                    cv2.circle(src1, (int(c[0]),int(c[1])), 5, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(metrics_dir,'surface-gaze_cloud'+surface_name+'.png'),src1)

                np.savetxt(os.path.join(metrics_dir,'surface-gaze_cloud'+surface_name+'.txt'), s.output_data['gaze'])
                #src2 = cv2.imread(os.path.join(metrics_dir,'heatmap'+surface_name+'.png'))
                #dst = cv2.addWeighted(src1, .9, src2, .1, 0.0);                
                #cv2.imwrite(os.path.join(metrics_dir,'surface-heatmap'+surface_name+'.png'),dst)
            
            self.g_pool.capture.seek_to_frame(current_frame_index)
            logger.info("Done exporting reference surface data.")

del Offline_Marker_Detector