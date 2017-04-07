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
from pathlib import Path

base_dir = Path(__file__).parents[2]
print(base_dir)
sys.path.append(os.path.join(base_dir,'pupil_plugins_shared'))

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
from screen_tracker import Screen_Tracker
from offline_surface_tracker import Offline_Surface_Tracker
from square_marker_detect import draw_markers,m_marker_to_screen
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
        # heatmap
        # self.min_marker_perimeter = 100
        # self.robust_detection = True
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
        super().__init__(*args,self.mode,100,False,True,)
        
        Trim_Marks_Extended_Exist = False
        for p in self.g_pool.plugins:
            if p.class_name == 'Trim_Marks_Extended':
                Trim_Marks_Extended_Exist = True
                break

        if not Trim_Marks_Extended_Exist:
            from trim_marks_patch import Trim_Marks_Extended
            self.g_pool.plugins.add(Trim_Marks_Extended)
            del Trim_Marks_Extended
            

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

    def init_gui(self):
        self.menu = ui.Scrolling_Menu('Offline Screen Tracker')
        self.g_pool.gui.append(self.menu)
        self.update_gui_markers()

        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))

    def init_marker_cacher(self):
        forking_enable(0) #for MacOs only
        from screen_detector_cacher import fill_cache
        visited_list = [False if x == False else True for x in self.cache]
        video_file_path =  self.g_pool.capture.source_path
        timestamps = self.g_pool.capture.timestamps
        self.cache_queue = Queue()
        self.cacher_seek_idx = Value('i',0)
        self.cacher_run = Value(c_bool,True)
        self.cacher = Process(target=fill_cache, args=(visited_list,video_file_path,timestamps,self.cache_queue,self.cacher_seek_idx,self.cacher_run,self.min_marker_perimeter_cacher))
        self.cacher.start()

    def update_marker_cache(self):
        while not self.cache_queue.empty():
            idx,c_m = self.cache_queue.get()
            self.cache.update(idx,c_m)
            for s in self.surfaces:
                s.update_cache(self.cache,camera_calibration=self.camera_calibration,min_marker_perimeter=self.min_marker_perimeter,min_id_confidence=self.min_id_confidence,idx=idx)
            # if self.cacher_run.value == False:
            #     self.recalculate()


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

    def raise_bug(self):
        raise 's'

    def update_gui_markers(self):

        def close():
            self.alive = False

        def set_min_marker_perimeter(val):
            self.min_marker_perimeter = val
            self.notify_all_delayed({'subject':'min_marker_perimeter_changed'},delay=1)

        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',close))
        self.menu.append(ui.Slider('min_marker_perimeter',self,min=20,max=500,step=1,setter=set_min_marker_perimeter))
        self.menu.append(ui.Info_Text('The offline screen tracker will look for a screen for each frame of the video. By default it uses surfaces defined in capture. You can change and add more surfaces here.'))
        self.menu.append(ui.Selector('mode',self,setter=self.set_mode,label='Mode',selection=["Show Markers and Surfaces","Show marker IDs","Show Heatmaps","Show Gaze Cloud", "Show Kmeans Correction","Show Mean Correction","Show Metrics"] ))
        
        if self.mode == 'Show Markers and Surfaces':
            self.menu.append(ui.Info_Text('To split the screen in two (left,right) surfaces 1) add two surfaces; 2) name them as "Left" and "Right"; 3) press Left Right segmentation'))
            self.menu.append(ui.Button("Left Right segmentation",self.screen_segmentation))
            self.menu.append(ui.Button("Matrix segmentation", self.matrix_segmentation))
            self.menu.append(ui.Button("Add M surfaces", self.add_matrix_surfaces))
            self.menu.append(ui.Button("bug", self.raise_bug))
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

        self.menu.append(ui.Info_Text('Select a section. To see heatmap, surface metrics, gaze cloud or gaze correction visualizations, click (re)-calculate gaze distributions. Set "X size" and "Y size" for each surface to see heatmap visualizations.'))
        self.menu.append(ui.Button("(Re)-calculate gaze distributions",self.recalculate))
        self.menu.append(ui.Info_Text('To use data from all sections to generate visualizations click the next button instead.'))
        self.menu.append(ui.Button("(Re)-calculate",self.recalculate_all_sections))
        self.menu.append(ui.Button("Add screen surface",lambda:self.add_surface('_')))
        
        self.menu.append(ui.Info_Text('Export gaze metrics. We recalculate metrics for each section when exporting all sections. Press the recalculate button before export the current selected section.'))
        self.menu.append(ui.Info_Text("Press the export button or type 'e' to start the export for the current section."))
        self.menu.append(ui.Button("Export all sections", self.export_all_sections))

        self.menu.append(ui.Info_Text('Requires segmentation plugin.'))
        self.menu.append(ui.Button("Export all distances", self.export_all_distances))
        self.menu.append(ui.Button("Precision Report", self.precision_report))
        self.menu.append(ui.Button("Slice 1.5 - precision", self.export_all_precision))

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

    # def update(self,frame,events):
    #     super().update(frame, events)
    #     # locate surfaces
    #     for s in self.surfaces:
    #         if not s.locate_from_cache(frame.index):
    #             s.locate(self.markers)
    #     #     if s.detected:
    #     #         pass
    #     #         # events.append({'type':'marker_ref_surface','name':s.name,'uid':s.uid,'m_to_screen':s.m_to_screen,'m_from_screen':s.m_from_screen, 'timestamp':frame.timestamp,'gaze_on_srf':s.gaze_on_srf})

    def recalculate(self):
        pass

    # def recalculate(self):
    #     #super().recalculate()
    #     # calc heatmaps
    #     in_mark = self.g_pool.trim_marks.in_mark
    #     out_mark = self.g_pool.trim_marks.out_mark
    #     section = slice(in_mark,out_mark)
        
    #     for s in self.surfaces:
    #         if s.defined:
    #             s.heatmap_blur = self.heatmap_blur
    #             s.heatmap_blur_gradation = self.heatmap_blur_gradation
    #             s.heatmap_colormap = self.heatmap_colormap
    #             s.heatmap_use_kdata = self.heatmap_use_kdata
    #             s.gaze_correction_block_size = self.gaze_correction_block_size
    #             s.gaze_correction_min_confidence = self.gaze_correction_min_confidence
    #             s.gaze_correction_k = self.gaze_correction_k

    #             s.generate_gaze_cloud(section)
    #             s.generate_gaze_correction(section)
    #             s.generate_mean_correction(section)
    #             s.generate_heatmap(section)


    #     # calc distirbution accross all surfaces.
    #     results = []
    #     for s in self.surfaces:
    #         gaze_on_srf  = s.gaze_on_srf_in_section(section)
    #         results.append(len(gaze_on_srf))
    #         self.metrics_gazecount = len(gaze_on_srf)

    #     if results == []:
    #         logger.warning("No surfaces defined.")
    #         return
    #     max_res = max(results)
    #     results = np.array(results,dtype=np.float32)
    #     if not max_res:
    #         logger.warning("No gaze on any surface for this section!")
    #     else:
    #         results *= 255./max_res
    #     results = np.uint8(results)
    #     results_c_maps = cv2.applyColorMap(results, cv2.COLORMAP_JET)

    #     for s,c_map in zip(self.surfaces,results_c_maps):
    #         heatmap = np.ones((1,1,4),dtype=np.uint8)*125
    #         heatmap[:,:,:3] = c_map
    #         s.metrics_texture = Named_Texture()
    #         s.metrics_texture.update_from_ndarray(heatmap)

    def recalculate_all_sections(self):
        """
            treats all sections as one
            should not be used to overlaid sections
        """
        # for now, it requires trim_marks_patch.py
        sections_alive = False
        for p in self.g_pool.plugins:
            if p.class_name == 'Trim_Marks_Extended':
                sections_alive = True

        if sections_alive:
            sections = self.g_pool.trim_marks.sections     
            for s in self.surfaces:
                if s.defined:
                    # assign user defined variables
                    s.heatmap_blur = self.heatmap_blur
                    s.heatmap_blur_gradation = self.heatmap_blur_gradation
                    s.heatmap_use_kdata = self.heatmap_use_kdata
                    s.heatmap_colormap = self.heatmap_colormap
                    s.gaze_correction_block_size = self.gaze_correction_block_size
                    s.gaze_correction_min_confidence = self.gaze_correction_min_confidence
                    s.gaze_correction_k = self.gaze_correction_k
                    
                    # generate visualizations
                    s.generate_gaze_cloud(sections, True)
                    s.generate_gaze_correction(sections, True)
                    s.generate_heatmap(sections, True)
                    s.generate_mean_correction(sections, True)

            logger.info("Recalculate visualizations done.")
                    
        else:
            logger.error("Trim_Marks_Extended not found. Have you opened it?")

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

    def precision_report(self, custom_tag=None):
        sections_alive = False
        if self.g_pool.trim_marks.class_name == 'Trim_Marks_Extended':
            sections_alive = True

        segmentation = None
        for p in self.g_pool.plugins:
            if p.class_name == 'Segmentation':
                if p.alive:
                    segmentation = p
                    break

        if (segmentation is not None) and sections_alive:
            export_path = os.path.join(self.g_pool.rec_dir,'exports')
            save_path = os.path.join(export_path,"precision_report")
            if os.path.isdir(save_path):
                logger.info("Overwriting data on precision_report")
            else:
                try:
                    os.mkdir(save_path)
                except:
                    logger.warning("Could not make dir %s"%save_path)
                    return

            angles,x1,y1 = segmentation.scapp_report['Angle'],segmentation.scapp_report['X1'],segmentation.scapp_report['Y1']
            unique_distances = sorted(set(zip(angles,x1,y1)))
            unique_responses = sorted(set(segmentation.scapp_report['ExpcResp']))
            segmentation.filter_by_expresp = True
            segmentation.filter_by_distance = True
            segmentation.filter_by_angle = False
            segmentation.mode = 'in out pairs'

            filtered_gaze = []
            metadata=[]
            for unique_distance in unique_distances:
                segmentation.distance = str(unique_distance)
                for unique_response in unique_responses:
                    (s1, s2, s3) = unique_distance
                    metadata.append("r_%s_distance_%s-%s-%s"%(unique_response, s1, s2, s3))
                    segmentation.expected_response = str(unique_response)
                    segmentation.clean_add_trim()

                    sections = self.g_pool.trim_marks.sections
                    gaze_no_confidence = 0
                    no_surface = 0
                    all_gaze = []
                    for s in self.surfaces: 
                        if s.defined:
                            for sec in sections:
                                in_mark = sec[0]
                                out_mark = sec[1]
                                sec = slice(in_mark,out_mark)
                                for frame_idx,c_e in enumerate(s.cache[sec]):
                                    if c_e:
                                        frame_idx+=sec.start
                                        for i, gp in enumerate(s.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen'])):
                                            if gp['base']['confidence'] >= self.gaze_correction_min_confidence:
                                                all_gaze.append({'frame':frame_idx,'i':i,'norm_pos':gp['norm_pos'],'metatag':'%s-%s-%s-%s'%(unique_response, s1, s2, s3)})
                                            else:
                                                gaze_no_confidence += 1
                                    else:
                                        no_surface += 1

                    if not all_gaze:
                        logger.error("No Gaze points found.")
                        metadata.append("No gaze points found.")
                        return
                    else:
                        gaze_count = len(all_gaze)
                        metadata.append('Found %s frames with no screen/surface.'%no_surface)
                        metadata.append("Found %s gaze points."%gaze_count)
                        metadata.append("Removed '{0}' with confidence < '{1}'".format(gaze_no_confidence, self.gaze_correction_min_confidence))

                    filtered_gaze.append(all_gaze)

            if custom_tag:
                np.save(os.path.join(save_path,'data_ordered_by_metatag'+custom_tag),filtered_gaze)
            else:
                np.save(os.path.join(save_path,'data_ordered_by_metatag'),filtered_gaze)
            #np.savetxt(os.path.join(save_path,'metadata.txt'),metadata)  

            segmentation.clean_custom_events()
            for unique_distance in unique_distances:
                segmentation.distance = str(unique_distance)
                for unique_response in unique_responses:
                    segmentation.expected_response = str(unique_response)
                    segmentation.add_filtered_events()
            
            segmentation.auto_trim()
            filtered_gaze = []
            mean_at_zero_cluster = []
            norm_gaze = []
            for s in self.surfaces: 
                if s.defined:
                    for sec in self.g_pool.trim_marks.sections:
                        section_gaze = []
                        in_mark = sec[0]
                        out_mark = sec[1]
                        sec = slice(in_mark,out_mark)
                        for frame_idx,c_e in enumerate(s.cache[sec]):
                            if c_e:
                                frame_idx+=sec.start
                                for i, gp in enumerate(s.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen'])):
                                    trial = segmentation.trial_from_timestamp(gp['base']['timestamp'])
                                    if gp['base']['confidence'] >= self.gaze_correction_min_confidence:
                                        section_gaze.append({'frame':frame_idx,'i':i,'norm_pos':gp['norm_pos'],'trial':trial})

                        filtered_gaze.append(section_gaze)
            if custom_tag:
                np.save(os.path.join(save_path,'data_ordered_by_trial'+custom_tag),filtered_gaze)
            else:         
                np.save(os.path.join(save_path,'data_ordered_by_trial'),filtered_gaze)                       
        else:
            logger.error("Please, open the segmentation plugin.")

    def export_all_precision(self):
        segmentation = None
        for p in self.g_pool.plugins:
            if p.class_name == 'Segmentation':
                if p.alive:
                    segmentation = p
                    break

        segmentation.onset = 0.0
        segmentation.offset = 1.5
        for status in range(16):
            tag = '_%s-%s'%(segmentation.onset,segmentation.offset)
            tag = tag.replace('.','-')
            logger.info(str(status)+tag)
            self.precision_report(tag)
            segmentation.onset += 0.1 
            segmentation.offset -= 0.1
        logger.info('end')

    def export_all_distances(self):
        segmentation = None
        for p in self.g_pool.plugins:
            if p.class_name == 'Segmentation':
                if p.alive:
                    segmentation = p
                    break

        if segmentation is not None:
            angles,x1,y1 = segmentation.scapp_report['Angle'],segmentation.scapp_report['X1'],segmentation.scapp_report['Y1']
            unique_items = sorted(set(zip(angles,x1,y1)))
            for unique_distance in unique_items:
                segmentation.distance = str(unique_distance)
                segmentation.clean_add_trim()
                in_mark = self.g_pool.trim_marks.in_mark
                out_mark = self.g_pool.trim_marks.out_mark

                # generate visualizations and data
                self.recalculate_all_sections()
                export_path = os.path.join(self.g_pool.rec_dir,'exports')
                save_path = os.path.join(export_path,"distance_%s-%s-%s"%unique_distance)

                if os.path.isdir(save_path):
                    logger.info("Overwriting data on distance %s-%s-%s"%unique_distance)
                else:
                    try:
                        os.mkdir(save_path)
                    except:
                        logger.warning("Could not make dir %s"%save_path)
                        return

                for s in self.surfaces:
                    surface_name = '_'+s.name.replace('/','')+'_'+s.uid
                    if s.heatmap is not None:
                        logger.info("Saved Heatmap as .png file.")
                        cv2.imwrite(os.path.join(save_path,'heatmap'+surface_name+'.png'),s.heatmap)

                    if s.gaze_cloud is not None:
                        logger.info("Saved Gaze Cloud as .png file.")
                        cv2.imwrite(os.path.join(save_path,'gaze_cloud'+surface_name+'.png'),s.gaze_cloud)

                    if s.gaze_correction is not None:
                        logger.info("Saved Gaze Correction as .png file.")
                        cv2.imwrite(os.path.join(save_path,'gaze_correction'+surface_name+'.png'),s.gaze_correction)

                    # export a surface image from the center of the first section for visualization purposes only 
                    self.export_section_image(save_path, s, in_mark, out_mark, os.path.join(save_path,'surface'+surface_name+'.png'))

                    # if s.gaze_correction_mean is not None:
                    #     logger.info("Saved Gaze Correction Mean as .png file.")
                    #     cv2.imwrite(os.path.join(save_path,'gaze_correction_mean'+surface_name+'.png'),s.gaze_correction_mean)

                    np.save(os.path.join(save_path,'source_data'),s.output_data)

    def export_section_image(self,save_path,s,in_mark,out_mark,surface_path):
        # lets save out the current surface image found in video
        seek_pos = in_mark + ((out_mark - in_mark)/2)
        self.g_pool.capture.seek_to_frame(seek_pos)
        new_frame = self.g_pool.capture.get_frame()
        frame = new_frame.copy()
        self.update(frame, {})
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
            cv2.imwrite(surface_path,srf_in_video)
            logger.info("Saved: '%s'"%surface_path)
        else:
            logger.info("'%s' is not currently visible. Seek to appropriate frame and repeat this command."%s.name)


    def export_all_sections(self):
        for section in self.g_pool.trim_marks.sections:
            self.g_pool.trim_marks.focus = self.g_pool.trim_marks.sections.index(section)
            in_mark = self.g_pool.trim_marks.in_mark
            out_mark = self.g_pool.trim_marks.out_mark
            export_path = export_path = os.path.join(self.g_pool.rec_dir,'exports')
            if os.path.isdir(export_path):
                logger.info("Will overwrite export_path")
            else:
                try:
                    os.mkdir(export_path)
                except:
                    logger.warning("Could not make metrics_dir %s"%export_path)
                    return

            metrics_dir = os.path.join(export_path,"%s-%s"%(in_mark,out_mark))
            if os.path.isdir(metrics_dir):
                logger.info("Will overwrite metrics_dir")
            else:
                try:
                    os.mkdir(metrics_dir)
                except:
                    logger.warning("Could not make metrics_dir %s"%metrics_dir)
                    return

            self.recalculate()
            self.save_surface_statsics_to_file(slice(in_mark,out_mark), metrics_dir)

            surface_dir = os.path.join(metrics_dir,'surfaces')

            for s in self.surfaces:
                surface_name = '_'+s.name.replace('/','')+'_'+s.uid
                if s.heatmap is not None:
                    logger.info("Saved Heatmap as .png file.")
                    cv2.imwrite(os.path.join(surface_dir,'heatmap'+surface_name+'.png'),s.heatmap)

                if s.gaze_cloud is not None:
                    logger.info("Saved Gaze Cloud as .png file.")
                    cv2.imwrite(os.path.join(surface_dir,'gaze_cloud'+surface_name+'.png'),s.gaze_cloud)

                if s.gaze_correction is not None:
                    logger.info("Saved Gaze Correction as .png file.")
                    cv2.imwrite(os.path.join(surface_dir,'gaze_correction'+surface_name+'.png'),s.gaze_correction)

                surface_path = os.path.join(surface_dir,'surface'+surface_name+'.png')

                # export a surface image from the center of the section for visualization purposes only
                self.export_section_image(surface_dir, s, in_mark, out_mark, surface_path)

                # lets create alternative versions of the surfaces *.pngs
                src1 = cv2.imread(surface_path)
                for g in s.output_data['gaze']:
                    cv2.circle(src1, (int(g[0]),int(g[1])), 3, (0, 0, 0), 0)

                for c in s.output_data['kmeans']:
                    cv2.circle(src1, (int(c[0]),int(c[1])), 5, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(surface_dir,'surface-gaze_cloud'+surface_name+'.png'),src1)

                np.savetxt(os.path.join(surface_dir,'surface-gaze_cloud'+surface_name+'.txt'), s.output_data['gaze'])
                #src2 = cv2.imread(os.path.join(surface_dir,'heatmap'+surface_name+'.png'))
                #dst = cv2.addWeighted(src1, .9, src2, .1, 0.0);                
                #cv2.imwrite(os.path.join(surface_dir,'surface-heatmap'+surface_name+'.png'),dst)
            
            self.g_pool.capture.seek_to_frame(in_mark)
            logger.info("Done exporting reference surface data.")

    def export_raw_data(self):
        """
        .surface_gaze_positions - gaze_timestamp, surface_norm_x, surface_norm_y

        """
        sections_alive = False
        if self.g_pool.trim_marks.class_name == 'Trim_Marks_Extended':
            sections_alive = True

        segmentation = None
        for p in self.g_pool.plugins:
            if p.class_name == 'Segmentation':
                if p.alive:
                    segmentation = p
                    break

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
        elif notification['subject'] is "should_export":
            logger.info('should_export.. not implemented')
            #logger.info('Min marper perimeter adjusted. Re-detecting surfaces.')
            #self.save_surface_statsics_to_file(notification['range'],notification['export_dir'])


del Screen_Tracker
del Offline_Surface_Tracker