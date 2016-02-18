# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import os, cv2
import numpy as np
from pyglui import ui

from plugin import Plugin

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Export_Images(Plugin):
    """
        We copied some lines from Pupil code to:
        - Export surface (if any) image of the current frame as png (with homographic transformation).
        - Export current frame image as png (as is).

    """
    def __init__(self,g_pool):
        # self.order = .5
        super(Export_Images, self).__init__(g_pool)
        self.image_dir = os.path.join(self.g_pool.rec_dir,"export_images")
        if os.path.isdir(self.image_dir):
            logger.info("Saving to:"+self.image_dir)
        else:
            try:
                os.mkdir(self.image_dir)
                if os.path.isdir(self.image_dir):
                    logger.info("Saving to:"+self.image_dir)
            except:
                logger.error("Images will not be saved, error creating dir %s"%self.image_dir)
                self.alive = False
                return

        self.menu = None
        self.frame = None
        self.screen_detector = None
        for p in self.g_pool.plugins:
            if p.class_name == 'Offline_Screen_Detector':
                if p.alive:
                    self.screen_detector = p
                    break
        
        if not self.screen_detector:
            self.alive = False
            logger.error('Open the Offline Screen Detector.')

    def init_gui(self):
        """
        if the app allows a gui, you may initalize your part of it here.
        """
        self.menu = ui.Scrolling_Menu('Export Images')
        self.g_pool.gui.append(self.menu)
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',self.unset_alive))
        self.menu.append(ui.Info_Text('Export surface image in current frame as a .png file.\
                                       Surface perspective is transformed from trapezoid to rectangular.'))
        self.menu.append(ui.Button("Export Surfaces",self.export_selected_surface))
        self.menu.append(ui.Info_Text('Export current frame image.'))
        self.menu.append(ui.Button("Export Frame",self.export_current_frame))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def update(self, frame, events):
        self.frame = frame

    def export_current_frame(self):
        frame = self.frame.copy()
        if frame.img is not None:
            cv2.imwrite(os.path.join(self.image_dir,'frame_'+str(frame.index)+'.png'),frame.img)
            logger.info("Saved current frame image as .png file.")
        else:
            logger.error('No frame found.')

    def export_selected_surface(self):
        """
        source: offline_reference_surface.py
        export selected surface in the current frame
        prototype note: surface selection functionality is not implemented
        """
        # lets save out the current surface image found in video
        if self.screen_detector.surfaces:
            s = self.screen_detector.surfaces[0]
        else:
            logger.error('No surfaces were found.')
            return

        frame = self.frame.copy()
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
            file_name = os.path.join(self.image_dir,'frame_'+str(frame.index)+'_surface'+'_'+s.name.replace('/','')+'_'+s.uid+'.png')
            cv2.imwrite(file_name,srf_in_video)
            logger.info("Saved surface image as .png")
        else:
            logger.info("'%s' is not currently visible. Seek to appropriate frame and repeat this command."%s.name)


    ## if you want a session persistent plugin implement this function:
    # def get_init_dict(self):
    #     raise NotImplementedError()
    #     # d = {}
    #     # # add all aguments of your plugin init fn with paramter names as name field
    #     # # do not include g_pool here.
    #     # return d
    def unset_alive(self):
        self.alive = False

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()