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
    def __init__(self, g_pool, target_surface_name= 'None'):
        # self.order = .5
        super(Export_Images, self).__init__(g_pool)
        self.image_dir = os.path.join(self.g_pool.rec_dir,"exported_images")
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

        self.target_surface_name = target_surface_name
        self.menu = None
        self.frame = None
        self.surface_tracker = None
        for p in self.g_pool.plugins:
            if p.class_name == 'Screen_Tracker_Offline':
                if p.alive:
                    self.surface_tracker = p
                    break

            elif p.class_name == 'Offline_Surface_Tracker':
                if p.alive:
                    self.surface_tracker = p
                    break
        
        if not self.surface_tracker:
            self.alive = False
            logger.error('Open the Offline Screen Tracker or the Offline Surface Tracker.')
        else:
            if self.surface_tracker.surfaces:
                self.surface_names = [s.name for s in self.surface_tracker.surfaces]

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Export Images'
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close', self.close))
        self.menu.append(ui.Info_Text(
            'Export surface image in current frame as a .png file.'+
            ' Surface perspective is transformed from trapezoid to rectangular.'+
            ' Close and open this plugin to update the target surface list.'))
        self.menu.append(ui.Button("Export Surfaces",self.export_selected_surface))
        self.menu.append(ui.Info_Text('Export current frame image.'))
        self.menu.append(ui.Button("Export Frame",self.export_current_frame))
        self.menu.append(ui.Selector('target_surface_name',
            self,label='Target Surface',selection=['None']+self.surface_names))            

    def recent_events(self, events):
        self.frame = events.get('frame')

    def export_current_frame(self):
        frame = self.frame.copy()
        if frame.img is not None:
            cv2.imwrite(os.path.join(self.image_dir,'frame_%05d'%frame.index+'.png'),frame.img)
            logger.info("Saved current frame image as .png file.")
        else:
            logger.error('No frame found.')

    def export_selected_surface(self):
        """
        source: offline_reference_surface.py
        export selected surface in the current frame
        """
        if self.target_surface_name == 'None':
            logger.error('Select a surface first.')
            return

        for target_surface in self.surface_tracker.surfaces:
            if target_surface.name == self.target_surface_name:
                surface = target_surface

        if surface is None:
            logger.error('Selected surface does not exist.')
            return

        frame = self.frame.copy()
        if surface.detected and frame.img is not None:
            # get the verts of the surface quad in norm_coords
            mapped_space_one = np.array(
                ((0,0),(1,0),(1,1),(0,1)),dtype=np.float32).reshape(-1,1,2)
            screen_space = cv2.perspectiveTransform(
                mapped_space_one,surface.m_to_screen).reshape(-1,2)
            
            # convert to image pixel coords
            screen_space[:,1] = 1-screen_space[:,1]
            screen_space[:,1] *= frame.img.shape[0]
            screen_space[:,0] *= frame.img.shape[1]
            s_0,s_1 = surface.real_world_size['x'], surface.real_world_size['y'] 
            
            # flip vertically again setting mapped_space verts accordingly
            mapped_space_scaled = np.array(((0,s_1),(s_0,s_1),(s_0,0),(0,0)),dtype=np.float32)
            M = cv2.getPerspectiveTransform(screen_space,mapped_space_scaled)
            
            # perspective transformation
            srf_in_video = cv2.warpPerspective(
                frame.img,M, (int(surface.real_world_size['x']),
                              int(surface.real_world_size['y'])))

            # save to file
            file_name = '_'.join([
                'frame',
                '%05d'%frame.index,
                'surface',
                surface.name.replace('/',''),
                surface.uid,
                '.png'])
            cv2.imwrite(os.path.join(self.image_dir, file_name), srf_in_video)
            logger.info("Saved surface image as .png")
        else:
            logger.info(
                'Surface "%s" is not currently visible.'%surface.name+
                ' Seek to appropriate frame and repeat this command.')
    
    def get_init_dict(self):
        return {'target_surface_name': self.target_surface_name}

    def close(self):
        self.alive = False

    def deinit_ui(self):
        self.remove_menu()