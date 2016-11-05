# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import cv2
from gl_utils import cvmat_to_glmat,clear_gl_screen
from glfw import *
from OpenGL.GL import *
from pyglui.cygl.utils import Named_Texture, draw_points_norm, RGBA

from methods import normalize, denormalize

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from offline_reference_surface import Offline_Reference_Surface

from colormaps import magma, inferno, plasma, viridis

class Offline_Reference_Surface_Extended(Offline_Reference_Surface):
    """
        Extend Offline_Reference_Surface to
        1) show gaze cloud
        2) allow heatmap gradation
        3) 

    """
    def __init__(self,g_pool,name="unnamed",saved_definition=None):
        super(Offline_Reference_Surface_Extended, self).__init__(g_pool,name,saved_definition)
        self.g_pool = g_pool
        self.cache = None
        self.gaze_on_srf = [] # points on surface for realtime feedback display

        # these vars are set at the screen/marker detector __init__
        self.heatmap_blur = None
        self.heatmap_blur_gradation = None
        self.heatmap_colormap = None
        self.heatmap_use_kdata = None

        self.gaze_cloud = None
        self.gaze_cloud_texture = None

        self.gaze_correction = None
        self.gaze_correction_texture = None
        self.gaze_correction_block_size = None
        self.gaze_correction_min_confidence = None
        self.gaze_correction_k = None

        self.mean_correction = None
        self.mean_correction_texture = None

        self.output_data = {}

    def gl_display_gaze_cloud(self):
        if self.gaze_cloud_texture and self.detected:
            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_to_screen)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1,-1,1) # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            #apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            self.gaze_cloud_texture.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

    def gl_display_gaze_correction(self):
        if self.gaze_correction_texture and self.detected:
            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_to_screen)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1,-1,1) # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            #apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            self.gaze_correction_texture.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

    def gl_display_mean_correction(self):
        if self.gaze_correction_texture and self.detected:
            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_to_screen)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1,-1,1) # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            #apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            self.mean_correction_texture.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

    #### fns to draw surface in seperate window
    def gl_display_in_window(self,world_tex):
        """
        here we map a selected surface onto a seperate window.
        """
        if self._window and self.detected:
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            clear_gl_screen()

            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_from_screen)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1,-1,1) # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            #apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            world_tex.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()


            if self.heatmap_texture:
                self.heatmap_texture.draw()

            if self.gaze_cloud_texture:
                self.gaze_cloud_texture.draw()

            if self.gaze_correction_texture:
                self.gaze_correction_texture.draw()

            if self.mean_correction_texture:
                self.mean_correction_texture.draw()

            # now lets get recent pupil positions on this surface:
            for gp in self.gaze_on_srf:
                draw_points_norm([gp['norm_pos']],color=RGBA(0.0,0.8,0.5,0.8), size=80)

            glfwSwapBuffers(self._window)
            glfwMakeContextCurrent(active_window)

    def generate_heatmap(self,section, use_all_sections=False):
        if self.cache is None:
            logger.warning('Surface cache is not build yet.')
            return

        def add_unbiased_gaze(_):
            try:
                for data in self.output_data['unbiased_gaze']:
                    all_gaze.append(normalize(data['gaze'],(x_size, y_size),flip_y=True)) 
            except KeyError, e:
                logger.warning("No unbiased_gaze data was found.",e)

        def add_gaze_on_srf_by_frame(sec):
            for frame_idx,c_e in enumerate(self.cache[sec]):
                if c_e:
                    frame_idx+=sec.start
                    for gp in self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen']):
                        all_gaze.append(gp['norm_pos'])
            
        # removing encapsulation
        x_bin, y_bin = 1, 1
        x_size, y_size = self.real_world_size['x'], self.real_world_size['y']

        # create equidistant edges based on the user defined interval/size
        x_bin = [x for x in xrange(0,int(x_size + 1), int(x_bin))]
        y_bin = [y for y in xrange(0,int(y_size + 1), int(y_bin))]

        all_gaze = []

        if use_all_sections: # 'section' becomes 'trim_marks.sections' 
            for sec in section:
                in_mark = sec[0]
                out_mark = sec[1]
                sec = slice(in_mark,out_mark)
                if self.heatmap_use_kdata:
                    add_unbiased_gaze(sec)
                else:
                    add_gaze_on_srf_by_frame(sec)

        else:  # 'section' becomes 'slice' of current selected section
            if self.heatmap_use_kdata:
                add_unbiased_gaze(section)
            else:
                add_gaze_on_srf_by_frame(section)

        if not all_gaze:
            logger.warning("No gaze data on surface for heatmap found.")
            all_gaze.append((-1., -1.))

        all_gaze = np.array(all_gaze)
        all_gaze *= [x_size, y_size]

        try:
            hist, xedge, yedge = np.histogram2d(all_gaze[:, 0], all_gaze[:, 1],
                                                bins = (x_bin, y_bin),
                                                # range = [[0, x_size], [0, y_size]],
                                                normed = False,
                                                weights = None)
        except (ValueError) as e:
            logger.error("Error:%s" %(e))
            return

        # numpy.histogram2d does not follow the Cartesian convention
        hist = np.rot90(hist)

        # smoothing
        if self.heatmap_blur:
            # must be odd
            kernel_size = (int(self.heatmap_blur_gradation * len(x_bin)/2)*2 +1)
            sigma = kernel_size /6.

            hist = cv2.GaussianBlur(hist, (kernel_size, kernel_size), sigma)

        # scale convertion necessary for the colormapping
        maxval = np.amax(hist)
        if maxval:
            scale = 255./maxval
        else:
            scale = 0
        hist = np.uint8(hist * (scale))
        
        # colormapping
        colormaps = ['magma', 'inferno', 'plasma', 'viridis']

        if self.heatmap_colormap in colormaps:
            if self.heatmap_colormap == 'magma': 
                v = np.asarray(magma)
            elif self.heatmap_colormap == 'inferno': 
                v = np.asarray(inferno)
            elif self.heatmap_colormap == 'plasma': 
                v = np.asarray(plasma)
            elif self.heatmap_colormap == 'viridis': 
                v = np.asarray(viridis)

            v *= 255
            v = v.astype(np.uint8)

            RLUT = v[:,0]
            GLUT = v[:,1]
            BLUT = v[:,2] 
            
            self.heatmap = np.zeros((y_size,x_size,4), np.uint8)
            self.heatmap[:,:,0] = cv2.LUT(hist, BLUT)
            self.heatmap[:,:,1] = cv2.LUT(hist, GLUT)
            self.heatmap[:,:,2] = cv2.LUT(hist, RLUT)
        else:
            c_map = cv2.applyColorMap(hist, cv2.COLORMAP_JET)

            # we need a 4 channel image to apply transparency
            self.heatmap = cv2.cvtColor(c_map, cv2.COLOR_BGR2BGRA)

        # alpha blend/transparency
        c_alpha = hist
        c_alpha[c_alpha>0] = 150

        self.heatmap[:, :, 3] = c_alpha

        # here we approximate the image size trying to inventing as less data as possible
        # so resizing with a nearest-neighbor interpolation gives good results
        self.filter_resize = False
        if self.filter_resize:
            inter = cv2.INTER_NEAREST
            dsize = (int(x_size), int(y_size)) 
            self.heatmap = cv2.resize(src=self.heatmap, dsize=dsize, fx=0, fy=0, interpolation=inter)

        # texturing
        self.heatmap_texture = Named_Texture()
        self.heatmap_texture.update_from_ndarray(self.heatmap)

    def generate_gaze_cloud(self,section,use_all_sections=False):
        if self.cache is None:
            logger.warning('Surface cache is not build yet.')
            return
            
        x_size, y_size = self.real_world_size['x'], self.real_world_size['y']

        all_gaze = []

        if use_all_sections:
            for sec in section:
                in_mark = sec[0]
                out_mark = sec[1]
                sec = slice(in_mark,out_mark)        
                for frame_idx,c_e in enumerate(self.cache[sec]):
                    if c_e:
                        frame_idx+=sec.start
                        for gp in self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen']):
                            all_gaze.append(gp['norm_pos'])

        else:    
            for frame_idx,c_e in enumerate(self.cache[section]):
                if c_e:
                    frame_idx+=section.start
                    for gp in self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen']):
                        all_gaze.append(gp['norm_pos'])

        if not all_gaze:
            logger.warning("No gaze data on surface for gaze cloud found.")
            # all_gaze.append((-1., -1.))
            return
            
        all_gaze = np.array(all_gaze)

        img = np.zeros((y_size,x_size,4), np.uint8)
        img += 255

        # plot gaze
        all_gaze_flipped = [denormalize(g,(x_size, y_size),True) for g in all_gaze]

        for g in all_gaze_flipped:
            cv2.circle(img, (int(g[0]),int(g[1])), 3, (0, 0, 0), 0)
    
        # plot kmeans centers
        all_gaze_float = np.float32(all_gaze_flipped)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _,_,centers = cv2.kmeans(all_gaze_float,self.gaze_correction_k,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        for c in centers:
            #print c
            cv2.circle(img, (int(c[0]),int(c[1])), 5, (0, 0, 255), -1)

        self.output_data = {'gaze':all_gaze_flipped,'kmeans':centers}

        alpha = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        alpha[alpha == 0] = 150
        alpha[alpha == 255] = 0
        alpha[alpha > 0] = 150
        img[:,:,3] = alpha

        self.gaze_cloud = img
        self.gaze_cloud_texture = Named_Texture()
        self.gaze_cloud_texture.update_from_ndarray(self.gaze_cloud)

    def add_gaze(self,sections, use_all_sections):
        gaze_no_confidence = 0
        gaze_outside_srf = 0
        no_surface = 0
        all_gaze = []
        if not use_all_sections:
            sections = [[sections.start,sections.stop]]

        for sec in sections:
            section = slice(sec[0],sec[1])
            for frame_idx,c_e in enumerate(self.cache[section]):
                if c_e:
                    frame_idx += section.start
                    for i, gp in enumerate(self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen'])):
                        if gp['on_srf']:
                            if gp['base_data']['confidence'] >= self.gaze_correction_min_confidence:
                                all_gaze.append({'frame':frame_idx,'i':i,'norm_pos':gp['norm_pos']})
                            else:
                                gaze_no_confidence += 1
                        else:
                            gaze_outside_srf += 1
                else:
                    no_surface += 1

        return gaze_no_confidence, gaze_outside_srf, no_surface, all_gaze

    def generate_gaze_correction(self,section,use_all_sections=False):
        # todo: implement more robust outlier handling
        # def remove_outliers(gaze_points):
        kmeans_plugin_alive = False
        for p in self.g_pool.plugins:
            if p.class_name == 'KMeans_Gaze_Correction':
                if p.alive:
                    kmeans_plugin_alive = True 
                    kmeans_plugin = p
                    kmeans_plugin.alive = False
                    break

        def bias(gaze_block, k=2):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
            _, _, centers = cv2.kmeans(data=gaze_block,
                                       K=k,
                                       criteria=criteria,
                                       attempts=10,
                                       flags=cv2.KMEANS_RANDOM_CENTERS)

            return np.array([x_size/2.0, y_size/2.0]) - centers.mean(axis = 0) 

        def correction(gaze_block, bias):
            gaze_block[:, 0] += bias[0]
            gaze_block[:, 1] += bias[1]
            return gaze_block

        if self.cache is None:
            logger.error('Surface cache is not build yet.')
            return

        gaze_no_confidence, gaze_outside_srf, no_surface, all_gaze = self.add_gaze(section, use_all_sections)
        
        if not all_gaze:
            logger.error("No gaze point on surface found.")
            return
        else:
            gaze_count = len(all_gaze)
            logger.info('Found %s frames with no surface.'%no_surface)
            logger.info("Found %s gaze points."%gaze_count)
            logger.info("Removed %s outside surface."%gaze_outside_srf)
            logger.info("Removed '{0}' with confidence < '{1}'".format(gaze_no_confidence, self.gaze_correction_min_confidence))

        # denormalize (and flip y)
        # right now, we must denormalize before the conversion from float64 to float32 
        # would be excelent to have a kmeans implementation that does not require such a conversion
        x_size, y_size = self.real_world_size['x'], self.real_world_size['y'] 
        clamped_gaze = np.array([denormalize(g['norm_pos'], (x_size, y_size), True) for g in all_gaze]).astype('float32')

        min_block_size = int(self.gaze_correction_block_size)

        if gaze_count < min_block_size:
            logger.error("Too few data to proceed. Using min_block_size = gaze_count ")
            min_block_size = gaze_count

        bias_along_blocks = []
        unbiased_gaze = []
        for block_start in range(0, gaze_count, min_block_size):
            block_end = block_start + min_block_size
            if block_end <= gaze_count:
                gaze_block = clamped_gaze[block_start:block_end, :]
                gaze_bias = bias(gaze_block, self.gaze_correction_k)
            else:
                block_end = gaze_count
                gaze_block = clamped_gaze[block_start:block_end, :]

            bias_along_blocks.append({'bias':gaze_bias, 'block':[block_start,block_end]})
            unbiased_gaze.append(correction(gaze_block, gaze_bias))

        unbiased_gaze = np.vstack(unbiased_gaze)
    
        # draw
        img = np.zeros((y_size,x_size,4), np.uint8)
        img += 255

        for g in unbiased_gaze:
            cv2.circle(img, (int(g[0]),int(g[1])), 3, (0, 0, 0), 0)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _,_,centers = cv2.kmeans(unbiased_gaze.astype('float32'),self.gaze_correction_k,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        for c in centers:
             cv2.circle(img, (int(c[0]),int(c[1])), 5, (0, 0, 255), -1)

        bias_to_draw = [[(x_size/2.)-b['bias'][0], (y_size/2.)-b['bias'][1]] for b in bias_along_blocks]
        for b in bias_to_draw:
            cv2.circle(img, (int(b[0]),int(b[1])), 5, (255, 0, 0), -1)

        unbiased_gaze = [{'frame':g['frame'],
                          'i': g['i'],
                          'gaze':unbiased_gaze[i],
                          'gaze_base':g['norm_pos']} for i, g in enumerate(all_gaze)]

        self.output_data['unbiased_gaze'] = unbiased_gaze
        self.output_data['unbiased_kmeans'] = centers
        self.output_data['bias_along_blocks'] = bias_along_blocks

        # transparent background; data opacity
        alpha = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        alpha[alpha == 0] = 150
        alpha[alpha == 255] = 0
        alpha[alpha > 0] = 150
        img[:,:,3] = alpha

        self.gaze_correction = img
        self.gaze_correction_texture = Named_Texture()
        self.gaze_correction_texture.update_from_ndarray(self.gaze_correction)

        if kmeans_plugin_alive:
            kmeans_plugin.alive = True

    def generate_mean_correction(self,section,use_all_sections=False):
        kmeans_plugin_alive = False
        for p in self.g_pool.plugins:
            if p.class_name == 'KMeans_Gaze_Correction':
                if p.alive:
                    kmeans_plugin_alive = True 
                    kmeans_plugin = p
                    kmeans_plugin.alive = False
                    break
        
        if self.cache is None:
            logger.error('Surface cache is not build yet.')
            return

        def bias(gaze_block):
            x_bias = np.mean(gaze_block[:, 0])
            y_bias = np.mean(gaze_block[:, 1])
            return np.array([0.5-x_bias, 0.5-y_bias])

        def correction(gaze_block, bias):
            gaze_block[:, 0] += bias[0]
            gaze_block[:, 1] += bias[1]
            return gaze_block

        if self.cache is None:
            logger.error('Surface cache is not build yet.')
            return

        gaze_no_confidence, gaze_outside_srf, no_surface, all_gaze = self.add_gaze(section, use_all_sections)

        if not all_gaze:
            logger.error("No gaze point on surface found.")
            return
        else:
            gaze_count = len(all_gaze)
            logger.info('Found %s frames with no surface.'%no_surface)
            logger.info("Found %s gaze points."%gaze_count)
            logger.info("Removed %s outside surface."%gaze_outside_srf)
            logger.info("Removed '{0}' with confidence < '{1}'".format(gaze_no_confidence, self.gaze_correction_min_confidence))

        clamped_gaze = np.array([g['norm_pos'] for g in all_gaze])
        #clamped_gaze = np.array([denormalize(g['norm_pos'], (x_size, y_size), True) for g in all_gaze]).astype('float32')

        min_block_size = int(self.gaze_correction_block_size)

        if gaze_count < min_block_size:
            logger.error("Too few data to proceed. Using min_block_size = gaze_count ")
            min_block_size = gaze_count

        bias_along_blocks = []
        unbiased_gaze = []
        for block_start in range(0, gaze_count, min_block_size):
            block_end = block_start + min_block_size
            if block_end <= gaze_count:
                gaze_block = clamped_gaze[block_start:block_end, :]
                gaze_bias = bias(gaze_block)
            else:
                block_end = gaze_count
                gaze_block = clamped_gaze[block_start:block_end, :]

            bias_along_blocks.append({'bias':gaze_bias, 'block':[block_start,block_end]})
            unbiased_gaze.append(correction(gaze_block, gaze_bias))

        unbiased_gaze = np.vstack(unbiased_gaze)
    
        # draw
        x_size, y_size = self.real_world_size['x'], self.real_world_size['y'] 
        img = np.zeros((y_size,x_size,4), np.uint8)
        img += 255

        gaze_to_draw = [denormalize(g, (x_size, y_size), True) for g in unbiased_gaze]
        for g in gaze_to_draw:
            cv2.circle(img, (int(g[0]),int(g[1])), 3, (0, 0, 0), 0)

        
        bias_to_draw = [denormalize(b['bias'], (x_size, y_size), False) for b in bias_along_blocks]  
        
        # top left = (0,0)
        bias_to_draw = [[(x_size/2.)-b[0], (y_size/2.)+b[1]] for b in bias_to_draw]

        for b in bias_to_draw:
            cv2.circle(img, (int(b[0]),int(b[1])), 5, (255, 0, 0), -1)

        unbiased_gaze = [{'frame':g['frame'],
                          'i': g['i'],
                          'gaze':unbiased_gaze[i],
                          'gaze_base':g['norm_pos']} for i, g in enumerate(all_gaze)]

        self.output_data['unbiased_gaze_mean'] = unbiased_gaze
        self.output_data['bias_along_blocks_mean'] = bias_along_blocks

        alpha = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        alpha[alpha == 0] = 150
        alpha[alpha == 255] = 0
        alpha[alpha > 0] = 150
        img[:,:,3] = alpha

        self.mean_correction = img
        self.mean_correction_texture = Named_Texture()
        self.mean_correction_texture.update_from_ndarray(self.mean_correction)

        if kmeans_plugin_alive:
            kmeans_plugin.alive = True


    def invalidate(self):
        self.cache = None
        self.heatmap = None
        self.gaze_cloud = None
    
    def move_vertex(self, vertex_index, newpos, incremental=False):
        """
        ######### #########
        # 0 . 1 # # tl.tr #
        # .   . # # .   . #
        # 3 . 2 # # bl.br #
        # uv #### #########

        #########
        # 3 . 2 #
        # .   . #
        # 0 . 1 #
        # sv #### 
        """
        if incremental:
            before = self.markers.values()[0].uv_coords # uv
        else:
            before = marker_corners_norm # sv
        after = before.copy()
        after[vertex_index] = newpos
        transform = cv2.getPerspectiveTransform(after,before)
        for m in self.markers.values():
            m.uv_coords = cv2.perspectiveTransform(m.uv_coords,transform)

    @property
    def left_top(self):
        return self.markers.values()[0].uv_coords[0][0]

    @left_top.setter
    def left_top(self, value):
        self.move_vertex(0,value, incremental=True)

    @property
    def right_top(self):
        return self.markers.values()[0].uv_coords[1][0]

    @right_top.setter
    def right_top(self, value):
        self.move_vertex(1,value, incremental=True)

    @property
    def left_bottom(self):
        return self.markers.values()[0].uv_coords[3][0]

    @left_bottom.setter
    def left_bottom(self, value):
        self.move_vertex(3,value, incremental=True)

    @property
    def right_bottom(self):
        return self.markers.values()[0].uv_coords[2][0]

    @right_bottom.setter
    def right_bottom(self, value):
        self.move_vertex(2  ,value, incremental=True)

del Offline_Reference_Surface