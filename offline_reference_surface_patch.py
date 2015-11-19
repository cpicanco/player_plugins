# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

        self.heatmap_blur = False
        self.heatmap_blur_gradation = .2

        self.gaze_cloud = None
        self.gaze_cloud_texture = None

        self.gaze_correction = None
        self.gaze_correction_texture = None

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

            # now lets get recent pupil positions on this surface:
            for gp in self.gaze_on_srf:
                draw_points_norm([gp['norm_pos']],color=RGBA(0.0,0.8,0.5,0.8), size=80)

            glfwSwapBuffers(self._window)
            glfwMakeContextCurrent(active_window)

    def generate_heatmap(self,section):
        if self.cache is None:
            logger.warning('Surface cache is not build yet.')
            return
            
        # removing encapsulation
        x_bin, y_bin = 1, 1
        x_size, y_size = self.real_world_size['x'], self.real_world_size['y']

        # create equidistant edges based on the user defined interval/size
        x_bin = [x for x in xrange(0,int(x_size + 1), int(x_bin))]
        y_bin = [y for y in xrange(0,int(y_size + 1), int(y_bin))]

        all_gaze = []

        for frame_idx,c_e in enumerate(self.cache[section]):
            if c_e:
                frame_idx+=section.start
                for gp in self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen']):
                    all_gaze.append(gp['norm_pos'])

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
            kernel_size = (int(self.heatmap_blur_gradation * len(x_bin)/2)*2 +1)
            sigma = kernel_size /6.

            hist = cv2.GaussianBlur(hist, (kernel_size, kernel_size), sigma)

        # scale convertion necessary for the colormapping
        maxval = np.amax(hist)
        if maxval:
            scale = 255./maxval
        else:
            scale = 0

        # colormapping
        hist = np.uint8(hist * (scale))
        c_map = cv2.applyColorMap(hist, cv2.COLORMAP_JET)

        # we need a 4 channel image to apply transparency
        x, y, channels = c_map.shape
        self.heatmap = np.zeros((x, y, 4), dtype = np.uint8)

        # lets assign the color channels
        self.heatmap[:, :, :3] = c_map

        # alpha blend/transparency
        self.heatmap[:, :, 3] = 127

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

    def generate_gaze_cloud(self,section):
        if self.cache is None:
            logger.warning('Surface cache is not build yet.')
            return
            
        x_size, y_size = self.real_world_size['x'], self.real_world_size['y']

        all_gaze = []
        for frame_idx,c_e in enumerate(self.cache[section]):
            if c_e:
                frame_idx+=section.start
                for gp in self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen']):
                    all_gaze.append(gp['norm_pos'])

        if not all_gaze:
            logger.warning("No gaze data on surface for gaze cloud found.")
            all_gaze.append((-1., -1.))

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
        _,_,centers = cv2.kmeans(all_gaze_float,2,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        for c in centers:
            #print c
            cv2.circle(img, (int(c[0]),int(c[1])), 5, (0, 0, 255), -1)

        self.output_data = {'gaze':all_gaze_flipped,'kmeans':centers}

        alpha = img.copy()
        alpha -= .5*255
        alpha *= -1
        img[:,:,3] = alpha[:,:,0]

        self.gaze_cloud = img
        self.gaze_cloud_texture = Named_Texture()
        self.gaze_cloud_texture.update_from_ndarray(self.gaze_cloud)

    def generate_gaze_correction(self,section):
        # todo: implement more robust outlier handling
        # def remove_outliers(gaze_points):
 
        def bias(gaze_block, k=2):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
            _, _, centers = cv2.kmeans(gaze_block, k, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
            #screen_center = normalize([x_size/2.0, y_size/2.0],(x_size,y_size), True)
            return centers.mean(axis = 0) - (0.5, 0.5) #screen_center

        def correct(gaze_block, bias):
            gaze_block[:, 0] -= bias[0]
            gaze_block[:, 1] -= bias[1]
            return gaze_block

        if self.cache is None:
            logger.error('Surface cache is not build yet.')
            return
            
        all_gaze = []
        gaze_outside_srf = []
        gaze_no_confidence = []
        for frame_idx,c_e in enumerate(self.cache[section]):
            if c_e:
                frame_idx+=section.start
                for gp in self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen']):
                    if gp['on_srf']:
                        if gp['base']['confidence'] > 0.98:
                            all_gaze.append(gp['norm_pos'])
                        else:
                            gaze_no_confidence.append(gp)
                    else:
                        gaze_outside_srf.append(gp)

        if not all_gaze:
            logger.error("No gaze point on surface found.")
            return
        else:
            gaze_count = len(all_gaze)
            logger.info("Found %s gaze points."%gaze_count)
            logger.info("Removed %s outside surface."%len(gaze_outside_srf))
            logger.info("Removed %s with < 0.5"%len(gaze_no_confidence))

        # plot gaze
        # denormalize and flip y
        # all_gaze *= [x_size, y_size]
        # all_gaze_flipped = np.float32([[g[0], abs(g[1]-y_size)] for g in all_gaze])

        # screen_center = np.array([x_size/2.0, y_size/2.0])
        clamped_gaze = np.float32(np.array(all_gaze))

        min_block_size = 1000
        if gaze_count < min_block_size:
            logger.error("Too few data to proceed.")
            return

        bias_along_blocks = []
        unbiased_gaze = []
        for block_start in range(0, gaze_count, min_block_size):
            block_end = block_start + min_block_size
            print block_start, block_end
            if block_end <= gaze_count:
                gaze_block = clamped_gaze[block_start:block_end, :]
                gaze_bias = bias(gaze_block)
            else:
                gaze_block = clamped_gaze[block_start:gaze_count, :]

            bias_along_blocks.append(gaze_bias)
            unbiased_gaze.append(correct(gaze_block, gaze_bias))

        bias_along_blocks = bias_along_blocks
        unbiased_gaze = np.vstack(unbiased_gaze)

        x_size, y_size = self.real_world_size['x'], self.real_world_size['y'] 
        bias_along_blocks = [denormalize(b, (x_size, y_size), True) for b in bias_along_blocks]
        unbiased_gaze = np.float32([denormalize(g, (x_size, y_size), True) for g in unbiased_gaze])
        #unbiased_gaze *= [x_size, y_size]
        #unbiased_gaze = np.float32([[g[0], abs(g[1]-y_size)] for g in unbiased_gaze])

        img = np.zeros((y_size,x_size,4), np.uint8)
        img += 255

        for g in unbiased_gaze:
            cv2.circle(img, (int(g[0]),int(g[1])), 5, (0, 0, 0), 0)

        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        #_,_,centers = cv2.kmeans(unbiased_gaze,2,criteria,20,cv2.KMEANS_RANDOM_CENTERS)

        for b in bias_along_blocks:
            cv2.circle(img, (int(b[0]),int(b[1])), 10, (0, 0, 255), -1)

        # for c in centers:
        #     #print c
        #     cv2.circle(img, (int(c[0]),int(c[1])), 5, (0, 0, 255), -1)

        #self.output_data = {'gaze':unbiased_gaze,'kmeans':centers,'bias':bias_along_blocks}

        alpha = img.copy()
        alpha -= .5*255
        alpha *= -1
        img[:,:,3] = alpha[:,:,0]

        self.gaze_correction = img
        self.gaze_correction_texture = Named_Texture()
        self.gaze_correction_texture.update_from_ndarray(self.gaze_correction)

del Offline_Reference_Surface