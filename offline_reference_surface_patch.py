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
from pyglui.cygl.utils import create_named_texture,update_named_texture, draw_named_texture, draw_points_norm, RGBA

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from offline_reference_surface import Offline_Reference_Surface

class Offline_Reference_Surface_Extended(Offline_Reference_Surface):
    """
        Extend Offline_Reference_Surface to
        1) show gaze cloud
        2) allow heatmap gradation
        3) 

    """
    def __init__(self,g_pool,name="unnamed",saved_definition=None):
        super(Offline_Reference_Surface_Patch, self).__init__(name,saved_definition)
        for p in g_pool.plugins:
            if p.class_name = 'Offline_Reference_Surface':
                p.alive = False
                break

        self.g_pool = g_pool
        self.cache = None
        self.gaze_on_srf = [] # points on surface for realtime feedback display

        self.heatmap_blur = False
        self.heatmap_blur_gradation = .2

        self.gaze_cloud = None

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

            draw_named_texture(self.gaze_cloud_texture)

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

    #### fns to draw surface in seperate window
    def gl_display_in_window(self,world_tex_id):
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

            draw_named_texture(world_tex_id)

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()


            if self.heatmap_texture:
                draw_named_texture(self.heatmap_texture)

            if self.gaze_cloud_texture:
                draw_named_texture(self.gaze_cloud_texture)

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

        hist, xedge, yedge = np.histogram2d(all_gaze[:, 0], all_gaze[:, 1],
                                            bins = (x_bin, y_bin),
                                            # range = [[0, x_size], [0, y_size]],
                                            normed = False,
                                            weights = None)

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
        self.heatmap_texture = create_named_texture(self.heatmap.shape)
        update_named_texture(self.heatmap_texture, self.heatmap)

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
        all_gaze *= [x_size, y_size]
        all_gaze_flipped = [[g[0], abs(g[1]-y_size)] for g in all_gaze]

        for g in all_gaze_flipped:
            cv2.circle(img, (int(g[0]),int(g[1])), 3, (0, 0, 0), 0)
    
        # plot kmeans centers
        all_gaze_float = np.float32(all_gaze_flipped)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _,_,centers = cv2.kmeans(all_gaze_float,2,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        for c in centers:
            print c
            cv2.circle(img, (int(c[0]),int(c[1])), 5, (0, 0, 255), -1)

        self.output_data = {'gaze':all_gaze_flipped,'kmeans':centers}

        alpha = img.copy()
        alpha -= .5*255
        alpha *= -1
        img[:,:,3] = alpha[:,:,0]

        self.gaze_cloud = img

        self.gaze_cloud_texture = create_named_texture(self.gaze_cloud.shape)
        update_named_texture(self.gaze_cloud_texture, self.gaze_cloud)


del Offline_Reference_Surface