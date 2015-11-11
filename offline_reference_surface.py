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
from methods import GetAnglesPolyline,normalize
from cache_list import Cache_List

#ctypes import for atb_vars:
from ctypes import c_int,c_bool,create_string_buffer
from time import time

import logging
logger = logging.getLogger(__name__)

from reference_surface import Reference_Surface

class Offline_Reference_Surface(Reference_Surface):
    """docstring for Offline_Reference_Surface"""
    def __init__(self,g_pool,name="unnamed",saved_definition=None):
        super(Offline_Reference_Surface, self).__init__(name,saved_definition)
        self.g_pool = g_pool
        self.cache = None
        self.gaze_on_srf = [] # points on surface for realtime feedback display

        self.heatmap_blur = False
        self.heatmap_blur_gradation = .2
        self.heatmap = None
        self.heatmap_texture = None
        self.gaze_cloud = None
        self.gaze_cloud_texture =None
        self.metrics_gazecount = None
        self.metrics_texture = None

        self.output_data = {}

    #cache fn for offline marker
    def locate_from_cache(self,frame_idx):
        if self.cache == None:
            #no cache available cannot update from cache
            return False
        cache_result = self.cache[frame_idx]
        if cache_result == False:
            #cached data not avaible for this frame
            return False
        elif cache_result == None:
            #cached data shows surface not found:
            self.detected = False
            self.m_from_screen = None
            self.m_to_screen = None
            self.gaze_on_srf = []
            self.detected_markers = 0
            return True
        else:
            self.detected = True
            self.m_from_screen = cache_result['m_from_screen']
            self.m_to_screen =  cache_result['m_to_screen']
            self.detected_markers = cache_result['detected_markers']
            self.gaze_on_srf = self.gaze_on_srf_by_frame_idx(frame_idx,self.m_from_screen)
            return True
        raise Exception("Invalid cache entry. Please report Bug.")


    def update_cache(self,marker_cache,idx=None):
        '''
        compute surface m's and gaze points from cached marker data
        entries are:
            - False: when marker cache entry was False (not yet searched)
            - None: when surface was not found
            - {'m_to_screen':,'m_from_screen':,'detected_markers':,gaze_on_srf}
        '''

        # iterations = 0

        if self.cache == None:
            pass
            # self.init_cache(marker_cache)
        elif idx != None:
            #update single data pt
            self.cache.update(idx,self.answer_caching_request(marker_cache,idx))
        else:
            # update where marker cache is not False but surface cache is still false
            # this happens when the markercache was incomplete when this fn was run before
            for i in range(len(marker_cache)):
                if self.cache[i] == False and marker_cache[i] != False:
                    self.cache.update(i,self.answer_caching_request(marker_cache,i))
                    # iterations +=1
        # return iterations



    def init_cache(self,marker_cache):
        if self.defined:
            logger.debug("Full update of surface '%s' positons cache"%self.name)
            self.cache = Cache_List([self.answer_caching_request(marker_cache,i) for i in xrange(len(marker_cache))],positive_eval_fn=lambda x:  (x!=False) and (x!=None))


    def answer_caching_request(self,marker_cache,frame_index):
        visible_markers = marker_cache[frame_index]
        # cache point had not been visited
        if visible_markers == False:
            return False
        # cache point had been visited
        marker_by_id = dict( [ (m['id'],m) for m in visible_markers] )
        visible_ids = set(marker_by_id.keys())
        requested_ids = set(self.markers.keys())
        overlap = visible_ids & requested_ids
        detected_markers = len(overlap)
        if len(overlap)>=min(2,len(requested_ids)):
            yx = np.array( [marker_by_id[i]['verts_norm'] for i in overlap] )
            uv = np.array( [self.markers[i].uv_coords for i in overlap] )
            yx.shape=(-1,1,2)
            uv.shape=(-1,1,2)
            m_to_screen,mask = cv2.findHomography(uv,yx)
            m_from_screen,mask = cv2.findHomography(yx,uv)

            return {'m_to_screen':m_to_screen,
                    'm_from_screen':m_from_screen,
                    'detected_markers':len(overlap)}
        else:
            #surface not found
            return None


    def gaze_on_srf_by_frame_idx(self,frame_index,m_from_screen):
        return self._on_srf_by_frame_idx(frame_index,m_from_screen,self.g_pool.gaze_positions_by_frame[frame_index])


    def fixations_on_srf_by_frame_idx(self,frame_index,m_from_screen):
        return self._on_srf_by_frame_idx(frame_index,m_from_screen,self.g_pool.fixations_by_frame[frame_index])


    def _on_srf_by_frame_idx(self,frame_idx,m_from_screen,data_by_frame):
        data_on_srf = []
        for d in data_by_frame:
            pos = np.array([d['norm_pos']]).reshape(1,1,2)
            mapped_pos = cv2.perspectiveTransform(pos , m_from_screen )
            mapped_pos.shape = (2)
            on_srf = bool((0 <= mapped_pos[0] <= 1) and (0 <= mapped_pos[1] <= 1))
            data_on_srf.append( {'norm_pos':(mapped_pos[0],mapped_pos[1]),'on_srf':on_srf,'base':d } )
        return data_on_srf


    def gl_display_heatmap(self):
        if self.heatmap_texture and self.detected:

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

            draw_named_texture(self.heatmap_texture)

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()


    def gl_display_metrics(self):
        if self.metrics_texture and self.detected:


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

            draw_named_texture(self.metrics_texture)

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

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


    def visible_count_in_section(self,section):
        #section is a slice
        #return number of frames where surface is visible.
        #If cache is not available on frames it is reported as not visible
        if self.cache is None:
            return 0
        section_cache = self.cache[section]
        return sum(map(bool,section_cache))

    def gaze_on_srf_in_section(self,section=slice(0,None)):
        #section is a slice
        #return number of gazepoints that are on surface in section
        #If cache is not available on frames it is reported as not visible
        if self.cache is None:
            return []
        gaze_on_srf = []
        for frame_idx,c_e in enumerate(self.cache[section]):
            frame_idx+=section.start
            if c_e:
                gaze_on_srf += [gp for gp in self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen']) if gp['on_srf']]
        return gaze_on_srf

