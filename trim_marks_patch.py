# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
# modified version of:
# https://github.com/pupil-labs/pupil/blob/ffc548a0764cb26fc4459ff129c9c216f1fafc8d/pupil_src/player/trim_marks.py

from OpenGL.GL import *
from pyglui.cygl.utils import RGBA,draw_points,draw_polyline
from glfw import glfwGetWindowSize,glfwGetCurrentContext,glfwGetCursorPos,GLFW_RELEASE,GLFW_PRESS,glfwGetFramebufferSize
from trim_marks import Trim_Marks

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Trim_Marks_Extended(Trim_Marks):
    """
        Extend trim_marks system plugin to allow multisections.
    """
    def __init__(self, g_pool, focus=0, sections=[]):
        super(Trim_Marks_Extended, self).__init__(g_pool)
        for p in g_pool.plugins:
            if p.class_name == 'Trim_Marks':
                p.alive = False
                break
        g_pool.trim_marks = self # attach self for ease of access by others.

        # focused section
        self._focus = focus

        # sections
        if sections:
            self._sections = sections
            self._in_mark, self._out_mark = self._sections[self._focus]
        else:
            self._in_mark = 0
            self._out_mark = self.frame_count
            sections.append([self._in_mark, self._out_mark])
            self._sections = sections

        self.mid_sections = [self.get_mid_section(s)for s in self._sections] 

    @property
    def sections(self):
        return self._sections

    @sections.setter
    def sections(self, value):
        self._sections = value
        self.mid_sections = [self.get_mid_section(s) for s in self._sections]       

    @property
    def focus(self):
        return self._focus

    @focus.setter
    def focus(self, value):
        self._focus = value
        (self._in_mark, self._out_mark) = self.sections[self._focus]

    @property
    def in_mark(self):
        return self._in_mark

    @in_mark.setter
    def in_mark(self, value):
        self._in_mark = int(min(self._out_mark,max(0,value)))
        self.sections[self.focus][0] = self._in_mark
        self.mid_sections[self.focus] = self.get_mid_section(self.sections[self.focus])

    @property
    def out_mark(self):
        return self._out_mark

    @out_mark.setter
    def out_mark(self, value):
        self._out_mark = int(min(self.frame_count,max(self.in_mark,value)))
        self.sections[self.focus][1] = self._out_mark
        self.mid_sections[self.focus] = self.get_mid_section(self.sections[self.focus])

    def get_mid_section(self, s):
        return int(s[0] + ((s[1]-s[0])/2))

    def set(self,mark_range):
        super(Trim_Marks_Extended, self).set(mark_range)
        self.sections[self.focus][0] = self._in_mark
        self.sections[self.focus][1] = self._out_mark
        self.mid_sections[self.focus] = self.get_mid_section(self.sections[self.focus])

    def on_click(self,img_pos,button,action):
        """
        gets called when the user clicks in the window screen
        """
        hdpi_factor = float(glfwGetFramebufferSize(glfwGetCurrentContext())[0]/glfwGetWindowSize(glfwGetCurrentContext())[0])
        pos = glfwGetCursorPos(glfwGetCurrentContext())
        pos = pos[0]*hdpi_factor,pos[1]*hdpi_factor

        #drag the seek point
        if action == GLFW_PRESS:
            screen_in_mark_pos = self.bar_space_to_screen((self.in_mark,0))
            screen_out_mark_pos = self.bar_space_to_screen((self.out_mark,0))

            #in mark
            dist = abs(pos[0]-screen_in_mark_pos[0])+abs(pos[1]-screen_in_mark_pos[1])
            if dist < 10:
                if self.distance_in_pix(self.in_mark,self.capture.get_frame_index()) > 20:
                    self.drag_in=True
                    return
            #out mark
            dist = abs(pos[0]-screen_out_mark_pos[0])+abs(pos[1]-screen_out_mark_pos[1])
            if dist < 10:
                if self.distance_in_pix(self.out_mark,self.capture.get_frame_index()) > 20:
                    self.drag_out=True

        elif action == GLFW_RELEASE:
            if self.drag_out or self.drag_in:
                logger.info("Section: "+self.get_string())
                self.drag_out = False
                self.drag_in = False

            # would be great to expand the click area horizontally for big sections
            for s in self.sections:
                if s is not self.sections[self.focus]:
                    midsec = self.mid_sections[self.sections.index(s)]
                    screen_midsec_pos = self.bar_space_to_screen((midsec,0))
                    dist = abs(pos[0]-screen_midsec_pos[0])+abs(pos[1]-screen_midsec_pos[1])
                    if dist < 10:
                        if self.distance_in_pix(midsec,self.capture.get_frame_index()) > 20:
                            self.focus = self.sections.index(s)
                            break

    def gl_display(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        glOrtho(-self.h_pad,  (self.frame_count)+self.h_pad, -self.v_pad, 1+self.v_pad,-1,1) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        color1 = RGBA(.1,.9,.2,.5)
        color2 = RGBA(.1,.9,.9,.5)

        if self.in_mark != 0 or self.out_mark != self.frame_count:
            draw_polyline( [(self.in_mark,0),(self.out_mark,0)],color=color1,thickness=2)
 
        draw_points([(self.in_mark,0),],color=color1,size=10)
        draw_points([(self.out_mark,0),],color=color1,size=10)

        if self.sections:
            for s in self.sections:
                if self.sections.index(s) != self.focus:
                    draw_polyline( [(s[0],0),(s[1],0)],color=RGBA(.1,.9,.9,.2),thickness=2)
                for mark in s:
                    draw_points([(mark,0),],color=color2,size=5)

        if self.mid_sections:
            for m in self.mid_sections:
                draw_points([(m,0),],color=RGBA(.1,.9,.9,.1),size=10)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
  
del Trim_Marks
