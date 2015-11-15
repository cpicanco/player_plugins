# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  Pupil Player is part of Pupil, a Pupil Labs (C) software, see <http://pupil-labs.com>.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
# modified version of display_recent_gaze

from plugin import Plugin
from pyglui.cygl.utils import draw_points_norm,RGBA
from pyglui import ui

class Smoothing_Filter(object):
    def __init__(self):
        super(Smoothing_Filter, self).__init__()
        self.prev = None
        self.prev_ts = None
        self.smoother = 0.5
        self.cut_dist = 0.01


    def filter(self,vals,ts):
        self.prev = vals
        self.pref_ts = ts
        self.filter = self._filter
        return vals


    def _filter(self,vals,ts):
        result = []
        for v,ov in zip(vals,self.prev):
            if abs(ov-v)>self.cut_dist:
                self.prev = tuple(vals)
                return vals
            else:
                result.append(ov+self.smoother*(v-ov))
        self.prev = result
        return type(vals)(result)


class Display_Recent_Gaze(Plugin):
    """
    DisplayGaze shows the three most
    recent gaze position on the screen
    """

    def __init__(self, g_pool, filter_active=True):
        super(Display_Recent_Gaze, self).__init__(g_pool)
        self.order = .8
        self.pupil_display_list = []
        self.filter_active = filter_active
        self.filter = Smoothing_Filter()
        self.menu=None

    def update(self,frame,events):
        if self.filter_active:
            for pt in events.get('gaze_positions',[]):
                self.pupil_display_list.append(self.filter.filter(pt['norm_pos'],pt['timestamp']))
        else:
            for pt in events.get('gaze_positions',[]):
                self.pupil_display_list.append(pt['norm_pos'])
        self.pupil_display_list[:-3] = []

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Recent Gaze')
        # add ui elements to the menu
        self.menu.append(ui.Button('Close', self.unset_alive))
        self.menu.append(ui.Switch('filter_active',self,label='Smooth gaze visualization'))
        self.g_pool.gui.append(self.menu)


    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def cleanup(self):
        self.deinit_gui()

    def gl_display(self):
        draw_points_norm(self.pupil_display_list,
                        size=35,
                        color=RGBA(1.,.2,.4,.6))

    def get_init_dict(self):
        return {'filter_active':True}
