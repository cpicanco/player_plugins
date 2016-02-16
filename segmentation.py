# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2015 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

from os import path
from ast import literal_eval

from pyglui.cygl.utils import RGBA,draw_points,draw_polyline
from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D
from glfw import glfwGetWindowSize, glfwGetCurrentContext, GLFW_KEY_V, GLFW_KEY_COMMA
from pyglui import ui
import numpy as np

from plugin import Plugin

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) 

class TrialContainer(object):
    def __init__(self):
        self.Angle = None
        self.ExpectedResponse = None
        self.TimeEvents = []
        self.Timestamps = []
        self.Distance = []
        # self.GazePoints = []
        # self.CirPoints = []
        # self.CirRanking = []
        # self.CirAPoints = []
        # self.CirBPoints = []

class Segmentation(Plugin):
    """
    The user can manually create events by pressing
    keyboard keys.

    This plugin will display vertical bars at the bottom seek bar
    based on those events.

    One should be able to send those events
    as sections to the trim_marks plugin (auto-trim).

    The auto-trim functionality includes two options:
    - example events = list(1, 50, 100, 150)
    - chain
      - would return the following sections[(1,50), (51,100), (101, 150)]
    - in out pairs
      - would return the following sections[(1,50), (100, 150)]
  
    Todo:
    - import events from Pupil like timestamps
    - selector to manage multiple saved sections

    """
    def __init__(self, g_pool, custom_events=[], mode='chain', keep_create_order=True,
                expected_response='NA',filter_by_expresp=True,
                angle='NA',filter_by_angle=True,
                distance='NA',filter_by_distance=True,
                offset=0.0,onset=0.0,
                color='red'):
        super(Segmentation, self).__init__(g_pool)
        Trim_Marks_Extended_Exist = False
        for p in g_pool.plugins:
            if p.class_name == 'Trim_Marks_Extended':
                Trim_Marks_Extended_Exist = True
                break

        if not Trim_Marks_Extended_Exist:
            from trim_marks_patch import Trim_Marks_Extended
            g_pool.plugins.add(Trim_Marks_Extended)
            del Trim_Marks_Extended

        # Pupil Player system configs
        self.trim_marks = g_pool.trim_marks
        self.order = .8
        self.uniqueness = "by_class"

        # Pupil Player data
        self.capture = g_pool.capture
        #self.current_frame_index = self.capture.get_frame_index()
        self.frame_count = self.capture.get_frame_count()
        self.frame_index = None
        # self.timestamps = g_pool.timestamps

        # display layout
        self.padding = 20. #in screen pixel

        # initialize empty menu and local variables
        self.menu = None
        self.mode = mode
        self.keep_create_order = keep_create_order


        # persistence
        self.custom_events_path = path.join(self.g_pool.rec_dir,'custom_events.npy')
        try:
            self.custom_events = list(np.load(self.custom_events_path))
            if not self.custom_events:
                logger.warning("List is empty. Loading from cache.")
                self.custom_events = custom_events
            else:
                logger.info("Custom events loaded: "+ self.custom_events_path)
        except:
            logger.warning("No custom events at: "+ self.custom_events_path)
            self.custom_events = custom_events
            if not self.custom_events:
                logger.warning("No chached events were found.")
            else:
                logger.warning("Using chached events. Please, save them if necessary.")

        # stimulus control application data
        self.scapp_output = None
        try:
            self.load_scapp_output()
        except Exception, e:
            logger.warning("scapp_output.timestamps error")
        
        self.scapp_report = None
        self.load_scapp_report()

        self.scapp_output_npy = None
        self.load_scapp_output_npy()

        # todo: find a way to load these properties dinamically based on column names
        self.expected_response = expected_response
        self.filter_by_expresp = filter_by_expresp

        self.angle = angle
        self.filter_by_angle = filter_by_angle

        self.distance = distance
        self.filter_by_distance = filter_by_distance

        self.offset = offset
        self.onset = onset

        self.color = color

    def load_scapp_output_npy(self):
        scapp_output_path = path.join(self.g_pool.rec_dir,"scapp_output.npy")
        if not path.isfile(scapp_output_path):
            logger.warning("File not found: "+ scapp_output_path)
            return

        scapp_output = np.load(scapp_output_path)
        self.scapp_output_npy = [[]]
        for line in scapp_output:
            trial_no = line[0] 
            timestamp = line[1]
            event = line[2]

            i = int(trial_no)

            if i > len(self.scapp_output_npy):
                self.scapp_output_npy.append([])
            self.scapp_output_npy[i - 1].append((timestamp, event))


    def load_scapp_output(self): # timestamps of scapp events
        """
        __________________________________________________________

        dependency: validy self.g_pool.rec_dir + '\scapp_output' path
        __________________________________________________________


        - scapp is an acronymous for Stimulus Control Application
        
        - scapp_output has the following format:
            
            (Trial_No, timestamp, event:session_time)

          where:
            - 'Trial_No' is the trial number as in scapp_report
            - 'timestamp' is the timestamps sent by Pupil Server and received by scapp Client
            - 'event is an scapp event, now there are four types:
                - S  : Starter onset | Trial onset | ITI ending
                - *R : first response after S | Starter ending
                - R  : responses after *R
                - C  : Consequence onset | Trial ending | ITI onset
                - session_time is event occurence time in ms from the session onset.

        > examples:
        > scapp_output 
            ('1', '232.5674', 'S:029367')
            ('1', '232.5675', '*R:029368')
            ('1', '232.5676', 'C:029369')
            ('2', '232.5684', 'S:029377')
            ('2', '232.5685', '*R:029378')
            ('2', '232.5686', 'R:029379')
            ('2', '232.5687', 'C:029380')
     
        > scapp_output loaded
            [  [ ('232.5674', 'S:029367'), ('232.5675', '*R:029368'), ('232.5676', 'C:029369') ],
               [ ('232.5684', 'S:029377'), ('232.5685', '*R:029378'), ('232.5686', 'R:029379'), ('232.5687', 'C:029380') ]  ]

        """
        scapp_output_path = path.join(self.g_pool.rec_dir,'scapp_output.timestamps')      
        if path.isfile(scapp_output_path):
            self.scapp_output = [[]]
            with open(scapp_output_path, 'r') as scapp_output:
                for line in scapp_output:
                    (trial_no, timestamp, event) = literal_eval(line)

                    i = int(trial_no)

                    if i > len(self.scapp_output):
                        self.scapp_output.append([])
                    self.scapp_output[i - 1].append((timestamp, event))
        else:
            logger.warning("File not found: "+ scapp_output_path)

    def load_scapp_report(self):
        """
        __________________________________________________________
        
        dependency: validy self.g_pool.rec_dir + '\scapp_report' path

        report_type: string | 'fpe', 'eos', 'vlh'
        __________________________________________________________

        

           Source Header Names for Feature Positive Effect (fpe) trials:
           
           Trial_No : Trial increment number (unique).              (INT)
           Trial_Id : Trial identification number (can repeat).     (INT)
           TrialNam : Trial String Name.                            (STR)
           ITIBegin : Consequence / Inter Trial Interval onset.     (TIME)
           __ITIEnd : Starter begin / End of Inter Trial Interval   (TIME)
           StartLat : Latency of the starter response.              (TIME)
           StmBegin : Trial stimulus/stimuli onset.                 (TIME)
           _Latency : Latency.                                      (TIME)
           __StmEnd : End of the stimulus/stimuli removal.          (TIME)

           ExpcResp : Expected response / Contingency.              (STR)
                Positiva
                Negativa
           __Result : Type of the Response emmited.                 (STR)
                MISS
                HIT
                NONE
           RespFreq : Number of responses emmited                   (INT)



           Source Header Names for Eye Orientation Study (eos) trials:

           Trial_No : idem
           Trial_Id : idem
           TrialNam : idem
           ITIBegin : idem
           __ITIEnd : idem
           StmBegin : idem
           _Latency : idem
           __StmEnd : idem
           ___Angle : Angle                                          (STR)
                0, 45, 90, 135
           ______X1 : left 1
           ______Y1 : top 1
           ______X2 : left 2
           ______Y2 : top 2
           ExpcResp : Expected response
                0 = no gap/false
                1 = gap/true
           RespFreq : idem

           Source Header Names for Variable Limited Hold Study (vlh) trials:

           Trial_No : idem
           Trial_Id : idem
           TrialNam : idem
           ITIBegin : idem
           __ITIEnd : idem
           StmBegin : idem
           _Latency : idem
           ___Cycle :
           __Timer2 :
           _Version :
           ____Mode :
           RespFreq : 

           All time variables are in miliseconds. Counting started
            at the beginning of the session.
        """
        scapp_report_path = path.join(self.g_pool.rec_dir,'scapp_report.data')
        if path.isfile(scapp_report_path):
            try:
                self.scapp_report = np.genfromtxt(scapp_report_path,
                    delimiter="\t", missing_values=["NA"], skip_header=6, skip_footer=1,
                    filling_values=None, names=True, deletechars='_', autostrip=True,
                    dtype=None)
            except ValueError, e:
                logger.warning("genfromtxt error")
        else:
            logger.warning("File not found: "+ scapp_report_path)

    def event_undo(self, arg):
        if self.custom_events:
            self.custom_events.pop()
            if not self.keep_create_order:
                self.custom_events = sorted(self.custom_events, key=int)
  
    def create_custom_event(self, arg):
        if self.frame_index:
            if self.frame_index not in self.custom_events:
                self.custom_events.append(self.frame_index)
                if not self.keep_create_order:
                    self.custom_events = sorted(self.custom_events, key=int)

    def save_custom_events(self):  
        np.save(self.custom_events_path,np.asarray(self.custom_events))

    def auto_trim(self):
        # create sections and pass them to the trim_marks
        sections = []
        events = sorted(self.custom_events, key=int)
        size = len(events)
        if size > 1:
            i = 0
            while True:
                if self.mode == 'chain':
                    if i == 0:
                        sections.append([events[i],events[i+1]])
                    elif (i > 0) and (i < (size-1)):
                        sections.append([events[i]+1,events[i+1]])
                    i += 1
                
                elif self.mode == 'in out pairs':
                    if i < (size-1):
                        sections.append([events[i],events[i+1]])
                    i += 2

                if i > (size-1):
                    break

        self.trim_marks.sections = sections
        self.trim_marks.focus = 0

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Segmentation')
        # add ui elements to the menu
        self.menu.append(ui.Button('Close', self.unset_alive))
        self.menu.append(ui.Info_Text('You can create custom events by pressing "v". To undo press ", (comma)". Remember to save them when your were done.')) 
        self.menu.append(ui.Switch('keep_create_order',self,label="Keep Creation Order"))
        # maybe thumbs instead keyboard keys?
        self.menu.append(ui.Hot_Key('create_event',setter=self.create_custom_event,getter=lambda:True,label='V',hotkey=GLFW_KEY_V))
        self.menu.append(ui.Hot_Key('event_undo',setter=self.event_undo,getter=lambda:True,label=',',hotkey=GLFW_KEY_COMMA))
        self.menu.append(ui.Button('Save Events',self.save_custom_events))
        self.menu.append(ui.Button('Clean All Events',self.clean_custom_events))
        self.menu.append(ui.Info_Text('You can auto-trim based on avaiable events. Choose the Trim Mode that fit your needs.'))
        self.menu.append(ui.Selector('mode',self,label='Trim Mode',selection=['chain','in out pairs'] )) 
        self.menu.append(ui.Button('Auto-trim',self.auto_trim))

        # todo: for each data column, load filters dinamically based on filtered lines (after removing repetition) 
        # first guess is to use switchs
        # in order to allow easy to add/remove a filter, but list of switchs too long would not be good

        # todo: not all reports will fit into this... Need a more abstract way
        if self.scapp_report != None:
            s_menu = ui.Growing_Menu("Filters")
            s_menu.collapsed=False

            unique_items = sorted(set(self.scapp_report['Angle']))
            s_menu.append(ui.Switch('filter_by_angle',self,label="by Angle"))
            s_menu.append(ui.Selector('angle',self,label='Angles',selection=[str(i) for i in unique_items] ))

            unique_items = sorted(set(self.scapp_report['ExpcResp']))
            s_menu.append(ui.Switch('filter_by_expresp',self,label="by Expected Response"))
            s_menu.append(ui.Selector('expected_response',self,label='Expected Response',selection=[str(i) for i in unique_items]))

            unique_items = sorted(set(zip(self.scapp_report['Angle'],self.scapp_report['X1'],self.scapp_report['Y1'])))
            s_menu.append(ui.Switch('filter_by_distance',self,label="by Distance"))
            s_menu.append(ui.Selector('distance',self,label='Distance',selection=[str(i) for i in unique_items]))

            s_menu.append(ui.Slider('onset',self,min=0.00,step=0.1,max=2.0,label='onset'))
            s_menu.append(ui.Slider('offset',self,min=0.00,step=0.1,max=2.0,label='offset'))
            s_menu.append(ui.Button('Add Events',self.add_filtered_events))
            s_menu.append(ui.Button('Clean, Add, Trim',self.clean_add_trim))
            self.menu.append(s_menu)

        s_menu = ui.Growing_Menu("Filters 2")
        s_menu.collapsed=False
        s_menu.append(ui.Selector('color',self,label='Color',selection=['red', 'blue'] ))

        #s_menu.append(ui.Switch('filter_by_expresp',self,label="by Expected Response"))
        #s_menu.append(ui.Selector('expected_response',self,label='Expected Response',selection=['0', '1'] ))
        s_menu.append(ui.Button('Add Events',self.add_filtered_events_npy))
        s_menu.append(ui.Button('Clean, Add, Trim',self.clean_add_trim_2))
        # self.menu.append(ui.Info_Text('Dispersion'))
        # 0, 1, 2, 3, 4 .. n

        self.menu.append(s_menu)

        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))

    def clean_add_trim(self):
        self.clean_custom_events()
        self.add_filtered_events()
        self.auto_trim()

    def clean_add_trim_2(self):
        self.clean_custom_events()
        self.add_filtered_events_npy()
        self.auto_trim()

    def trial_from_timestamp(self, timestamp):
        """
        timestamp: float
        Returns the nearest trial index associated with a given timestamp.
        """
        for i, trial in enumerate(self.scapp_output):
            trial_begin = float(trial[0][0])
            trial_end = float(trial[-1][0])
            if np.logical_and(trial_end >= timestamp, timestamp >= trial_begin):
            #if trial_end >= timestamp >= trial_begin:
                return i

    def clean_custom_events(self):
        self.custom_events = []

    def add_filtered_events_npy(self):
        def check_last_even():
            if not (len(self.custom_events) % 2) == 0:                
                begin = np.abs(self.g_pool.timestamps - np.float64(Trials[len(Trials)-1].TimeEvents[-1][0])).argmin()
                self.custom_events.append(begin)

        if self.scapp_output_npy:
            Trials = [TrialContainer() for _ in self.scapp_output_npy]
            
            # fill it with data
            if self.color == 'red':
                for n, trial in enumerate(Trials):
                    trial.TimeEvents = self.scapp_output_npy[n]
                    begin = np.abs(self.g_pool.timestamps - np.float64(Trials[n].TimeEvents[0][0])).argmin()
                    self.custom_events.append(begin)
                check_last_even()

            if self.color == 'blue':
                for n, trial in enumerate(Trials):
                    if n > 0:
                        trial.TimeEvents = self.scapp_output_npy[n]
                        begin = np.abs(self.g_pool.timestamps - np.float64(Trials[n].TimeEvents[0][0])).argmin()
                        self.custom_events.append(begin)
                check_last_even()
        else:
            logger.error("The scapp_output_npy data was not loaded.")

    def add_filtered_events(self):
        # create a container with the size of the total trials
        Trials = [TrialContainer() for _ in self.scapp_output]

        # fill it with some data
        for n, trial in enumerate(Trials):
            trial.ExpectedResponse = self.scapp_report[n]['ExpcResp']
            trial.Angle = str(self.scapp_report[n]['Angle'])
            trial.Distance = (self.scapp_report[n]['Angle'],self.scapp_report[n]['X1'],self.scapp_report[n]['Y1'])
            trial.TimeEvents = self.scapp_output[n]

            # find frame of correspondent event (firstResponse, starter onset...)
            firstResponse = np.abs(self.g_pool.timestamps - np.float64(Trials[n].TimeEvents[1][0])-self.onset).argmin()
            endLimitedHold = np.abs(self.g_pool.timestamps - np.float64(Trials[n].TimeEvents[-1][0])+self.offset).argmin()

            # conditions
            filtering_conditions = []
            if self.filter_by_expresp:
                filtering_conditions.append(str(trial.ExpectedResponse) == self.expected_response)

            if self.filter_by_angle:
                filtering_conditions.append(trial.Angle == self.angle)

            if self.filter_by_distance:
                filtering_conditions.append(str(trial.Distance) == self.distance)
         
            # add frames to the custom events if all conditions are true
            if filtering_conditions != []:
                if all(filtering_conditions):
                    self.custom_events.append(firstResponse)
                    self.custom_events.append(endLimitedHold)
            else:
                logger.warning("Check at least one filter condition before adding events.")
                
            # 2 seconds interval
            # frameInterval = range(firstResponse, endLimitedHold)
            # print firstResponse, endLimitedHold

    def on_window_resize(self,window,w,h):
        self.window_size = w,h
        self.h_pad = self.padding * self.frame_count/float(w)
        self.v_pad = self.padding * 1./h

    def update(self,frame,events):
        if self.frame_index != frame.index:
            self.frame_index = frame.index

    def gl_display(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(-self.h_pad,  (self.frame_count)+self.h_pad, -self.v_pad, 1+self.v_pad) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # custom events
        for e in self.custom_events:
            draw_polyline([(e,.06),(e,.005)], color = RGBA(.8, .8, .8, .8))

        size = len(self.custom_events)
        if size > 1:
            for i, e in enumerate(self.custom_events):
                draw_points([(e, .03)], size = 5, color = RGBA(.1, .5, .5, 1.)) 

            i = 0
            while True:
                if i == 0:
                    draw_polyline([(self.custom_events[i],.03),(self.custom_events[i+1],0.03)], color = RGBA(.8, .8, .8, .8))
                elif (i > 0) and (i < (size-1)):
                    draw_polyline([(self.custom_events[i] +1,.03),(self.custom_events[i+1],0.03)], color = RGBA(.8, .8, .8, .8))

                if 'chain' in self.mode:
                    i += 1
                elif 'in out pairs' in self.mode:
                    i += 2

                if i > (size-1):
                    break

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None
            
    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'custom_events':self.custom_events,
                'mode':self.mode,
                'keep_create_order':self.keep_create_order,
                'expected_response':self.expected_response,
                'filter_by_expresp':self.filter_by_expresp,
                'angle':self.angle,
                'filter_by_angle':self.filter_by_angle,
                'distance':self.distance,
                'filter_by_distance':self.filter_by_distance,
                'offset':self.offset,
                'onset':self.onset,
                'color':self.color}