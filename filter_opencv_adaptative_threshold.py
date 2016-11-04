# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import cv2
from pyglui import ui

from plugin import Plugin

blue, green, red = 0, 1, 2

class Filter_Opencv_Adaptative_Threshold(Plugin):
	"""
		Apply cv2.adaptativeThreshold in each channel of the (world) frame.img
	"""
	uniqueness = "not_unique"
	def __init__(self, g_pool, threshold=255, thresh_mode="BINARY",
		adaptive_method="GAUSSIAN", block_size=5,constant=-1,blur=3):
		super(Filter_Opencv_Adaptative_Threshold, self).__init__(g_pool)
		# run before all plugins
		# self.order = .1

		# run after all plugins
		self.order = .99

		# initialize empty menu
		self.menu = None

		# filter properties
		self.threshold = threshold
		self.thresh_mode = thresh_mode
		self.adaptive_method = adaptive_method
		self.block_size = block_size
		self.constant = constant
		self.blur = blur

	def update(self,frame,events):
		# thresh_mode
		if self.thresh_mode == "NONE":
			return

		if self.thresh_mode == "BINARY":
			cv2_thresh_mode = cv2.THRESH_BINARY

		if self.thresh_mode == "BINARY_INV":
			cv2_thresh_mode = cv2.THRESH_BINARY_INV

		if self.adaptive_method == "MEAN":
			cv2_adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

		if self.adaptive_method == "GAUSSIAN":
			cv2_adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

		# apply the threshold to each channel
		for i, channel in enumerate((frame.img[:,:,blue], frame.img[:,:,green], frame.img[:,:,red])):
			if self.blur > 1:
				channel = cv2.GaussianBlur(channel,(self.blur,self.blur),0)

			edg = cv2.adaptiveThreshold(channel,
									maxValue=self.threshold,
									adaptiveMethod = cv2_adaptive_method,
									thresholdType = cv2_thresh_mode,
									blockSize = self.block_size,
									C = self.constant)
			frame.img[:,:,i] = edg


	def init_gui(self):
		# initialize the menu
		self.menu = ui.Scrolling_Menu('Adaptative Threshold')

		# add menu to the window
		self.g_pool.gui.append(self.menu)

		# append elements to the menu
		self.menu.append(ui.Button('remove',self.unset_alive))
		self.menu.append(ui.Info_Text('Filter Properties'))
		self.menu.append(ui.Selector('thresh_mode',self,label='Thresh Mode',selection=["NONE","BINARY","BINARY_INV"] ))
		self.menu.append(ui.Selector('adaptive_method',self,label='Adaptive Method',selection=["GAUSSIAN","MEAN"] ))
		
		self.menu.append(ui.Slider('threshold',self,min=0,step=1,max=255,label='Threshold'))
		self.menu.append(ui.Slider('block_size',self,min=3,step=2,max=55,label='Block Size'))
		self.menu.append(ui.Slider('constant',self,min=-30,step=1,max=30,label='Constant'))
		self.menu.append(ui.Slider('blur',self,min=1,step=2,max=55,label='Blur'))

	def deinit_gui(self):
		if self.menu:
			self.g_pool.gui.remove(self.menu)
			self.menu = None

	def unset_alive(self):
		self.alive = False

	def get_init_dict(self):
		# persistent properties throughout sessions
		return {'threshold':self.threshold,
				'thresh_mode':self.thresh_mode,
				'adaptive_method':self.adaptive_method,
				'block_size':self.block_size,
				'constant':self.constant,
				'blur':self.blur}

	def cleanup(self):
		""" called when the plugin gets terminated.
		This happens either voluntarily or forced.
		if you have a GUI or glfw window destroy it here.
		"""
		self.deinit_gui()