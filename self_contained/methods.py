# -*- coding: utf-8 -*-
'''
	Copyright (C) 2016 Rafael Picanço.

	The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

	You should have received a copy of the GNU General Public License
	along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import numpy as np
import math

def get_pixels_per_degree(sw_px,sh_px,sw_d,sh_d):
	return np.sqrt((sw_px**2)+(sh_px**2))/np.sqrt((sw_d**2)+(sh_d**2))

def get_values_per_degree(sw_d,sh_d):
	return np.sqrt((1**2)+(1**2))/np.sqrt((sw_d**2)+(sh_d**2))

# http://en.wikipedia.org/wiki/Visual_angle
def get_visual_angle(sw, sd):
	V = 2 * math.atan(sw/(sd*2))
	# print 'Radians:', V
	degrees = math.degrees(V)

	# x, y
	return degrees, (764*degrees)/1280

import constants as K

def normalized_to_pixel(gp):
	"""
	gp:numpy.array.shape(x, 2)
	"""
	gp[:,0] *= K.SCREEN_WIDTH_PX
	gp[:,1] *= K.SCREEN_HEIGHT_PX
	return gp

def pixel_to_degree(gp):
	"""
	gp:numpy.array.shape(x, 2)
	"""
	gp[:,0] /= K.PIXELS_PER_DEGREE
	gp[:,1] /= K.PIXELS_PER_DEGREE
	return gp

# def normalized_to_degree(gp):
#     """
#     gp:numpy.array.shape(x, 2)
#     """
#     values_per_degree = get_values_per_degree(K.SCREEN_WIDTH_DEG,K.SCREEN_HEIGHT_DEG)
#     gp[:,0] /= values_per_degree
#     gp[:,1] /= values_per_degree
#     return gp

def move_mean_to_zero(gp):
	"""
	gp:numpy.array.shape(x, 2)
	"""
	MX = np.mean(gp[:,0])
	MY = np.mean(gp[:,1])
	gp[:,0] = MX - gp[:,0]
	gp[:,1] = MY - gp[:,1]
	return gp

def root_mean_square(gp):
	"""
	gp:numpy.array.shape(x, 2)
	"""
	RMSX = np.sqrt(np.mean(gp[:,0]**2))
	RMSY = np.sqrt(np.mean(gp[:,1]**2))
	# return np.sqrt((RMSX**2)+(RMSY**2))
	return np.sqrt(np.mean(gp**2)), RMSX,RMSY

# stimuli timestamps
def color_pair(behavioral_data, pair):  
	"""
		behavioral_data: np.genfromtxt object; "behavioral_events.txt" as path
	"""
	def all_events(string):
		return [line['time'] for line in behavioral_data if line['event'] == string]
		
	return [[all_events('1a'), all_events('1b')],
	        [all_events('1b'), all_events('2a')],
	        [all_events('2a'), all_events('2b')],
	        [all_events('2b'), all_events('1a')[1:]]][pair]

def stimuli_onset(behavioral_data):  
	"""
		behavioral_data: np.genfromtxt object; "behavioral_events.txt" as path
	"""
	def all_events(string):
		return [line['time'] for line in behavioral_data if line['event'] == string]
		
	return [all_events('1a'), all_events('2a')] # [[R1,R2,R3,..],[B1,B2,B3,..]] 

def all_stimuli(behavioral_data):
	"""
		behavioral_data: np.genfromtxt object; "behavioral_events.txt" as path
	"""
	return [line['time'] for line in behavioral_data if line['event_type'] == 'stimulus']

# responses timestamps
def all_responses(behavioral_data): 
	"""
		behavioral_data: np.genfromtxt object; "behavioral_events.txt" as path
	"""
	return [line['time'] for line in behavioral_data if line['event_type'] == 'response']