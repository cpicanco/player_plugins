# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import os
import operator
import csv
import numpy as np
from ast import literal_eval
from glob import glob

def convert(src, dst):
	# output
	beha_output = os.path.join(dst, 'behavioral_events.txt')
	gaze_output = os.path.join(dst, 'gaze_coordenates_on_screen.txt')
	fixa_output = os.path.join(dst, 'fixations_on_screen.txt')

	# input
	scapp_timestamps_path = os.path.join(src[1], "scapp_output.timestamps")
	stimuli_path = os.path.join(src[1], 'scapp_output.npy')
	gaze = glob(os.path.join(src[0],'gaze_positions_on_surface*'))[0]
	fixa = glob(os.path.join(src[0],'fixations_on_surface*'))[0]

	aGazeFile = np.genfromtxt(gaze, delimiter="\t",missing_values=["NA"],
		  filling_values=None,names=True, autostrip=True, dtype=None)

	aFixationFile = np.genfromtxt(fixa, delimiter="\t",missing_values=["NA"],
	      filling_values=None,names=True, autostrip=True, dtype=None)

	time_start = np.min(aGazeFile['gaze_timestamp'])

	# gaze events
	with open(gaze_output, 'w+') as f:
	  f.write("\t".join(('time','x_norm','y_norm'))+'\n')
	  for line in aGazeFile:
		  timestamp = '%.3f'%round(line[0]-time_start, 3)
		  X = '%.3f'%round(line[3], 3)
		  Y = '%.3f'%round(line[4], 3)
		  f.write("\t".join((timestamp, X, Y))+'\n')

	# behavioral events

	# stimuli
	width,height = 130, 130
	left, top, right, bottom = 'NA', 'NA', 'NA', 'NA'

	with open(scapp_timestamps_path, 'r') as t:
		timestamps2 = np.load(stimuli_path)
		with open(beha_output, 'w+') as f:
			f.write("\t".join(('time','event_type','event','left', 'right', 'bottom', 'top'))+'\n')
			
			# stimuli events
			for line in timestamps2:
				trial_no = line[0] 
				timestamp = '%.3f'%round(line[1].astype('float64')-time_start, 3)
				event = line[2]

				if event == '1':
					event = '1a'

				if event == '2':
					event = '2a'

				if 'a' in event:
					event_type = 'stimulus'
					left, top = 362, 319
					right, bottom = left+width, top+height

				elif 'b' in event:
					event_type = 'stimulus'
					left, top = 789, 319
					right, bottom = left+width, top+height

				if not 'NA' == left:
					left = left/1280.0

				if not 'NA' == right:
					right = right/1280.0
				
				if not 'NA' == top:
					top = top/764.0
					top = 1-top
				
				if not 'NA' == bottom:
					bottom = bottom/764.0
					bottom = 1-bottom

				left = '%.3f'%round(left, 3)
				right = '%.3f'%round(right, 3)
				top = '%.3f'%round(top, 3)
				bottom = '%.3f'%round(bottom, 3)
				
				f.write("\t".join((timestamp, event_type, event, left, right, bottom, top))+'\n')

			# responses and virtual events
			for line in t:
				(trial_no, timestamp, event_s) = literal_eval(line)
				left, top, right, bottom = 'NA', 'NA', 'NA', 'NA'
				timestamp = round(float(timestamp)-time_start, 3) 
				event = line[1]

				if 'S' in event_s:
					event = 'S'
					event_type = 'virtual'
				elif 'E' in event_s:
					event = 'E'
					event_type = 'virtual'
				elif 'R' in event_s:
					event = 'R'
					event_type = 'response'
				else:
					continue

				f.write("\t".join((str(timestamp), event_type, event, left, right, bottom, top))+'\n')

	# sort if necessary
	# for l in sorted(reader, key=operator.itemgetter(0), reverse=False): # http://stackoverflow.com/a/2100384	
	# 	print l

	# fixations
	with open(fixa_output, 'w+') as f:
	  f.write("\t".join(('id','start_time', 'duration','norm_pos_x','norm_pos_y'))+'\n')
	  for line in aFixationFile:
		  timestamp = round(line[1]-time_start, 3)
		  duration = round(line[2], 3)
		  x = round(line[5], 3)
		  y = round(line[6], 3)
		  f.write("\t".join((str(line[0]),'%.3f'%timestamp, '%.3f'%duration, '%.3f'%x, '%.3f'%y))+'\n')