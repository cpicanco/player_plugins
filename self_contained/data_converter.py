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
from fix_time_dups import has_duplicates, remove_duplicates

def convert_gaze(src, dst,time_start):
	with open(dst, 'w+') as f:
	  f.write("\t".join(('time','x_norm','y_norm'))+'\n')
	  for timestamp, X, Y in zip(src['gaze_timestamp'], src['x_norm'], src['y_norm']):
		  timestamp -= time_start
		  timestamp = '%.3f'%timestamp
		  X = '%.3f'%round(X, 3)
		  Y = '%.3f'%round(Y, 3)
		  f.write("\t".join((timestamp, X, Y))+'\n')


def convert(src, dst, fix_time_dups=False):
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

	if fix_time_dups:
		if has_duplicates(aGazeFile['gaze_timestamp']):
			aGazeFile['gaze_timestamp'] = remove_duplicates(aGazeFile['gaze_timestamp'])

		if has_duplicates(aFixationFile['start_timestamp']):
			aFixationFile['start_timestamp'] = remove_duplicates(aFixationFile['start_timestamp'])

	time_start = aGazeFile['gaze_timestamp'][0]

	print repr(aGazeFile['gaze_timestamp'][0])
    
	# gaze events
	convert_gaze(aGazeFile,gaze_output,time_start)

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
				timestamp = '%.9f'%(line[1].astype('float64')-time_start)
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
				timestamp = '%.9f'%(float(timestamp)-time_start) 
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

				f.write("\t".join((timestamp, event_type, event, left, right, bottom, top))+'\n')

	# sort if necessary
	# for l in sorted(reader, key=operator.itemgetter(0), reverse=False): # http://stackoverflow.com/a/2100384	
	# 	print l

	# fixations
	with open(fixa_output, 'w+') as f:
	  f.write("\t".join(('id','start_time', 'duration','norm_pos_x','norm_pos_y'))+'\n')
	  c = zip(aFixationFile['id'],aFixationFile['start_timestamp'],aFixationFile['duration'],aFixationFile['norm_pos_x'],aFixationFile['norm_pos_y'])
	  for fid, timestamp, duration, X, Y in c:
		  timestamp -= time_start
		  duration = '%.3f'%round(duration, 3)
		  X = '%.3f'%round(X, 3)
		  Y = '%.3f'%round(Y, 3)
		  f.write("\t".join((str(fid),timestamp, duration, X, Y))+'\n')

if __name__ == '__main__':
	gaze_input ='/home/rafael/doutorado/data_doc/003-Natan/2015-05-13/exports/0-11710/surfaces/gaze_positions_on_surface_Screen_1455571281.87.csv'
	gaze_input = np.genfromtxt(gaze_input, delimiter="\t",missing_values=["NA"],
		  filling_values=None,names=True, autostrip=True, dtype=None)

	gaze_output = '/home/rafael/git/abpmc-2016/P000/000/gaze_coordenates_on_screen.txt'
	time_start = gaze_input['gaze_timestamp'][0]
	convert_gaze(gaze_input, gaze_output, time_start)