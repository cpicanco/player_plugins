# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
#from sklearn.preprocessing import StandardScaler
from glob import glob
from methods import stimuli_onset, all_stimuli,color_pair
from temporal_perfil import plot_temporal_perfil, draw_all

def categorize_points(src_xy, eps = 0.06, min_samples = 1000, return_dict=False):
	# so far no need for scaling, our data is assumed to be gaussian and normalized
	# src_xy = StandardScaler().fit_transform(src_xy)
	dbsc = DBSCAN(eps=eps, min_samples=min_samples).fit(src_xy)

	# return a dictionary with clusters and noises
	if return_dict:
		dictionary = {}
		labels = dbsc.labels_
		core_samples_mask = np.zeros_like(labels, dtype = bool)
		core_samples_mask[dbsc.core_sample_indices_] = True

		# Black removed and is used for noise instead.
		unique_labels = set(labels)
		colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
		for k, col in zip(unique_labels, colors):
			if k == -1:
				# Black used for noise.
				col = 'k'

			class_member_mask = (labels == k)

			# cluster
			xy = src_xy[class_member_mask & core_samples_mask]
			dictionary['cluster_'+str(k)] = xy

			# noise
			xy = src_xy[class_member_mask & ~core_samples_mask]
			dictionary['noise_'+str(k)] = xy

		return dictionary
	else:
		# return the raw data and the DBSCAN object instead
		return dbsc

def categorize_timestamps(src_timestamps, dbsc):
	clusters = {}
	labels = dbsc.labels_
	core_samples_mask = np.zeros_like(labels, dtype = bool)
	core_samples_mask[dbsc.core_sample_indices_] = True

	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = 'k'

		class_member_mask = (labels == k)

		# cluster
		timestamps = src_timestamps[class_member_mask & core_samples_mask]
		clusters['cluster_'+str(k)] = timestamps

		# noise
		timestamps = src_timestamps[class_member_mask & ~core_samples_mask]
		clusters['noise_'+str(k)] = timestamps
	return clusters

def categorize_masks(src_timestamps, dbsc):
	masks = {}
	labels = dbsc.labels_
	core_samples_mask = np.zeros_like(labels, dtype = bool)
	core_samples_mask[dbsc.core_sample_indices_] = True

	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = 'k'

		class_member_mask = (labels == k)

		# cluster
		masks['cluster_'+str(k)] = class_member_mask & core_samples_mask

		# noise
		masks['noise_'+str(k)] = class_member_mask & ~core_samples_mask
	return masks

def plot_dbscan(src_xy, dbsc, doplot=False):
	dictionary = {}
	labels = dbsc.labels_
	core_samples_mask = np.zeros_like(labels, dtype = bool)
	core_samples_mask[dbsc.core_sample_indices_] = True

	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	print "clusters:",n_clusters_

	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	print "labels:",unique_labels
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = 'k'

		class_member_mask = (labels == k)

		# clusters
		xy = src_xy[class_member_mask & core_samples_mask]
		if doplot:
			plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=3, label=str(k))
		dictionary['cluster_'+str(k)] = xy

		# noise
		xy = src_xy[class_member_mask & ~core_samples_mask]
		if doplot:
			plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=1)
	if doplot:
		axes = plt.gca()
		axes.set_ylim(ymax = 1, ymin = 0)
		axes.set_xlim(xmax = 1, xmin = 0)
		axes.legend()
		# plt.axis('equal')
		plt.title('')
		plt.show()
	return dictionary

def draw_single(src_dir, show=False):
# Targets
	BlueLeft = '#011efe'
	RedLeft = '#fe0000'

	# Distractors
	GreenRight = '#49ac15' 
	CianoRight = '#a8e6cf' 

	ID = os.path.basename(os.path.dirname(src_dir))
	basepath = os.path.dirname(os.path.dirname(src_dir))

	print src_dir, os.path.join(basepath,'P005/2015-05-19')

	if src_dir == os.path.join(basepath,'P001/2015-05-19'):
		data = [{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':1000}]

	elif src_dir == os.path.join(basepath,'P001/2015-05-27'):
		data = [{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':1000}]

	elif src_dir == os.path.join(basepath,'P002/2015-05-19'):
		data = [{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':1000}]

	elif src_dir == os.path.join(basepath,'P002/2015-05-20'):
		data = [{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':500}]

	elif src_dir == os.path.join(basepath,'P003/2015-05-20'):
		data = [{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':800}]

	elif src_dir == os.path.join(basepath,'P004/2015-05-20'):
		data = [{'eps':0.06, 'min_samples':1500},
						{'eps':0.06, 'min_samples':1500},
						{'eps':0.06, 'min_samples':1500}]

	elif src_dir == os.path.join(basepath,'P005/2015-05-19'):
		data = [{'eps':0.08, 'min_samples':500},
						{'eps':0.07, 'min_samples':500},
						{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':1000}]

	elif src_dir == os.path.join(basepath,'P006/2015-05-25'):
		data = [{'eps':0.065, 'min_samples':1700},
						{'eps':0.065, 'min_samples':1300},
						{'eps':0.065, 'min_samples':1200}]


	elif src_dir == os.path.join(basepath,'P007/2015-05-25'):
		data = [{'eps':0.06, 'min_samples':1200},
						{'eps':0.06, 'min_samples':1200},
						{'eps':0.06, 'min_samples':1200}]

	elif src_dir == os.path.join(basepath,'P008/2015-05-26'):
		data = [{'eps':0.06, 'min_samples':1200},
						{'eps':0.06, 'min_samples':1200},
						{'eps':0.06, 'min_samples':1200}]

	elif src_dir == os.path.join(basepath,'P009/2015-05-26'):
		data = [{'eps':0.06, 'min_samples':1700},
						{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':1000},
						{'eps':0.06, 'min_samples':1000}]

	elif src_dir == os.path.join(basepath,'P010/2015-05-26'):
		data = [{'eps':0.06, 'min_samples':600},
						{'eps':0.06, 'min_samples':600},
						{'eps':0.06, 'min_samples':600},
						{'eps':0.06, 'min_samples':600}]

	paths = sorted(glob(os.path.join(src_dir,'0*')))
	for i, path in enumerate(paths):	
		data_folder = os.path.join(src_dir, path)
		beha_events_path = os.path.join(data_folder, "behavioral_events.txt")
		gaze_events_path = os.path.join(data_folder, 'gaze_coordenates_on_screen.txt')
		
		gaze_data = np.genfromtxt(gaze_events_path, delimiter='\t',missing_values=['NA'],
			filling_values=None,names=True, autostrip=True, dtype=None)

		data[i]['beha_data'] = np.genfromtxt(beha_events_path, delimiter="\t",missing_values=["NA"],
			filling_values=None,names=True, autostrip=True, dtype=None)

		# DBSCAN expects data with shape (-1,2), we need to transpose ours first
		data[i]['src_xy'] = np.array([gaze_data['x_norm'], gaze_data['y_norm']]).T

		data[i]['dbsc'] = categorize_points(data[i]['src_xy'], data[i]['eps'], data[i]['min_samples'])
		data[i]['points_categorized'] = plot_dbscan(data[i]['src_xy'], data[i]['dbsc'])

		data[i]['masks'] = categorize_masks(data[i]['src_xy'], data[i]['dbsc'])

		data[i]['src_timestamps'] = gaze_data['time']
		data[i]['time_categorized'] = categorize_timestamps(data[i]['src_timestamps'],data[i]['dbsc'])


	x_label = 'Ciclo'
	y_label = 'dir. < Taxa (gaze/s) > esq.'
	title = 'Particip. '+ID+': Taxa de movim. oculares durante cada cor'

	n_plots = len(data)
	if n_plots == 1:
		figsize = (4, 4)
	elif n_plots == 2:
		figsize = (8, 4)
	elif n_plots == 3:
		figsize = (12, 4)
	else:
		figsize = (14, 4)

	# figure.add_axes([0.1, 0.1, 0.8, 0.8], frameon = 0)
	figure, axarr = plt.subplots(1, n_plots, sharey=True, sharex=False, figsize=figsize) 
	figure.suptitle(title);
	figure.text(0.5, 0.02, x_label)

	# look rate at left 
	# look rate at right
	
	for i, d in enumerate(data):
		turnover_count = 0
		turnover = [a for a,b in zip(d['masks']['cluster_0'],d['masks']['cluster_1']) if a or b]
		for c, n in zip(turnover,turnover[1:]):
			if c != n:
				turnover_count += 1
		print '\n','turnover_count:',turnover_count


		left_right_xy = []
		left_right_timestamps = []
		for time, points in zip (d['time_categorized'].iteritems(),d['points_categorized'].iteritems()):
			_, xy = points
			time_key, timestamps = time
			if len(timestamps) > 0 and 'cluster' in time_key:
				left_right_timestamps.append(timestamps)
				left_right_xy.append(xy)

		if np.mean(left_right_xy[0][0]) > np.mean(left_right_xy[1][0]):
			left_right_xy = [left_right_xy[1],left_right_xy[0]]
			left_right_timestamps = [left_right_timestamps[1],left_right_timestamps[0]]

		# all stimuli, left and right
		#plot_temporal_perfil(axarr[i],all_stimuli(data[i]['beha_data']), left_right_timestamps,"positions")

		# v+v, v+c, a+v, a+c
		plot_temporal_perfil(axarr[i],color_pair(data[i]['beha_data'],0), left_right_timestamps[0],'pair', c1=RedLeft, nsize=0)
		plot_temporal_perfil(axarr[i],color_pair(data[i]['beha_data'],0), left_right_timestamps[1],'pair', c1=GreenRight, nsize=0, doreversed=True)

		plot_temporal_perfil(axarr[i],color_pair(data[i]['beha_data'],1), left_right_timestamps[0],'pair', c1=RedLeft, nsize=1)
		plot_temporal_perfil(axarr[i],color_pair(data[i]['beha_data'],1), left_right_timestamps[1],'pair', c1=CianoRight, nsize=1, doreversed=True)

		plot_temporal_perfil(axarr[i],color_pair(data[i]['beha_data'],2), left_right_timestamps[0],'pair', c1=BlueLeft, nsize=2)
		plot_temporal_perfil(axarr[i],color_pair(data[i]['beha_data'],2), left_right_timestamps[1],'pair', c1=CianoRight, nsize=2, doreversed=True)
						
		plot_temporal_perfil(axarr[i],color_pair(data[i]['beha_data'],3), left_right_timestamps[0],'pair', c1=BlueLeft, nsize=3)
		plot_temporal_perfil(axarr[i],color_pair(data[i]['beha_data'],3), left_right_timestamps[1],'pair', c1=GreenRight, nsize=3, doreversed=True)
		
		# all stimuli (red blue), left
		# plot_temporal_perfil(axarr[i],stimuli_onset(data[i]['beha_data']), left_right_timestamps[0],'colors', c1=RedLeft, c2=BlueLeft)
	
		# all stimuli (red blue), right
		# plot_temporal_perfil(axarr[i],stimuli_onset(data[i]['beha_data']), left_right_timestamps[1],"colors",c1=RedRight,c2=BlueRight, doreversed=True)
	ticks = [30,20,10,0,10,20,30]
	axarr[0].set_yticklabels(labels=ticks)
	#([x for x in range(-30,31)],)

	plt.ylim(ymin = -30)
	plt.ylim(ymax = 30)


	# figure.subplots_adjust(wspace=0.1,left=0.05, right=.98,bottom=0.1,top=0.92)
	axarr[0].set_ylabel(y_label)
	figure.tight_layout()
	if show:			
		plt.show()

if __name__ == '__main__':
	font = {'family' : 'serif',
					'size'   : 16}

	matplotlib.rc('font', **font)
	#matplotlib.rcParams.update({'font.size': 22})

	draw_all('/home/rafael/git/abpmc-2016/',draw_single,'taxa_movimentos_oculares_A.png')

	#draw_single('/home/rafael/git/abpmc-2016/P005/2015-05-19', True)


# note: fps should be as constant as possible