# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import os

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def categorize_points(src_xy, return_dict=False):
	# so far no need for scaling, our data is assumed to be gaussian and normalized
	# src_xy = StandardScaler().fit_transform(src_xy)
	dbsc = DBSCAN(eps = 0.035, min_samples = 500).fit(src_xy)

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

def plot_dbscan(src_xy, dbsc):
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
		plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=3)

		# noise
		xy = src_xy[class_member_mask & ~core_samples_mask]
		plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=1)

	# axes = plt.gca()
	# axes.set_ylim(ymax = 1, ymin = 0)
	# axes.set_xlim(xmax = 1, xmin = 0)
	plt.axis('equal')
	plt.title('')
	plt.show()



def plot_temporal_perfil(axis,onsets,gtime):
	red_onset = onsets[0]
	blu_onset = onsets[1]
	g_rate = []
	for red, blue in zip(red_onset,blu_onset):
		g_inside = []
		for g in ge['time']:
			if (g >= red) and (g <= blue):
				g_inside.append(g)

		g_rate.append(len(g_inside)/(blue-red))


	# the actual data
	axis.plot(g_rate,color='red',label='Red')

	del red_onset[0]

	g_rate = []
	for red, blue in zip(red_onset,blu_onset):
		g_inside = []
		for g in ge['time']:
			if (g >= blue) and (g <= red):
				g_inside.append(g)

		g_rate.append(len(g_inside)/(red-blue))

	axis.plot(g_rate,color='blue',label='Blue')

	# remove frame
	axis.spines['top'].set_visible(False)
	axis.spines['bottom'].set_visible(False)
	axis.spines['left'].set_visible(False)
	axis.spines['right'].set_visible(False)

	#remove ticks
	axis.xaxis.set_ticks_position('none')
	axis.yaxis.set_ticks_position('none') 

if __name__ == '__main__':
	root = '/home/pupil/_rafael/data_doc/013-Oziele/2015-05-26/raw_data_organized'
	data_folder = os.path.join(root, '002')
	beha_events_path = os.path.join(data_folder, "behavioral_events.txt")
	gaze_events_path = os.path.join(data_folder, 'gaze_coordenates_on_screen.txt')
	
	gaze_data = np.genfromtxt(gaze_events_path, delimiter='\t',missing_values=['NA'],
	  filling_values=None,names=True, autostrip=True, dtype=None)

	beha_data = np.genfromtxt(beha_events_path, delimiter="\t",missing_values=["NA"],
	  filling_values=None,names=True, autostrip=True, dtype=None)

	# DBSCAN expects data with shape (-1,2), we need to transpose ours first
	src_xy = np.array([gaze_data['x_norm'], gaze_data['y_norm']])
	src_xy = src_xy.T

	dbsc = categorize_points(src_xy)
	#plot_dbscan(src_xy, dbsc)

	# clusters = categorize_points(src_xy, True)
	# for key, value in clusters.iteritems():
	# 	print key, ':', len(value)
	# 	if len(value) > 0:

	# 		# for normalized 2d data with p(x, y) where 1 => x, y >= 0
	# 		axes = plt.gca()
	# 		axes.set_ylim(ymax = 1, ymin = 0)
	# 		axes.set_xlim(xmax = 1, xmin = 0)
	# 		plt.plot(value[:,0], value[:,1], 'o')
	# 		plt.title(key)
	# 		plt.show()

	src_timestamps = gaze_data['time']
	time_categorized = categorize_timestamps(src_timestamps,dbsc)

	for key, value in time_categorized.iteritems():
		print key, ':', len(value)
		# if len(value) > 0: