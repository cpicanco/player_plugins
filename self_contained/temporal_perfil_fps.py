# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import os
import matplotlib.pyplot as plt
import numpy as np
from methods import stimuli_onset
from temporal_perfil import plot_temporal_perfil

def load_npdata_from_path(path):
	beha_events_path = os.path.join(path, "behavioral_events.txt")
	gaze_events_path = os.path.join(path, "gaze_coordenates_on_screen.txt")

	if not os.path.isfile(beha_events_path):
		raise IOError, "behavioral_events were not found."

	if not os.path.isfile(gaze_events_path):
		raise IOError, "gaze_coordenates were not found."

	be = np.genfromtxt(beha_events_path, delimiter="\t",missing_values=["NA"],
	  filling_values=None,names=True, autostrip=True, dtype=None)

	ge = np.genfromtxt(gaze_events_path, delimiter="\t",missing_values=["NA"],
	  filling_values=None,names=True, autostrip=True, dtype=None)

	return be, ge

if __name__ == '__main__':
	rpath = '/home/pupil/_rafael/data_doc/011-Priscila/2015-05-26/raw_data_organized/'
	paths = [
		os.path.join(rpath, "000"),
		os.path.join(rpath, "001"),
		os.path.join(rpath, "002")
		#os.path.join(rpath, "003")
	]

	x_label = 'Time block'
	y_label = 'Response rate'
	title = 'Fps by time block'

	n_plots = len(paths)
	if n_plots == 1:
		figsize = (6, 4)
	elif n_plots == 2:
		figsize = (11, 4)
	else:
		figsize = (16, 4)

	# figure.add_axes([0.1, 0.1, 0.8, 0.8], frameon = 0)
	figure, axarr = plt.subplots(1, n_plots, sharey=True, sharex=False, figsize=figsize) 
	figure.suptitle(title);
	figure.text(0.5, 0.02, x_label)

	for i, path in enumerate(paths):
		be, ge = load_npdata_from_path(path)
		plot_temporal_perfil(axarr[i], stimuli_onset(be), ge['time'])

	axarr[0].set_ylabel(y_label)
	#axarr.legend(loc=(0.0,0.73))

	# axarr[1].set_xlabel(x_label)
	plt.ylim(ymin = 0)

	figure.subplots_adjust(wspace=0.1,left=0.05, right=.98,bottom=0.1,top=0.92)
	#figure.tight_layout()
	plt.show()

	# print all column names
	# for key in be.dtype.names:
	#   print key

	# figure, axarr = plt.subplots(1, 1, sharey=True, sharex=False, figsize=(6, 4)) 
	# figure.suptitle('title');

	# axarr.plot(ge['x_scaled'], ge['y_scaled'], 'k.')

	# plt.show()