# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import sys
import os
import fnmatch

# from shutil import copyfile
from glob import glob, iglob

from data_converter import convert

# this script does not delete nor overrides any source data
# it only creates a folder (raw_data_organized) with interest formated data copied from the source location
# it DOES overrides any previous copies in the dst location 
# it must be run after pupil surface data have been exported 

def copy_and_format(pupil_data_directory, destination_directory=None):
	if destination_directory:
		destination = destination_directory
	else:
		destination = os.path.join(pupil_data_directory,'raw_data_organized')

	print 'base:',pupil_data_directory
	print 'destination:',destination

	surface_d = None
	directories = sorted(glob(os.path.join(pupil_data_directory,'0*')))
	for d in directories:
		basename = os.path.basename(d)
		raw_dirs = []
		for root, dirnames, filenames in os.walk(d):
			for dirname in fnmatch.filter(dirnames, '*surfaces'):
				raw_dirs.append(os.path.join(root, dirname))
		for r_d in raw_dirs:
			if r_d == []:
				print 'Warning:', r_d, ' has no surface folder'
			else:
				src_base = [r_d, d]
				dst_base  = os.path.join(destination,basename)

				if not os.path.exists(dst_base):
					try:
						os.makedirs(dst_base)
					except OSError as exc:
						if exc.errno != errno.EEXIST:
							raise

				convert(src_base, dst_base, True)
				# src_files = [
				# 			glob(os.path.join(r_d, 'fixations_on_surface_Screen*'))[0],
				# 			glob(os.path.join(r_d, 'gaze_positions_on_surface_Screen*'))[0],
				# 			glob(os.path.join(d, 'scapp_output.timestamps'))[0],
				# 			glob(os.path.join(d, 'scapp_output.npy'))[0]
				# 			]

				# dst_files = [
				# 			os.path.join(dst_base, 'fixations_on_surface_Screen.csv'),
				# 			os.path.join(dst_base, 'gaze_positions_on_surface_Screen.csv'),
				# 			os.path.join(dst_base, 'scapp_output.timestamps'),
				# 			os.path.join(dst_base, 'scapp_output.npy')
				# 			]

				# for src, dst in zip(src_files, dst_files):
				# 	print 'from:', src
				# 	print '__to:', dst

				# 	# http://stackoverflow.com/a/12517490
				# 	if not os.path.exists(os.path.dirname(dst)):
				# 		try:
				# 			os.makedirs(os.path.dirname(dst))
				# 		except OSError as exc:
				# 			if exc.errno != errno.EEXIST:
				# 				raise
				# 	if not os.path.exists(dst):
				# 		copyfile(src, dst)



if __name__ == '__main__':
	# if len(sys.argv) > 1:
	# 	source_directories = [directory for directory in sys.argv[1:] if os.path.exists(directory)]
	# else:
	print 'origin:',os.path.dirname(os.path.abspath(__file__))

	source_dir = '/home/rafael/doutorado/data_doc/'
	inner_paths = [
		'004-Cristiane/2015-05-19',
		'004-Cristiane/2015-05-27',
		'005-Marco/2015-05-19',
		'005-Marco/2015-05-20',
		'006-Renan/2015-05-20',
		'007-Gabriel/2015-05-20',
		'008-Thaiane/2015-05-19',
		'009-Rebeca/2015-05-25',
		'010-Iguaracy/2015-05-25',
		'011-Priscila/2015-05-26',
		'013-Oziele/2015-05-26',
		'014-Acsa/2015-05-26'
	]
	source_directories = [os.path.join(source_dir,s) for s in inner_paths]

	source_dir = '/home/rafael/doutorado/data_org/'
	inner_paths = [
		'P001/2015-05-19',
		'P001/2015-05-27',
		'P002/2015-05-19',
		'P002/2015-05-20',
		'P003/2015-05-20',
		'P004/2015-05-20',
		'P005/2015-05-19',
		'P006/2015-05-25',
		'P007/2015-05-25',
		'P008/2015-05-26',
		'P009/2015-05-26',
		'P010/2015-05-26'
	]
	destinat_directory = [os.path.join(source_dir,s) for s in inner_paths]

	for s, d in zip(source_directories, destinat_directory):
		copy_and_format(s,d)