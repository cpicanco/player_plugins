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

def copy_and_format(pupil_data_directory):
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

				convert(src_base, dst_base)
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
	print 'origin:',os.path.dirname(os.path.abspath(__file__))

	if len(sys.argv) > 1:
		source_directories = [directory for directory in sys.argv[1:] if os.path.exists(directory)]
	else:
		source_directories = ['/home/pupil/_rafael/data_doc/007-Gabriel/2015-05-20']

	for d in source_directories:
		copy_and_format(d)
