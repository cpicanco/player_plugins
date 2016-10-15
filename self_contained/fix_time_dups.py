import numpy as np

def has_duplicates(timestamps):
	for tlow, thigh in zip(timestamps,timestamps[1:]):
		if tlow == thigh:
			print "DUPS!"
			return True


def remove_duplicates(timestamps):
	ntimestamps = []
	for tlow, thigh in zip(timestamps,timestamps[1:]):
		if tlow == thigh:
			tlow -= .000000001 
		ntimestamps.append(tlow)
	ntimestamps.append(timestamps[-1])
	return ntimestamps

if __name__ == '__main__':
	gaze_path = '/home/rafael/doutorado/data_org/P003/2015-05-20/000/gaze_coordenates_on_screen.txt'

	aGazeFile = np.genfromtxt(gaze_path, delimiter="\t",filling_values=None,names=True, autostrip=True, dtype=None)

	if has_duplicates(aGazeFile['time']):
		timestamps = remove_duplicates(aGazeFile['time'])
		print has_duplicates(timestamps)
		print timestamps[-1]
		print aGazeFile['time'][-1]
		print len(timestamps), len(aGazeFile['time'])
	else:
		print 'No dups!'