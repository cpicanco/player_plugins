import numpy as np

data = np.load("/home/rafael/data_doc/004-Cristiane/2015-05-27/001/pupil_positions.npy")
np.savetxt("pupil_positions.txt",data )