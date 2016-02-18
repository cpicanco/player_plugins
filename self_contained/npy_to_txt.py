# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import numpy as np

data = np.load("/home/rafael/data_doc/004-Cristiane/2015-05-27/001/pupil_positions.npy")
np.savetxt("pupil_positions.txt",data )