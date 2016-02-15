# -*- coding: utf-8 -*-
'''
  Copyright (C) 2015 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
import os
import numpy as np

path = '/home/rafael/documents/doutorado/data_doc/003-Natan/2015-05-13'

ei_00_path = os.path.join(path,'eye_inspection_00.data')
ei_01_path = os.path.join(path,'eye_inspection_01.data')

ei_00 = np.genfromtxt(ei_00_path,delimiter="\t", dtype=None)
ei_01 = np.genfromtxt(ei_01_path,delimiter="\t", dtype=None)

c = [ei_00 == ei_01]
concordancy = (float(np.sum(c)))/len(c[0])
concordancy *= 100

# http://stackoverflow.com/questions/10741346/numpy-most-efficient-frequency-counts-for-unique-values-in-an-array
concordant_list = np.array([ei_00[i] for i, concordant in enumerate(c[0]) if concordant])

count = np.bincount(concordant_list)
unique = np.nonzero(count)[0]
print 'METADATA'
print 'Number of eye inspections (ei):', '2'
print 'Samples on each ei:',len(c[0])
print 'Used Samples:', len(concordant_list)
print 'Concordancy:', round(concordancy,2)
print 'Code legend:'
print '0=look at the expected stimulus'
print '1=look at the unexpected stimulus'
print '2=both'
print 'Counting from used samples (code, count):'
print zip(unique,count[unique])
