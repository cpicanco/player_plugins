# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

# plot timestamped events using a cumulative frequency graph

import sys, os
import numpy as np
from itertools import islice

#import matplotlib.pyplot as plt

def load_fpe_data(path):
    """
    Columns:
        Bloc__Id
        Bloc_Nam
        Trial_No
        Trial_Id
        TrialNam
        ITIBegin
        __ITIEnd
        StmBegin
        _Latency
        __StmEnd
        RespFreq
        ExpcResp
        __Result

    """
    if not os.path.isfile(path):
        raise "Path was not found:"+path

    data = np.genfromtxt(path,
        delimiter="\t",
        missing_values=["NA"],
        skip_header=13,
        skip_footer=1,
        filling_values=None,
        names=True,
        autostrip=True,
        dtype=None
    )
    return data

def load_fpe_timestamps(path):
    """
    Columns:
        Time
        Bloc__Id
        Trial_Id
        Trial_No
        Event
    """
    if not os.path.isfile(path):
        raise "Path was not found:"+path

    data = np.genfromtxt(path,
        delimiter="\t",
        missing_values=["NA"],
        skip_header=5,
        skip_footer=1,
        filling_values=None,
        names=True,
        autostrip=True,
        dtype=None
    )
    return data

def window(seq, n=2):
    """
    https://docs.python.org/release/2.3.5/lib/itertools-example.html
     "Returns a sliding window (of width n) over data from the iterable"
     "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def get_perfomance(data):
    return [line.decode("utf-8") for line in data['__Result']]

def main(paths):
    overall_performance = []
    count = 0
    for path in paths:
        count += 1
        data = load_fpe_data(path)
        trials = data['Trial_Id'].max()
        performance = get_perfomance(data) 
        hits = performance.count('HIT') 
        print('Session %s'%count)
        print('Trials:',trials)
        print('Hits:',hits, '(%s %%)'%round((hits/trials)*100,2),'\n')
        overall_performance += performance


    # consecutive hits
    size = 12
    count = 0
    for performance_chunk in window(overall_performance,size):
        count += 1
        if performance_chunk.count('HIT') == size:
            print('The participant reached %s consecutive hits in trial %s.'%(size,count+size-1),'\n')
            return

    print('The participant do not reached %s consecutive hits in %s trials.'%(size,count+size-1),'\n')
    



if __name__ == '__main__':
    
    paths = [
        "/home/pupil/recordings/DATA/2017_02_06/000_ELD/000/stimulus_control/000.data"
    ]
    main(paths)

    paths = [
        "/home/pupil/recordings/DATA/2017_04_12/000_ATL/000/stimulus_control/000.data",
        "/home/pupil/recordings/DATA/2017_04_12/000_ATL/001/stimulus_control/000.data"
    ]
    main(paths)

    paths = [
        "/home/pupil/recordings/DATA/2017_04_29/000_DEM/000/stimulus_control/000.data",
        "/home/pupil/recordings/DATA/2017_04_29/000_DEM/001/stimulus_control/000.data",
        "/home/pupil/recordings/DATA/2017_04_29/000_DEM/002/stimulus_control/000.data"
    ]
    main(paths)

    paths = [
        "/home/pupil/recordings/DATA/2017_04_29/000_JES/000/stimulus_control/000.data",
        "/home/pupil/recordings/DATA/2017_04_29/000_JES/001/stimulus_control/000.data"
    ]
    main(paths)

    paths = [
        "/home/pupil/recordings/DATA/2017_04_29/000_JUL/000/stimulus_control/000.data"
    ]
    main(paths)