# -*- coding: utf-8 -*-
'''
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import sys, os
import numpy as np
from itertools import islice

from fpe_drawing import draw_relative_rate

def load_fpe_data(path, skip_header):
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
        skip_header=skip_header,
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

def get_session_type(data):
    fp_strings = ['FP', 'Feature Positive']
    fn_strings = ['FN', 'Feature Negative']
    
    bloc_name = data['Bloc_Nam'][0].decode("utf-8")
    for fp_string in fp_strings:
        if bloc_name in fp_string:
            return 'feature positive'

    for fn_string in fn_strings:
        if bloc_name in fn_string:
            return 'feature negative'

    return None

def plot_fpfn(fp_paths, fn_paths):
    pass

def consecutive_hits(paths, skip_header=13):
    overall_performance = []
    count = 0
    for path in paths:
        count += 1
        data = load_fpe_data(path, skip_header)
        session_type = get_session_type(data)
        trials = data['Trial_Id'].max()
        performance = get_perfomance(data) 
        hits = performance.count('HIT') 

        print('Session %s'%count)
        print('Session Type: %s'%session_type)
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

    print('The participant DO NOT reached %s consecutive hits in %s trials.'%(size,count+size-1),'\n')
 
def get_events_per_trial_in_bloc(data, ts, target_bloc=2):
    session = zip(ts['Time'], ts['Bloc__Id'], ts['Trial_No'], ts['Event'])
    events_per_trial = {} 
    for time, bloc_id, trial, ev in session:
        event = ev.decode("utf-8")
        if bloc_id == target_bloc:
            if trial not in events_per_trial:
                events_per_trial[trial] = {'Type':'','Time':[],'Event':[]}
            
            if event == 'ITI R':
                events_per_trial[trial-1]['Time'].append(time)    
                events_per_trial[trial-1]['Event'].append(event)
            else:
                events_per_trial[trial]['Time'].append(time)    
                events_per_trial[trial]['Event'].append(event)

    session = zip(data['TrialNam'], data['Trial_No'])
    type1 = 'Positiva'
    type2 = 'Negativa'
    for trial_name, trial_number in session:
        name = trial_name.decode('utf-8')
        if type1 in name: 
            events_per_trial[trial_number]['Type'] = type1

        if type2 in name:
            events_per_trial[trial_number]['Type'] = type2

    return events_per_trial

def get_trial_starts(trials):
    starts = []
    for i, trial in trials.items():
        for time, event in zip(trial['Time'], trial['Event']):
            if event == 'TS':
                starts.append({'Time':time, 'Type':trial['Type']})

    last_type = list(trials.values())[-1]['Type']
    last_time = starts[-1]['Time']

    l = [end['Time'] - start['Time'] for start, end in zip(starts, starts[1:])] 
    last_time += np.mean(l) 
    
    starts.append({'Time':last_time, 'Type':last_type})
    return starts

def get_start_end_intervals_with_iti(starts):
    positive_intervals = []
    negative_intervals = []
    for start, end in zip(starts, starts[1:]):
        if start['Type'] == 'Positiva':
            positive_intervals.append([start['Time'], end['Time']])

        if start['Type'] == 'Negativa':
            negative_intervals.append([start['Time'], end['Time']])
        
    return positive_intervals, negative_intervals


def get_all_responses(ts):
    return [time for time, event in zip(ts['Time'], ts['Event']) \
        if event.decode('utf-8') == 'R' or 'ITI R']

def rate_in(time_interval_pairwise,timestamps):
    def is_inside(timestamps,rangein, rangeout):
        return [t for t in timestamps if (t >= rangein) and (t <= rangeout)]

    return [len(is_inside(timestamps, begin, end))/(end-begin) for begin, end in time_interval_pairwise]

def get_relative_rate(data1, data2):
    return [a/(b+a) if b+a > 0 else np.nan for a, b in zip(data1, data2)]


def rate(paths, skip_header=13):
    overall_performance = []
    count = 0
    for path in paths:
        data_file = load_fpe_data(path[0],skip_header)
        timestamps_file = load_fpe_timestamps(path[1])
        responses = get_all_responses(timestamps_file)
        trials = get_events_per_trial_in_bloc(data_file,timestamps_file)
        starts = get_trial_starts(trials)
        positive_intervals, negative_intervals = get_start_end_intervals_with_iti(starts)  
        positive_data = rate_in(positive_intervals,responses)
        negative_data = rate_in(negative_intervals,responses)
        relative_rate = get_relative_rate(positive_data, negative_data)
        title = path[0].replace('/home/pupil/recordings/DATA/','')
        title = title.replace('/stimulus_control/000.data','')
        title = title.replace('/','_')
        title = title+'_'+get_session_type(data_file)
        title = title.replace(' ', '_')
        draw_relative_rate(relative_rate,title, True)


def get_paths(paths):
    p = []
    for root in paths['root']:
        p.append([
            os.path.join(root, paths['file'][0]),
            os.path.join(root, paths['file'][1])])
    return p

if __name__ == '__main__':
    
    # paths = [
    #     "/home/pupil/recordings/DATA/2017_02_06/000_ELD/000/stimulus_control/000.data"
    # ]
    # main(paths)

    # paths = [
    #     "/home/pupil/recordings/DATA/2017_04_11/000_HER/001/stimulus_control/000.data" 
    # ]
    # consecutive_hits(paths, 14)

    # paths = [
    #     "/home/pupil/recordings/DATA/2017_04_12/000_ATL/000/stimulus_control/000.data",
    #     "/home/pupil/recordings/DATA/2017_04_12/000_ATL/001/stimulus_control/000.data"
    # ]
    # consecutive_hits(paths)

    # paths = [
    #     "/home/pupil/recordings/DATA/2017_04_29/000_DEM/000/stimulus_control/000.data",
    #     "/home/pupil/recordings/DATA/2017_04_29/000_DEM/001/stimulus_control/000.data",
    #     "/home/pupil/recordings/DATA/2017_04_29/000_DEM/002/stimulus_control/000.data"
    # ]
    # consecutive_hits(paths)

    # paths = [
    #     "/home/pupil/recordings/DATA/2017_04_29/000_JES/000/stimulus_control/000.data",
    #     "/home/pupil/recordings/DATA/2017_04_29/000_JES/001/stimulus_control/000.data"
    # ]
    # consecutive_hits(paths)

    # paths = [
    #     "/home/pupil/recordings/DATA/2017_04_29/000_JUL/000/stimulus_control/000.data"
    # ]

    # consecutive_hits(paths)


    d = {
        'root': [
            '/home/pupil/recordings/DATA/2017_04_11/000_HER/001/stimulus_control'
            ],
        'file': ['000.data', '000.timestamps']
        }

    rate(get_paths(d),14)

    d = {
        'root': [
            '/home/pupil/recordings/DATA/2017_04_12/000_ATL/000/stimulus_control',
            '/home/pupil/recordings/DATA/2017_04_12/000_ATL/001/stimulus_control'
            ],
        'file': ['000.data', '000.timestamps']
        }

    rate(get_paths(d))

    d = {
        'root': [
            '/home/pupil/recordings/DATA/2017_04_29/000_DEM/000/stimulus_control',
            '/home/pupil/recordings/DATA/2017_04_29/000_DEM/001/stimulus_control',
            '/home/pupil/recordings/DATA/2017_04_29/000_DEM/002/stimulus_control'

            ],
        'file': ['000.data', '000.timestamps']
        }

    rate(get_paths(d))


    d = {
        'root': [
            '/home/pupil/recordings/DATA/2017_04_29/000_JES/000/stimulus_control',
            '/home/pupil/recordings/DATA/2017_04_29/000_JES/001/stimulus_control'
        ],    
        'file': ['000.data', '000.timestamps']
    }

    rate(get_paths(d))

    d = {
        'root': [
            '/home/pupil/recordings/DATA/2017_04_29/000_JUL/000/stimulus_control'
            ],
        'file': ['000.data', '000.timestamps']
        }

    rate(get_paths(d))
