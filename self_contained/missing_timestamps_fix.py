# -*- coding: utf-8 -*-
'''
  Copyright (C) 2015 Rafael Picanço.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# fix missing stimulus onset (1, 1b, 2, 2b) on vlh (stimuli with variable limited hold durations) timestamps
# 
# s ------ 1 ------ 1b ------- 2 -------- 2b ----- 1 ------- (and so on...
# ----------- r*-c-- r*-c------r*-r-r-- ---r* --r-- r*-c-- (

import sys, os
import numpy as np
from ast import literal_eval

try:
    scapp_timestamps_path = sys.argv[0]
    scapp_vlh_report_path = sys.argv[1]
except:
    print "No arguments provided: 1) scapp output path 2) scapp report path"
    print "Using hardcoded names"

    try:
        scapp_timestamps_path = os.path.join(os.getcwd(),"scapp_output")
        if not os.path.isfile(scapp_timestamps_path):
            raise IOError, scapp_timestamps_path+" not found."

        scapp_vlh_report_path = os.path.join(os.getcwd(),"scapp_report")
        if not os.path.isfile(scapp_vlh_report_path):
            raise IOError, scapp_vlh_report_path+" not found."
    except IOError, e:
        print e, "using hardcoded paths"
        output_path = "/home/rafael/documents/doutorado/data_doc/005-Marco/2015-05-20/002"
        scapp_timestamps_path = os.path.join(output_path, "Data_002.timestamps")
        scapp_vlh_report_path = os.path.join(output_path, "Data_002.txt")

if not os.path.isfile(scapp_timestamps_path) or not os.path.isfile(scapp_vlh_report_path):
    raise IOError, "Source files were not found."
else:
    print "Using:"
    print "output:",scapp_timestamps_path
    print "report:",scapp_vlh_report_path

# attention to the 'skip' vars
scapp_report = np.genfromtxt(scapp_vlh_report_path,
    delimiter="\t", missing_values=["NA"], skip_header=9, skip_footer=1,
    filling_values=None, names=True, deletechars='_', autostrip=True,
    dtype=None)

# we need only the 'S' (start) event for the fix
with open(scapp_timestamps_path, 'r') as f:
    for line in f:
        (trial_no, timestamp, event) = literal_eval(line)
        break

# timestamp in float seconds from the start of pc
# event in milisecs from the start of the session
print timestamp, event

def get_condition(condition):
    if condition:
        return '1'
    else:
        return '2'

# find the missing timestamps from know events

output = []
trial = 1
output.append((trial, float(timestamp), '1'))
c = True
for n in scapp_report:
    try:
        miss_timestamp = ((int(n['Timer2']) - int(event.replace("S:","")))/1000)+float(timestamp)
        condition = get_condition(c)+"b"
        output.append((trial, miss_timestamp, condition))
        trial += 1
        c = not c
        condition = get_condition(c)
    except:
        # print "end"
        condition = get_condition(c)+"b"

    miss_timestamp = ((n['Cycle'] - int(event.replace("S:","")))/1000)+float(timestamp)
    output.append((trial, miss_timestamp, condition))

for line in output:
    print line
np.save(os.path.join(output_path,"scapp_output"), output)