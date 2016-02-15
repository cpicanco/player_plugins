# -*- coding: utf-8 -*-
'''
  Copyright (C) 2015 François Tonneau.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import numpy as np

import cv2

fileName = "input.txt"
fileDelimiter = "\t"

xyData = np.loadtxt(fileName, delimiter = fileDelimiter)
xyData = np.float32(xyData)

xmax = 1280
ymax = 768

def screenClamped(xy):
    xgood = (0 <= xy[:, 0]) & (xy[:, 0] < xmax)
    ygood = (0 <= xy[:, 1]) & (xy[:, 1] < ymax)
    xyClamped = xy[xgood & ygood, :]
    deletedCount = xy.shape[0] - xyClamped.shape[0]
    if deletedCount > 0:
        print "\nWarning: Removed", deletedCount, "data point(s) with", \
        "out-of-screen coordinates"
    return xyClamped

xyData = screenClamped(xyData)

def xyBias(xyBlock, clustersExpected):
    iterationMax = 10
    epsilon = 1.0
    criteria = ( \
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, \
        iterationMax, \
        epsilon \
    )
    attempts = 10
    flags = cv2.KMEANS_RANDOM_CENTERS
    totalVariance, xyLabels, xyCenters = \
        cv2.kmeans(xyBlock, clustersExpected, None, criteria, attempts, flags)
    xyMeanCenter = xyCenters.mean(axis = 0)
    screenCenter = np.array([xmax/2.0, ymax/2.0])
    return xyMeanCenter - screenCenter

def correct(block, bias):
    block[:, 0] = block[:, 0] - bias[0]
    block[:, 1] = block[:, 1] - bias[1]
    return block

dataCount = xyData.shape[0]
print "\nThere are", dataCount, "data points."

blockSize = 1000

if dataCount < blockSize:
    print "Too few data to proceed."
    print "Press ENTER to exit."
    raw_input()
    quit()

equallySpacedLines = range(0, dataCount, blockSize)

firstBlock = True

for blockStart in equallySpacedLines:
    blockEnd = blockStart + blockSize
    if blockEnd <= dataCount:
        dataBlock = xyData[blockStart:blockEnd, :]
        dataBias = xyBias(dataBlock, 2)
    else:
        dataBlock = xyData[blockStart:dataCount, :]
    if firstBlock:
        biasAlongBlocks = dataBias
        unbiasedData = correct(dataBlock, dataBias)
        firstBlock = False
    else:
        biasAlongBlocks = np.vstack((biasAlongBlocks, dataBias))
        unbiasedData = np.vstack((unbiasedData, correct(dataBlock, dataBias)))

print "\nBias along blocks:"
print biasAlongBlocks

np.savetxt("bias.txt", biasAlongBlocks)
np.savetxt("output.txt", unbiasedData)